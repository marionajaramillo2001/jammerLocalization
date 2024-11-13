# Federated learning with FedAvg algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from load_dataloader import Load_Dataloader

class CrossVal(object):
    def __init__(self, cv_model, batch_size, optimizer_nn, optimizer_theta, mu, max_epochs, betas=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crossval_model = cv_model
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.criterion = nn.MSELoss()
        self.mu = mu
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_rounds = 1 # in a non FL this is 1 (I kept it for the betas to make sense)
        if betas is True:
            self.betas = [10*(1-(i+1)/self.num_rounds) for i in range(self.num_rounds)]
        else:
            self.betas = [0 for i in range(self.num_rounds)]

    def loss_func(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)
    
    def model_update(self, dataloader, bins=None, y_mean=None, round=None):
        model = copy.deepcopy(self.server_model)
        model.to(self.device)
        optimizer_nn = self.optimizer_nn(model.model_NN.parameters())
        optimizer_theta = self.optimizer_theta(model.model_PL.parameters())

        model.train()
        loss = 0
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = self.loss_func(output, target)
                optimizer_nn.zero_grad()
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_nn.step()
                optimizer_theta.step()
                loss += loss.item()
        return model, loss / self.local_epochs

    def train(self, indices_folds, crossval_dataset, test_loader, real_loc):
        fold_iter = 0
        for fold in indices_folds:
            print(f"Fold {fold_iter}")
            print("-------")

            # Define the data loaders for the current fold
            # dataloader contains the batches, each batch is of the form ((sequences_padded, bwga), lengths, labels)
                    
            train_loader = DataLoader(
                dataset=crossval_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(fold["train_index"]),
            )

            val_loader = DataLoader(
                dataset=crossval_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(fold["val_index"]),
            )
            
            self.crossval_model.to(self.device)
            
            # When we pass model.parameters() to an optimizer (torch.optim.Adam in this case), 
            # the optimizer is informed about which parameters should be updated during training.
            optimizer_nn = self.optimizer_nn(self.crossval_model.model_NN.parameters())
            optimizer_theta = self.optimizer_theta(self.crossval_model.model_PL.parameters())
        
            train_losses = np.zeros(self.max_epochs)
            val_losses = np.zeros(self.max_epochs)

            for epoch_i in tqdm(range(self.max_epochs)):
                # =======================================
                #               Training
                # =======================================

                # Tracking time and loss
                total_loss = 0
                train_loss = []

                # Put the model into the training mode
                self.crossval_model.train()
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.crossval_model(data)
                    loss = self.loss_func(output, target)
                    optimizer_nn.zero_grad()
                    optimizer_theta.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_nn.step()
                    optimizer_theta.step()
                    total_loss += loss.item()

                    train_loss.append(loss.item())


                train_loss = np.mean(train_loss)
                train_losses[epoch_i] = train_loss

                # =======================================
                #               Evaluation
                # =======================================
                if val_loader is not None:
                    # After the completion of each training epoch, measure the model's
                    # performance on our validation set.
                    val_loss = self.test_reg(self.crossval_model, val_loader)

                    val_losses[epoch_i] = val_loss

            fold_iter += 1   
            
            
        global_test_loss = self.test_reg(self.crossval_model, test_loader)
    
        return train_losses, val_losses, global_test_loss

    
    def test_reg(self, model, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.criterion(output, target).item()
        test_loss /= len(test_loader.dataset)
        return test_loss
    
    def test_loc(self, real_loc):
        result = np.sqrt(np.mean((real_loc - self.server_model.get_theta().detach().numpy())**2))
        return result

    def test_field(self, model, t):
        model.eval()
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        with torch.no_grad():
            for i in range(len(X)):
                for j in range(len(Y)):
                    Z[i, j] = model(torch.tensor([[X[i, j], Y[i, j]]]).float()).item()
        fig = go.Figure(data=[go.Surface(z=Z,x=X, y=Y)])
        fig.update_layout(title='3D Surface Plot', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))
        fig.write_html(f'figs/field_{t}.html')
        

        
            