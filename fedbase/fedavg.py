# Federated learning with FedAvg algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
import numpy as np
import pandas as pd
import time
import os
import sys
import argparse
import logging
from joblib import Parallel, delayed
from joblib import parallel_backend
# from tensorboardX import SummaryWriter
# from pytorch_optimizer import Lion
import plotly.graph_objects as go

from load_dataloader import Load_Dataloader

class FedAvg(object):
    def __init__(self, model, optimizer_nn, optimizer_theta, algorithm, mu, local_epochs, num_rounds,weight_ratio='sample_power',min_value=0.0, max_value=100., weight_method='bins', betas=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.server_model = model
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.criterion = nn.MSELoss()
        self.algorithm = algorithm
        self.mu = mu
        self.local_epochs = local_epochs
        self.num_rounds = num_rounds
        self.weight_ratio = weight_ratio
        self.min_value = min_value
        self.max_value = max_value
        self.weight_method = weight_method
        if betas is True:
            # from 100 to 0, divide by local_epochs for epochs
            self.betas = [10*(1-(i+1)/self.num_rounds) for i in range(self.num_rounds)]
        else:
            self.betas = [0 for i in range(self.num_rounds)]

    def weight_loss(self, y_train, bins=None, y_mean=None):
        if self.weight_method == 'batch_power_mean':
            y_train_mean = torch.mean(y_train)
            weights = y_train_mean/y_train
        elif self.weight_method == 'global_power_mean':
            weights = -20/y_train
        elif self.weight_method == 'batch_power_mean2':
            y_train_mean = torch.mean(y_train)
            weights=torch.pow(y_train_mean/y_train,2)
        elif self.weight_method == 'bins':
            y_binned = np.digitize(y_train.view(-1).detach().numpy(), bins)
            weights = 1/np.bincount(y_binned)[y_binned]
            weights = torch.tensor(weights)
        elif self.weight_method == 'local_power_mean':
            weights = y_mean/y_train
        else:
            weights = torch.ones(len(y_train))
        return weights

    def loss_func(self, y_pred, y_true, weights):
        # loss function
        # y_pred: predicted value
        # y_true: true value
        # return: loss
        
        return torch.mean(weights * (y_pred - y_true) ** 2)
    
    def model_update(self, dataloader, bins=None, y_mean=None, round=None):
        model = copy.deepcopy(self.server_model)
        model.to(self.device)
        # optimizer = self.optimizer_nn([model.w])
        # optimizer = self.optimizer_nn(model.parameters())
        optimizer_nn = self.optimizer_nn(model.model_NN.parameters())
        optimizer_theta = self.optimizer_theta(model.model_PL.parameters())
        # scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=1, gamma=0.7)

        model.train()
        loss = 0
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                # loss = self.criterion(output, target)
                weights = self.weight_loss(target, bins, y_mean)
                loss = self.loss_func(output, target, weights)
                # loss=self.criterion(output, target)
                # set if fedprox
                if self.algorithm == 'Fedprox':
                # for name, param in model.named_parameters():
                #         loss += self.mu * torch.norm(param - self.server_model.state_dict()[name]) ** 2/2
                    for w, w_t in zip(model.parameters(), self.server_model.parameters()):
                        loss += self.mu/2*(w - w_t).norm(2)**2
                elif self.algorithm == 'Fedprox-PL':
                    l2_reg = 0
                    for w in model.model_NN.parameters():
                        l2_reg += self.betas[round]*torch.linalg.norm(w)**2
                    loss += self.mu*l2_reg
                optimizer_nn.zero_grad()
                optimizer_theta.zero_grad()
                # optimizer.zero_grad()
                loss.backward()
                optimizer_nn.step()
                optimizer_theta.step()
                # optimizer.step()
                # Step each scheduler
                # scheduler_theta.step()
                loss += loss.item()
                # with torch.no_grad():
                #     model.model_PL.theta.data = torch.clamp(model.model_PL.theta.data, self.min_value, self.max_value)
                #     model.w.data = torch.clamp(model.w.data, 0, 1)
            # self.writer.add_scalar(f'loss/local_train_loss_epoch_client{client_idx}', loss, epoch)
        return model, loss / self.local_epochs

    def server_update(self, model_list, uploaded_weights):
        # update server model with FedAvg algorithm
        # model_list: list of models
        # set the server model to 0
        with torch.no_grad():
            for name, param in self.server_model.model_NN.named_parameters():
                param.data.zero_()
            self.server_model.model_PL.get_theta().data.zero_()
        
        for w, client_model in zip(uploaded_weights, model_list):
            for server_param, client_param in zip(self.server_model.model_NN.parameters(), client_model.model_NN.parameters()):
                server_param.data += client_param.data * w

            # self.server_model.model_PL.get_theta().data += client_model.model_PL.get_theta().data * w
            self.server_model.model_PL.theta.data += client_model.model_PL.theta.data * w
            self.server_model.w.data += client_model.w.data * w
        return self.server_model
    

    def upload_weight(self, train_samples):
        # update weight of each client
        # train_samples: list of number of samples of each client
        # return: list of weight of each client
        ws = []
        for i in range(len(train_samples)):
            ws.append(train_samples[i] / sum(train_samples))
        ws = torch.tensor(ws)
        return ws

    def train(self, data_splited, real_loc):
        # train the server model
        # train_loaders: list of dataloader of each client
        # test_loader: dataloader of test dataset
        # return: server model
        # train_loaders, test_loaders, test_loader_global, y_power_ratios = Load_Dataloader(self.args)
        train_loaders, test_loader_global, max_values_each_partition, train_y_mean_splited, bins,  train_x_max, train_y_max = data_splited
        train_samples = torch.zeros(len(train_loaders))
        for i in range(len(train_loaders)):
            train_samples[i]=len(train_loaders[i].dataset)
        # start training
        uploaded_weights = self.upload_weight(train_samples)
        if self.weight_ratio == 'mean':
            # normalized the weight based on the train_y_mean_splited
            uploaded_weights = uploaded_weights*((1/torch.tensor(train_y_mean_splited))/torch.sum(1/torch.tensor(train_y_mean_splited)))
        elif self.weight_ratio == 'power':
            uploaded_weights = uploaded_weights*((10**torch.tensor(train_y_mean_splited))/torch.sum(10**torch.tensor(train_y_mean_splited)))
        if self.weight_ratio == 'mean_max':
            # normalized the weight based on the train_y_mean_splited
            uploaded_weights = uploaded_weights*((1/(torch.tensor(train_y_mean_splited)-train_y_max))/torch.sum(1/(torch.tensor(train_y_mean_splited)-train_y_max)))
        elif self.weight_ratio == 'power_max':
            uploaded_weights = uploaded_weights*((10**(torch.tensor(train_y_mean_splited)-train_y_max))/torch.sum(10**(torch.tensor(train_y_mean_splited)-train_y_max)))
        elif self.weight_ratio == 'equal':
            uploaded_weights = torch.ones(len(uploaded_weights))/len(uploaded_weights)
        else:
            uploaded_weights = uploaded_weights
        global_loc = np.zeros(self.num_rounds)
        global_losses = torch.zeros(self.num_rounds)
        # create a empty dataframe to save the results
        df = pd.DataFrame(columns=['round', 'loss','loc_error'])
        for round in range(self.num_rounds):
            model_list = []
            for i in range(len(train_loaders)):
                model, loss = self.model_update(train_loaders[i], bins, train_y_mean_splited[i],round)
                print('Client %d: train_loss = %f' % (i, loss))
                model_list.append(model)
                # test the model using the local dataset
                # local_test_acc_p, local_test_loss_p = self.test(model, test_loaders[i])
                # local_test_loss_p = self.test_reg(model, test_loaders[i])
                # print('Client %d: test_loss = %f' % (i, local_test_loss_p))
            # parallel training
            # with parallel_backend('multiprocessing'):
            #     results = Parallel(n_jobs=self.args.num_workers)(delayed(self.model_update)(train_loaders[i]) for i in range(len(train_loaders)))
            # get the model and loss
            # for i in range(len(train_loaders)):
            #     model_list.append(results[i][0])
            #     print('Client %d: train_loss = %f' % (i, results[i][1]))
                # test the model using the local dataset
                # local_test_acc_p, local_test_loss_p = self.test(model, test_loaders[i])
                # local_test_loss_p = self.test_reg(model_list[i], test_loaders[i])
                # print('Client %d: test_loss = %f' % (i, local_test_loss_p))

            self.server_model = self.server_update(model_list, uploaded_weights)
            # global_acc,global_loss = self.test(self.server_model, test_loader_global)
            global_loss = self.test_reg(self.server_model, test_loader_global)
            loc_error = self.test_loc(real_loc)
            print('Round %d: test_loss = %f' % (round, global_loss))
            print('Round %d:', (round, loc_error))
            print('Round %d: weight of path loss', (round, self.server_model.w.data))
            df.loc[round] = [round, global_loss, loc_error]
            # global_losses[round] = global_loss
            # global_loc[round] = loc_error
            # plot the field of the model if round %10 == 0
            if round % (self.num_rounds/10) == 0:
                self.test_field(self.server_model, round)
        # return self.server_model
        # # save the results
        # path = self.args.save_dir
        # if os.path.exists(path) == False:
        #     os.mkdir(path)
        # df.to_csv(f'{path}{self.args.dataset[-4:]}_{self.args.algorithm}_results.csv', index=False)

        return df #global_losses, global_loc

    # define test function output the accuracy and loss of the model
    def test(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        return 100*correct / len(test_loader.dataset), test_loss
    
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
        # self.server_model.get_theta()
        # abs = np.abs(real_loc - self.server_model.get_theta().detach().numpy())
        result = np.sqrt(np.mean((real_loc - self.server_model.get_theta().detach().numpy())**2))
        return result

    def test_field(self, model, t):
        # plot the field of the model
        # model: the model
        # return the figures of the field
        model.eval()
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        with torch.no_grad():
            for i in range(len(X)):
                for j in range(len(Y)):
                    Z[i, j] = model(torch.tensor([[X[i, j], Y[i, j]]]).float()).item()
        # plot the field value
        fig = go.Figure(data=[go.Surface(z=Z,x=X, y=Y)])
        fig.update_layout(title='3D Surface Plot', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))
        fig.write_html(f'figs/field_{t}.html')
        
        

        
            