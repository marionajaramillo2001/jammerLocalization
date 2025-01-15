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

from load_dataloader import Load_Dataloader

class FedAvg(object):
    def __init__(self, model, optimizer_nn, optimizer_theta, algorithm, mu, local_epochs, num_rounds,weight_ratio='sample_power',min_value=0.0, max_value=100., weight_method='bins', online=True):
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
        self.online = online

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
    
    def model_update(self, dataloader, bins=None, y_mean=None):
        model = copy.deepcopy(self.server_model)
        model.to(self.device)
        # optimizer = self.optimizer(model.parameters())
        optimizer_nn = self.optimizer_nn(model.model_NN.parameters())
        optimizer_theta = self.optimizer_theta(model.model_PL.parameters())
        scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=1, gamma=0.7)

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
                    for w in model.parameters():
                        l2_reg += torch.linalg.norm(w)**2
                    loss += self.mu*l2_reg
                optimizer_nn.zero_grad()
                optimizer_theta.zero_grad()
                loss.backward()
                optimizer_nn.step()
                optimizer_theta.step()
                # Step each scheduler
                # scheduler_theta.step()
                loss += loss.item()
                # with torch.no_grad():
                #     model.model_PL.theta.data = torch.clamp(model.model_PL.theta.data, self.min_value, self.max_value)
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
        train_loaders, test_loader_global,max_values_each_partition, train_y_mean_splited, bins,  train_x_max = data_splited
        train_samples = torch.zeros(len(train_loaders))
        for i in range(len(train_loaders)):
            train_samples[i]=len(train_loaders[i].dataset)
        # start training
        uploaded_weights = self.upload_weight(train_samples)
        if self.weight_ratio == 'sample_power':
            # uploaded_weights = uploaded_weights*y_power_ratios
            pass
        elif self.weight_ratio == 'power':
            # uploaded_weights = y_power_ratios
            pass
        elif self.weight_ratio == 'equal':
            uploaded_weights = torch.ones(len(uploaded_weights))/len(uploaded_weights)
        else:
            uploaded_weights = uploaded_weights
        global_loc = np.zeros(self.num_rounds)
        global_losses = torch.zeros(self.num_rounds)
        # create a empty dataframe to save the results
        df = pd.DataFrame(columns=['length', 'round', 'loss','loc_error'])
        # get the max batch length of each client
        max_batch_length = 0
        for i in range(len(train_loaders)):
            batch_length = len(train_loaders[i])
            if batch_length > max_batch_length:
                max_batch_length = batch_length
        if self.online == True:
            online_rounds = max_batch_length
        else:
            online_rounds = 1 # if not online, then only run one round
            print('Online_rounds:', online_rounds)
        # total_iter = len(online_rounds)*self.num_rounds
        n = 0
        for online_round in range(online_rounds):
            for round in range(self.num_rounds):
                model_list = []
                for i in range(len(train_loaders)):
                    # if the length of the dataloader is less than the max length, then use the length of the dataloader
                    if self.online == True:
                        if len(train_loaders[i]) < online_round+1:
                            length_for_client = len(train_loaders[i])
                        else:
                            length_for_client = online_round
                    else:
                        length_for_client = len(train_loaders[i])
                    # get the online batch data from 0 to j-th batch for each client
                    collected_data = []
                    collected_target = []
                    for j, (data_x, data_y) in enumerate(train_loaders[i]):
                        if j <= length_for_client:
                            collected_data.append(data_x)
                            collected_target.append(data_y)
                        else:
                            break
                    collected_data = torch.cat(collected_data, 0)
                    collected_target = torch.cat(collected_target, 0)
                    # create a new dataloader for each client
                    train_loader_new = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(collected_data, collected_target), batch_size=16, shuffle=True)
            
                    model, loss = self.model_update(train_loader_new, bins, train_y_mean_splited[i])
                    print('Client %d: train_loss = %f' % (i, loss))
                    model_list.append(model)
                
                self.server_model = self.server_update(model_list, uploaded_weights)
                # global_acc,global_loss = self.test(self.server_model, test_loader_global)
                global_loss = self.test_reg(self.server_model, test_loader_global)
                loc_error = self.test_loc(real_loc)
                print('Online_round %d, Round %d: test_loss = %f' % (online_round, round, global_loss))
                print('Online_round %d, Round %d:', (online_round, round, loc_error))
                df.loc[n] = [online_round, round, global_loss, loc_error]
                n += 1
                # global_losses[total_iter] = loc_error
                # global_loc[total_iter] = loc_error

        return df

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
