import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as date
from apbm_v2_fed.data_loader_APBM import data_process
from apbm_v2_fed.fedavg import FedAvg
from functools import partial
import random
from apbm_v2_fed.plots import *
from ray import tune, train
from ray.tune import loguniform, uniform
from apbm_v2_fed.model import Net, Polynomial3


# Set random seed for reproducibility
seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Set the current path as the working directory
os.chdir(os.getcwd())

def prepare_data(config):
    data_args = {
        'path': config['path'],
        'time_t': config['time_t'],
        'test_ratio': config['test_ratio'],
        'data_preprocessing': config['data_preprocessing'],
        'noise': config['noise'],  # add noise here, if data comes from MATLAB w/out noise
        'noise_std': config['noise_std'],
        'batch_size': config['batch_size'],
    }
    
    alg_args = {
        'batch_size': config['batch_size'],
        'local_epochs_nn': config['local_epochs_nn'],
        'local_epochs_pl': config['local_epochs_pl'],
        'num_rounds_nn': config['num_rounds_nn'],
        'num_rounds_pl': config['num_rounds_pl'],
    }
    
    model_nn_args = {
        'input_dim': config['input_dim'],
        'layer_wid': config['layer_wid'],
        'nonlinearity': config['nonlinearity'],
    }
    
    model_pl_args = {
        'gamma': config['gamma'],
    }
    
    # Data processing step to load and split the dataset
    d_p = data_process(**data_args)
    train_loader_splited, test_loader, max_values_each_partition, train_y_mean_splited, train_x_max, train_y_max = d_p.split_dataset(config['num_nodes'])

    # Define partial optimizers for NN and theta
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=0.0)
    optimizer_P0 = partial(optim.Adam, lr=config['lr_optimizer_P0'], weight_decay=0.0)
    optimizer_gamma = partial(optim.Adam, lr=config['lr_optimizer_gamma'], weight_decay=0.0)

    model_nn = Net(**model_nn_args)
    model_pl = Polynomial3(**model_pl_args)
    
    return model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p.trueJloc,train_loader_splited, test_loader, train_y_mean_splited, alg_args
    
    
def train_test(config):
    model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, true_jam_loc, train_loader_splited, test_loader, train_y_mean_splited, alg_args = prepare_data(config)
    
    # Create CrossVal instance and train the model
    train = FedAvg(model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    train_losses_nn_per_round, train_losses_pl_per_round, global_test_nn_loss, global_test_pl_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, trained_model_nn, trained_model_pl = train.train_test(train_loader_splited, test_loader, true_jam_loc, train_y_mean_splited)
    
    # nn loss
    plot_train_test_loss(train_losses_nn_per_round, global_test_nn_loss)
    
    # pl loss
    plot_train_test_loss(train_losses_pl_per_round, global_test_pl_loss)
    
    visualize_3d_model_output(trained_model_nn, trained_model_pl, train_loader_splited, test_loader, true_jam_loc, predicted_jam_loc, None)
    
    return global_test_pl_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma


if __name__ == '__main__':
    
    id_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT27/obs_time_1/',
        # 'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/PL2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'betas': True,
        'input_dim': 2,
        # 'layer_wid': [500, 1],
        # 'layer_wid': [128, 64, 32, 1],
        'layer_wid': [256, 128, 64, 1],
        # 'layer_wid': [512, 512, 1],
        'nonlinearity': 'leaky_relu',
        'gamma': 2,
        'num_nodes': 10,
        'local_epochs_nn': 50,
        'local_epochs_pl': 50,
        'num_rounds_nn': 30,
        'num_rounds_pl': 50,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0,
    }

    config = id_0
    global_test_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma = train_test(config)

    # Print averaged results for this run
    print(f"  Global test loss: {global_test_loss:.4f}")
    print(f"  Jammer localization error: {jam_loc_error:.4f}\n")
    print(f"  Predicted jammer location: {predicted_jam_loc}")
    print(f"  Real jammer location: {true_jam_loc}")
    print(f"  Learnt P0: {learnt_P0}")
    print(f"  Learnt gamma: {learnt_gamma}")