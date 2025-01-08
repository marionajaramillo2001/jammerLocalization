import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as date
from apbm.data_loader_APBM import data_process
from apbm.crossval import CrossVal
from functools import partial
import random
from apbm.plots import *
from ray import tune, train
from ray.tune import loguniform, uniform
from apbm.model import Net_augmented


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
        'bins_num': config['bins_num']
    }
    
    alg_args = {
        'max_epochs': config['max_epochs'],
        'patience': config['patience'],
        'early_stopping': config['early_stopping'],
        'batch_size': config['batch_size'],
        'mu': config['mu'],
        'betas': config['betas']
    }
    
    model_args = {
        'input_dim': config['input_dim'],
        'layer_wid': config['layer_wid'],
        'nonlinearity': config['nonlinearity'],
        'gamma': config['gamma'],
        'model_mode': config['model_mode'],  # Options: 'NN', 'PL', 'both'
    }
    
    # Data processing step to load and split the dataset
    d_p = data_process(**data_args)
    indices_folds, crossval_dataset, test_loader = d_p.split_dataset()

    # Define partial optimizers for NN and theta
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=0.0)
    optimizer_P0 = partial(optim.Adam, lr=config['lr_optimizer_P0'], weight_decay=0.0)
    optimizer_gamma = partial(optim.Adam, lr=config['lr_optimizer_gamma'], weight_decay=0.0)

    model = Net_augmented(**model_args)
    
    return model, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, indices_folds, crossval_dataset, test_loader, alg_args
    
def hypermarameter_tuning(config):
    model, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, indices_folds, crossval_dataset, test_loader, alg_args = prepare_data(config)
    
    # Create CrossVal instance and train the model
    crossval = CrossVal(model, config['theta_init'], optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    _, _, last_val_loss_mean_across_folds, _ = crossval.train_crossval(indices_folds, crossval_dataset)
    
    return {"last_val_loss_mean_across_folds": last_val_loss_mean_across_folds}

def crossval(config):
    model, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, indices_folds, crossval_dataset, test_loader, alg_args = prepare_data(config)
    
    # Create CrossVal instance and train the model
    crossval = CrossVal(model, config['theta_init'], optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    all_train_losses_per_fold, all_val_losses_per_fold, last_val_loss_mean_across_folds, mean_best_epoch = crossval.train_crossval(indices_folds, crossval_dataset)
    
    # Plot average training and validation losses for visual analysis
    plot_average_train_val_loss(all_train_losses_per_fold, all_val_losses_per_fold)
    return all_train_losses_per_fold, all_val_losses_per_fold, last_val_loss_mean_across_folds, mean_best_epoch
    
def train_test(config):
    model, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, _, train_dataset, test_loader, alg_args = prepare_data(config)
    
    true_jam_loc = d_p.trueJloc
    # Create CrossVal instance and train the model
    train = CrossVal(model, config['theta_init'], optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    train_losses_per_epoch, global_test_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, trained_model = train.train_test(train_dataset, test_loader, true_jam_loc)
    
    plot_train_test_loss(train_losses_per_epoch, global_test_loss)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    visualize_3d_model_output(trained_model, train_loader, test_loader, true_jam_loc, predicted_jam_loc, None)
    
    return global_test_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma


if __name__ == '__main__':
        
    initial_config_RT2_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT12/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'PL',
        'max_epochs': 100,
        'batch_size': 16,
        'lr_optimizer_nn': 0.1,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-5,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': False,
        'hyperparameter_tuning': False
    }
    
    best_config_PL2_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/PL2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-07,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT2_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-07,
        'weight_decay_optimizer_theta': 1e-08,
        'mu': 0.1,
        'patience': 40,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT2_1 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'random',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'NN',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-07,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT2_2 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'PL',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-07,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 10,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT12_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT12/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT12_0_noise = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT12/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 0,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
        
    best_config_RT12_1 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT12/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'NN',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-10,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 40,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT12_2 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT12/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'PL',
        'max_epochs': 150,
        'batch_size': 16,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-07,
        'mu': 0.1,
        'patience': 10,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT13_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/RT13/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT14_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT14/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT15_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT15/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 32,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 10,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT16_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT16/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 32,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 10,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT18_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT18/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 32,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'weight_decay_optimizer_nn': 1e-05,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 10,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT19_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT19/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT20_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT20/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 16,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-09,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT20_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT20/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 16,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-09,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT21_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT21/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT21_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT21/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.001,
        'weight_decay_optimizer_nn': 1e-08,
        'weight_decay_optimizer_theta': 1e-05,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT20_0_opt = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT20/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.1,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.1,
        'lr_optimizer_gamma': 0.001,
        'weight_decay_optimizer_nn': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT22_0_opt = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT22/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.001,
        'lr_optimizer_gamma': 0.1,
        'weight_decay_optimizer_nn': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT23_0_opt = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT23/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'lr_optimizer_P0': 0.0001,
        'lr_optimizer_gamma': 0.1,
        'weight_decay_optimizer_nn': 1e-07,
        'mu': 0.1,
        'patience': 15,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    best_config_RT23_0_opt = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT23/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 4,
        'lr_optimizer_nn': 0.01,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.001,
        'weight_decay_optimizer_nn': 1e-06,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    best_config_RT23_0_opt_2 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT23/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.001,
        'lr_optimizer_P0': 0.0001,
        'lr_optimizer_gamma': 0.1,
        'weight_decay_optimizer_nn': 1e-10,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': True
    }
    
    id_25 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT18/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'max_loc',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [500, 200, 50, 1],
        'nonlinearity': 'tanh',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 150,
        'batch_size': 32,
        'lr_optimizer_nn': 0.0001,
        'lr_optimizer_theta': 0.01,
        'lr_optimizer_P0': 0.1,
        'lr_optimizer_gamma': 0.1,
        'weight_decay_optimizer_nn': 1e-06,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
    
    id_26 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT29/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'theta_init': 'mean_max_n_random',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [128, 64, 32, 16, 1],
        'nonlinearity': 'softplus',
        'gamma': 2,
        'model_mode': 'both',
        'max_epochs': 200,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.01,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0.0,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': False,
        'hyperparameter_tuning': False
    }
    
    id_27 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/PL2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 1,
        'bins_num': 10,
        'theta_init': 'fix',
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [128, 64, 32, 16, 1],
        'nonlinearity': 'relu',
        'gamma': 3,
        'model_mode': 'both',
        'max_epochs': 200,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.001,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0.0,
        'mu': 0.1,
        'patience': 30,
        'early_stopping': True,
        'hyperparameter_tuning': False
    }
        
    # Configuration dictionary
    search_space = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT18/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        "theta_init": tune.choice(['fix']),  # Initialization method for theta
        "runs": 1,  # Number of complete training runs
        "monte_carlo_runs": 1,  # Number of Monte Carlo simulations per run
        "betas": True,  # Whether to use betas for training
        'input_dim': 2,
        'layer_wid': [64, 64, 32, 16, 1],
        'nonlinearity': tune.choice(['relu', 'tanh', 'softplus']),
        'gamma': 2,
        'model_mode': 'both',  # Options: 'NN', 'PL', 'both'
        "max_epochs": 150,  # Maximum number of training epochs
        "batch_size": tune.choice([4, 8, 16, 32]),  # Batch size for training
        "lr_optimizer_nn": tune.grid_search([0.001, 0.01, 0.1]),
        "lr_optimizer_theta": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        'lr_optimizer_P0': tune.grid_search([0.0001, 0.001, 0.01, 0.1]),
        'lr_optimizer_gamma': tune.grid_search([0.001, 0.01, 0.1]),
        "weight_decay_optimizer_nn": tune.grid_search([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]),  # Weight decay for regularization (NN)
        "mu": 0.1,  # Additional hyperparameter
        "patience": tune.choice([15, 30]),  # Patience for early stopping
        "early_stopping": True,  # Whether to enable early stopping
        'hyperparameter_tuning': True
    }

    
    config = id_26

    # Initialize accumulators for overall averages across runs
    total_test_loss = 0
    total_jam_loc_error = 0

    # Run experiments and print results
    for r in range(config['runs']):
        # Initialize accumulators for this run
        config['time_t'] = r # TODO: I dont understand
        r_mc_test_loss = 0
        r_mc_jam_loc_error = 0

        print(f"\nStarting run {r + 1}/{config['runs']}:")

        for m in range(config['monte_carlo_runs']):
            print(f"  Monte Carlo run {m + 1}/{config['monte_carlo_runs']}...")
            
            if config['hyperparameter_tuning']:
                # Perform hyperparameter tuning using Ray Tune; config changes to the best config found
                tuner = tune.Tuner(hypermarameter_tuning, param_space=config)

                results = tuner.fit()
                config = results.get_best_result(metric="last_val_loss_mean_across_folds", mode="min").config
                print(config)
            
            all_train_losses_per_fold, all_val_losses_per_fold, last_val_loss_mean_across_folds, mean_best_epoch = crossval(config)

            # Change max_epochs to the mean of the best epochs across folds
            config['max_epochs'] = mean_best_epoch
            
            global_test_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma = train_test(config)
            
            # Accumulate results for this run
            r_mc_test_loss += global_test_loss
            r_mc_jam_loc_error += jam_loc_error

        # Compute average results for this run
        r_mc_test_loss /= config['monte_carlo_runs']
        r_mc_jam_loc_error /= config['monte_carlo_runs']

        # Print averaged results for this run
        print(f"Run {r + 1} results:")
        print(f"  Average last validation loss across folds: {last_val_loss_mean_across_folds:.4f}")
        print(f"  Average global test loss: {r_mc_test_loss:.4f}")
        print(f"  Jammer localization error: {r_mc_jam_loc_error:.4f}\n")
        print(f"  Predicted jammer location: {predicted_jam_loc}")
        print(f"  Real jammer location: {true_jam_loc}")
        print(f"  Learnt P0: {learnt_P0}")
        print(f"  Learnt gamma: {learnt_gamma}")

        # Accumulate for final average across all runs
        total_test_loss += r_mc_test_loss
        total_jam_loc_error += r_mc_jam_loc_error

    # Compute final average results across all runs
    final_avg_test_loss = total_test_loss / config['runs']
    final_avg_jam_loc_error = total_jam_loc_error / config['runs']

    # Print final results
    print("Final results across all runs:")
    print(f"  Final average global test loss: {final_avg_test_loss:.4f}")
    print(f"  Final average jammer localization error: {final_avg_jam_loc_error:.4f}")

    # Save results to the specified path
    output_path = 'results7/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)