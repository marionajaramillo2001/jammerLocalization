import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as date
from apbm_v2.data_loader_APBM import data_process
from apbm_v2.train import Train
from functools import partial
import random
from apbm_v2.plots import *
from ray import tune, train
from ray.tune import loguniform, uniform
from apbm_v2.model import Net, Polynomial3


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
        'max_epochs_nn': config['max_epochs_nn'],
        'max_epochs_pl': config['max_epochs_pl'],
        'batch_size': config['batch_size']
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
    indices_folds, train_dataset, test_loader = d_p.split_dataset()

    # Define partial optimizers for NN and theta
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=0.0)
    optimizer_P0 = partial(optim.Adam, lr=config['lr_optimizer_P0'], weight_decay=0.0)
    optimizer_gamma = partial(optim.Adam, lr=config['lr_optimizer_gamma'], weight_decay=0.0)

    model_nn = Net(**model_nn_args)
    model_pl = Polynomial3(**model_pl_args)
    
    return model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, train_dataset, test_loader, alg_args
    
    
def train_test(config):
    model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p, train_dataset, test_loader, alg_args = prepare_data(config)
    
    true_jam_loc = d_p.trueJloc
    # Create CrossVal instance and train the model
    train = Train(model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    train_losses_nn_per_epoch, train_losses_pl_per_epoch, global_test_nn_loss, global_test_pl_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, trained_model_nn, trained_model_pl = train.train_test(train_dataset, test_loader, true_jam_loc)
    
    # nn loss
    plot_train_test_loss(train_losses_nn_per_epoch, global_test_nn_loss)
    
    # pl loss
    plot_train_test_loss(train_losses_pl_per_epoch, global_test_pl_loss)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    visualize_3d_model_output(trained_model_nn, trained_model_pl, train_loader, test_loader, true_jam_loc, predicted_jam_loc, None)
    
    return global_test_pl_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma


if __name__ == '__main__':
    
    id_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT32/obs_time_1/',
        # 'path': '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/datasets/dataPLANS/4.definitive/PL2/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'bins_num': 10,
        'runs': 1,
        'monte_carlo_runs': 1,
        'betas': True,
        'input_dim': 2,
        # 'layer_wid': [500, 1],
        # 'layer_wid': [128, 64, 32, 1],
        'layer_wid': [256, 128, 64, 1],
        # 'layer_wid': [512, 512, 1],
        'nonlinearity': 'leaky_relu',
        'gamma': 2,
        'max_epochs_nn': 50,
        'max_epochs_pl':50,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0,
    }

    config = id_0

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
            global_test_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma = train_test(config)
            
            # Accumulate results for this run
            r_mc_test_loss += global_test_loss
            r_mc_jam_loc_error += jam_loc_error

        # Compute average results for this run
        r_mc_test_loss /= config['monte_carlo_runs']
        r_mc_jam_loc_error /= config['monte_carlo_runs']

        # Print averaged results for this run
        print(f"Run {r + 1} results:")
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