import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime as date
from apbm.data_loader_APBM import data_process
from apbm.crossval import CrossVal
from apbm.model import Net_augmented
from functools import partial
import random
from apbm.plots import *

# Set random seed for reproducibility
seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Set the current path as the working directory
os.chdir(os.getcwd())

def main(data_args, model_args, alg_args, theta_init):
    # Data processing step to load and split the dataset
    d_p = data_process(**data_args)
    indices_folds, crossval_dataset, test_loader = d_p.split_dataset()

    # Identify the position with the highest y (received signal strength) for 'max_loc' initialization
    if theta_init == 'max_loc':
        data_loader = DataLoader(crossval_dataset, batch_size=len(crossval_dataset))
        for data, target in data_loader:
            max_index = torch.argmax(target)  # Find the index of the max signal strength
            max_position = data[max_index]    # Get the corresponding position
            break  # Only need one pass over the data

        model = Net_augmented(**model_args, theta0=max_position + torch.randn(2))  # Optionally add noise

    elif theta_init == 'None':
        # Initialize the model without a specific theta0
        model = Net_augmented(**model_args)
    elif theta_init == 'random':
        # Random initialization of theta0 within a range
        model = Net_augmented(**model_args, theta0=99 * torch.rand(2))
    elif theta_init == 'fix':
        # Fixed theta0 initialization at a specific point
        model = Net_augmented(**model_args, theta0=torch.tensor([50.0, 50.0]))

    # Create CrossVal instance and train the model
    crossval = CrossVal(model, **alg_args)
    all_train_losses_per_fold, all_val_losses_per_fold, global_test_loss, jam_loc_error = crossval.train(
        indices_folds, crossval_dataset, test_loader, d_p.trueJloc
    )

    # Plot average training and validation losses for visual analysis
    plot_average_losses(all_train_losses_per_fold, all_val_losses_per_fold)

    return all_train_losses_per_fold, all_val_losses_per_fold, global_test_loss, jam_loc_error

if __name__ == '__main__':
    # Set data path based on the data_name variable
    data_name = 'RT2'
    if data_name == 'RT2':  # Smaller dataset (e.g., 100 samples)
        path = 'datasets/dataPLANS/4.definitive/RT2/'
    elif data_name == 'PL2':  # Larger dataset (e.g., 10000 samples)
        path = 'datasets/dataPLANS/4.definitive/PL2/'
    elif data_name == 'R11':
        path = 'datasets/dataPLANS/4.definitive/RT11/'
    elif data_name == 'PL11':
        path = 'datasets/dataPLANS/4.definitive/PL11/'
    elif data_name == 'RT12':
        path = 'datasets/dataPLANS/4.definitive/RT12/'
    elif data_name == 'RT13':
        path = 'datasets/dataPLANS/4.definitive/RT13/'

    # Configuration dictionary
    config = {
        "theta_init": 'max_loc',  # Options: 'fix', 'random', 'max_loc', 'None'
        "runs": 1,  # Number of complete training runs
        "monte_carlo_runs": 1,  # Number of Monte Carlo simulations per run
        "betas": True,  # Whether to use betas for training
        "model_args": {
            'input_dim': 2,
            'layer_wid': [500, 1],
            'nonlinearity': 'relu',
            'gamma': 2,
            'model_mode': 'NN',  # Options: 'NN', 'PL', 'both'
        },
        "max_epochs": 200,  # Maximum number of training epochs
        "batch_size": 8,  # Batch size for training
        "lr_optimizer_nn": 0.01,  # Learning rate for NN optimizer
        "lr_optimizer_theta": 0.01,  # Learning rate for theta optimizer
        "weight_decay_optimizer_nn": 1e-10,  # Weight decay for regularization (NN)
        "weight_decay_optimizer_theta": 1e-10,  # Weight decay for regularization (theta)
        "mu": 0.1,  # Additional hyperparameter
        "patience": 30,  # Patience for early stopping
        "early_stopping": True,  # Whether to enable early stopping
    }

    # Define partial optimizers for NN and theta
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=config['weight_decay_optimizer_theta'])

    # Algorithm arguments passed to CrossVal
    alg_args = {
        'max_epochs': config['max_epochs'],
        'patience': config['patience'],
        'early_stopping': config['early_stopping'],
        'batch_size': config['batch_size'],
        'optimizer_nn': optimizer_nn,
        'optimizer_theta': optimizer_theta,
        'mu': config['mu'],
        'betas': config['betas']
    }
    model_args = config['model_args']

    # Initialize accumulators for overall averages across runs
    total_test_loss = 0
    total_jam_loc_error = 0

    # Run experiments and print results
    for r in range(config['runs']):
        data_args = {
            'path': path,
            'time_t': r,
            'test_ratio': 0.2,
            'data_preprocesing': 1,
            'batch_size': config['batch_size'],
            'bins_num': 10
        }

        # Initialize accumulators for this run
        r_mc_test_loss = 0
        r_mc_jam_loc_error = 0

        print(f"\nStarting run {r + 1}/{config['runs']}:")

        for m in range(config['monte_carlo_runs']):
            print(f"  Monte Carlo run {m + 1}/{config['monte_carlo_runs']}...")
            
            all_train_losses_per_fold, all_val_losses_per_fold, global_test_loss, jam_loc_error = main(
                data_args, model_args, alg_args, config['theta_init']
            )

            # Accumulate results for this run
            r_mc_test_loss += global_test_loss
            r_mc_jam_loc_error += jam_loc_error

        # Compute average results for this run
        r_mc_test_loss /= config['monte_carlo_runs']
        r_mc_jam_loc_error /= config['monte_carlo_runs']

        # Print averaged results for this run
        print(f"Run {r + 1} results:")
        print(f"  Average global test loss: {r_mc_test_loss:.4f}")
        print(f"  Average jammer localization error: {r_mc_jam_loc_error:.4f}\n")

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

    # Optional: save results as a CSV or other format
    # df_loss = pd.DataFrame(loss_temp1)
    # df_loss.to_csv(f'{output_path}{data_name}_results_loss.csv', index=False)
    # df_loc = pd.DataFrame(loc_temp2)
    # df_loc.to_csv(f'{output_path}{data_name}_loc_nodes{num_nodes}_theta({theta_init})_weight({weight_method})_{date.datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)