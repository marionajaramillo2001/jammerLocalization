import torch
import numpy as np
import random
import os
import time
import torch.optim as optim
from functools import partial
from apbm_v2_fed.data_loader_APBM import data_process
from apbm_v2_fed.fedavg import FedAvg
from apbm_v2_fed.model import Net, Polynomial3
from apbm_v2_fed.plots import plot_train_test_loss, visualize_3d_model_output

# Set the current path as the working directory
os.chdir(os.getcwd())

# Function to prepare data and models
def prepare_data(config):
    data_args = {
        'path': config['path'],
        'time_t': config['time_t'],
        'test_ratio': config['test_ratio'],
        'data_preprocessing': config['data_preprocessing'],
        'noise': config['noise'],  # Add noise here, if data comes from MATLAB w/out noise
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
    
    return model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p.trueJloc, train_loader_splited, test_loader, train_y_mean_splited, alg_args


# Function to perform a single train-test cycle
def train_test(config, seed, show_losses_plot, show_fields_plot):
    # Set the seed for this Monte Carlo run
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, true_jam_loc, train_loader_splited, test_loader, train_y_mean_splited, alg_args = prepare_data(config)
    
    # Create FedAvg instance and train the model
    train = FedAvg(model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    train_losses_nn_per_round, train_losses_pl_per_round, test_losses_nn_per_round, test_losses_pl_per_round, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, _, trained_model = train.train_test_pipeline(train_loader_splited, test_loader, true_jam_loc, train_y_mean_splited)
    
    # Plot and visualize
    plot_train_test_loss(train_losses_nn_per_round, test_losses_nn_per_round, show_losses_plot, pl_or_apbm_or_nn='nn')
    plot_train_test_loss(train_losses_pl_per_round, test_losses_pl_per_round, show_losses_plot, pl_or_apbm_or_nn='pl')
    visualize_3d_model_output(trained_model, train_loader_splited, test_loader, true_jam_loc, predicted_jam_loc, None, show_fields_plot, train_or_test='train', pl_or_apbm_or_nn='pl')
    visualize_3d_model_output(trained_model, train_loader_splited, test_loader, true_jam_loc, predicted_jam_loc, None, show_fields_plot, train_or_test='test', pl_or_apbm_or_nn='pl')

    return test_losses_pl_per_round[-1], jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma


# Main function
if __name__ == '__main__':
    # Configuration dictionary
    config_id_0 = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT18/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': 1,
        'noise_std': 3,
        'betas': True,
        'input_dim': 2,
        'layer_wid': [256, 128, 64, 1],
        'nonlinearity': 'leaky_relu',
        'gamma': 2,
        'num_nodes': 10,
        'local_epochs_nn': 20,
        'local_epochs_pl': 20,
        'num_rounds_nn': 30,
        'num_rounds_pl': 30,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.1,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0,
    }

    # Number of Monte Carlo runs
    N_mc = 20
    mc_results = []

    base_seed = 42  # Base seed for reproducibility
    start_time = time.time()

    for mc_run in range(N_mc):
        # Generate a unique seed for this run
        run_seed = base_seed + mc_run**2

        print(f"Monte Carlo Run {mc_run + 1}/{N_mc} with Seed: {run_seed}")

        # Run train-test cycle
        show_losses_plot = False
        show_fields_plot = False
        global_test_loss, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma = train_test(config_id_0, run_seed, show_losses_plot, show_fields_plot)

        # Store results
        mc_results.append({
            'run': mc_run + 1,
            'seed': run_seed,
            'global_test_loss': global_test_loss,
            'jam_loc_error': jam_loc_error,
            'true_jam_loc': true_jam_loc,
            'predicted_jam_loc': predicted_jam_loc,
            'learnt_P0': learnt_P0,
            'learnt_gamma': learnt_gamma,
        })

        print(f"  Global Test Loss: {global_test_loss:.4f}")
        print(f"  Jammer Localization Error: {jam_loc_error:.4f}")
        print(f"  Predicted Jammer Location: {predicted_jam_loc}")
        print(f"  Real Jammer Location: {true_jam_loc}\n")

    # Aggregate and summarize results
    test_losses = [res['global_test_loss'] for res in mc_results]
    jam_loc_errors = [res['jam_loc_error'] for res in mc_results]

    print("\nMonte Carlo Results Summary:")
    print(f"Average Global Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
    print(f"Average Jammer Localization Error: {np.mean(jam_loc_errors):.4f} ± {np.std(jam_loc_errors):.4f}")

    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")