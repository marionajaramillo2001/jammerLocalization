import os
from contextlib import redirect_stdout
import time
import torch
import numpy as np
import random
import torch.optim as optim
from functools import partial
from apbm_v2_fed.data_loader_APBM import data_process
from apbm_v2_fed.fedavg import FedAvg
from apbm_v2_fed.model import Net, Polynomial3, Net_augmented
from apbm_v2_fed.plots import plot_train_test_loss, visualize_3d_model_output, plot_ECDF, plot_boxplot

# Set the current path as the working directory
os.chdir(os.getcwd())


def create_directories(execution_folder, scenario, model_type, experiment_type, value, mc_run):
    """
    Create the necessary directory structure for the experiment.
    """    
    scenario_folder = os.path.join(execution_folder, scenario)
    model_folder = os.path.join(scenario_folder, model_type)
    experiment_folder = os.path.join(model_folder, experiment_type)
    experiment_value_folder = os.path.join(experiment_folder, str(value))

    # Create directories
    os.makedirs(experiment_value_folder, exist_ok=True)

    # Subdirectories for results for each Monte Carlo run
    mc_run_folder = os.path.join(experiment_value_folder, f"mc_run_{mc_run}")
    os.makedirs(mc_run_folder, exist_ok=True)

    return experiment_folder, experiment_value_folder, mc_run_folder


def prepare_data(config):
    data_args = {
        'path': config['path'],
        'time_t': config['time_t'],
        'test_ratio': config['test_ratio'],
        'data_preprocessing': config['data_preprocessing'],
        'noise': config['noise'],  # Add noise here, if data comes from MATLAB w/out noise
        'meas_noise_var': config['meas_noise_var'],
        'batch_size': config['batch_size'],
        'num_obs': config['num_obs'],
    }
    
    alg_args = {
        'batch_size': config['batch_size'],
        'local_epochs_nn': config['local_epochs_nn'],
        'local_epochs_pl': config['local_epochs_pl'],
        'local_epochs_apbm': config['local_epochs_apbm'],
        'num_rounds_nn': config['num_rounds_nn'],
        'num_rounds_pl': config['num_rounds_pl'],
        'num_rounds_apbm': config['num_rounds_apbm'],
    }
    
    model_init_args = {
        'input_dim': config['input_dim'],
        'layer_wid': config['layer_wid'],
        'nonlinearity': config['nonlinearity'],
    }
    
    model_pl_args = {
        'gamma': config['gamma'],
    }
    
    model_apbm_args = {
        'input_dim': config['input_dim'],
        'layer_wid': config['layer_wid'],
        'nonlinearity': config['nonlinearity'],
        'gamma': config['gamma']
    }
    
    model_type = config['model_type']
    
    # Data processing step to load and split the dataset
    d_p = data_process(**data_args)
    train_loader_splited, test_loader, max_values_each_partition, train_y_mean_splited, train_x_max, train_y_max = d_p.split_dataset(config['num_nodes'])

    # Define partial optimizers for NN and theta
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=0.0)
    optimizer_P0 = partial(optim.Adam, lr=config['lr_optimizer_P0'], weight_decay=0.0)
    optimizer_gamma = partial(optim.Adam, lr=config['lr_optimizer_gamma'], weight_decay=0.0)

    model_init = Net(**model_init_args)
    if model_type == 'PL':
        model = Polynomial3(**model_pl_args)
    elif model_type == 'APBM':
        model = Net_augmented(**model_apbm_args)
    
    return model_init, model, model_type, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, d_p.trueJloc, train_loader_splited, test_loader, train_y_mean_splited, alg_args


def train_test(config, seed, output_dir, show_figures, mc_run=None):
    # Set the seed for this Monte Carlo run
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_init, model, model_type, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, true_jam_loc, train_loader_splited, test_loader, train_y_mean_splited, alg_args = prepare_data(config)
    
    # Create FedAvg instance and train the model
    train = FedAvg(model_init, model, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, **alg_args)
    train_losses_init_per_round, train_losses_model_per_round, test_losses_init_per_round, test_losses_model_per_round, jam_init_loc_error, theta_init, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, _, trained_model = train.train_test_pipeline(train_loader_splited, test_loader, true_jam_loc, model_type, train_y_mean_splited, show_figures, output_dir, mc_run)
    
    if show_figures:
        plot_train_test_loss(train_losses_init_per_round, test_losses_init_per_round, pl_or_apbm_or_nn='NN', folder=output_dir, mc_run=mc_run)
        plot_train_test_loss(train_losses_model_per_round, test_losses_model_per_round, pl_or_apbm_or_nn=model_type, folder=output_dir, mc_run=mc_run)
        visualize_3d_model_output(trained_model, train_loader_splited, test_loader, theta_init, true_jam_loc, predicted_jam_loc, None, train_or_test='train', pl_or_apbm_or_nn=model_type, folder=output_dir, mc_run=mc_run)
        visualize_3d_model_output(trained_model, train_loader_splited, test_loader, theta_init, true_jam_loc, predicted_jam_loc, None, train_or_test='test', pl_or_apbm_or_nn=model_type, folder=output_dir, mc_run=mc_run)

    # Redirect print statements to a file
    log_file = os.path.join(output_dir, "results_log.txt")
    with open(log_file, "w") as f:
        print(f"Seed: {seed}", file=f)
        print(f"Final Test Loss (Model): {test_losses_model_per_round[-1]}", file=f)
        print(f"Jammer Localization Error: {jam_loc_error}", file=f)

    return test_losses_model_per_round[-1], jam_init_loc_error, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma


if __name__ == '__main__':
    # Configuration dictionary
    config = {
        'path': '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT18/obs_time_1/',
        'time_t': 0,
        'test_ratio': 0.2,
        'data_preprocessing': 2,
        'noise': 1,
        'meas_noise_var': 1,
        'betas': True,
        'input_dim': 2,
        # 'layer_wid': [256, 128, 64, 1],
        'layer_wid': [500, 256, 128, 1],
        'nonlinearity': 'leaky_relu',
        'gamma': 2,
        'num_nodes': 10,
        'local_epochs_nn': 20,
        'local_epochs_pl': 20,
        'local_epochs_apbm': 20,
        'num_rounds_nn': 40,
        'num_rounds_pl': 40,
        'num_rounds_apbm': 40,
        'batch_size': 8,
        'lr_optimizer_nn': 0.001,
        'lr_optimizer_theta': 0.5,
        'lr_optimizer_P0': 0.01,
        'lr_optimizer_gamma': 0.01,
        'weight_decay_optimizer_nn': 0,
        'model_type': 'PL', # 'PL' or 'APBM'
        'num_obs': 1000,
    }

    # Number of Monte Carlo runs
    N_mc = 1
    
    # Execution type
    execution_type = 'one_experiment' # 'one_experiment' or 'all_experiments'

    base_seed = 42  # Base seed for reproducibility
    current_path = os.getcwd()
    base_path = os.path.join(current_path, "results")
    
    existing_folders = [folder for folder in os.listdir(base_path) if folder.startswith("Execution_")]
    existing_ids = [int(folder.split("_")[1]) for folder in existing_folders if folder.split("_")[1].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    execution_folder = os.path.join(base_path, f"Execution_{next_id}")
    os.makedirs(execution_folder, exist_ok=True)

    
    if execution_type == 'one_experiment':
        # scenarios = ['suburban_raytrace']
        scenarios = ['urban_raytrace']
        model_folder = ['PL']
        experiments = ['show_figures']
        numNodes = 15
        posEstVar = 0
        num_obs = 1000
        meas_noise_var = 1
        show_figures = True
    elif execution_type == 'all_experiments':
        # scenarios = ['suburban_raytrace', 'urban_raytrace', 'pathloss']
        scenarios = ['urban_raytrace']
        # model_folder = ['PL', 'APBM', 'PL_APBM']
        model_folder = ['PL']
        experiments = ['numNodes', 'posEstVar', 'num_obs', 'meas_noise_var']
        numNodes = np.array([1, 5, 10, 25, 50])
        posEstVar = np.array([0,36])
        num_obs = np.array([250, 500, 750, 1000])
        meas_noise_var = np.array([10, 10/np.sqrt(10),  1, 0.1])
        show_figures = False
    
    
    
    # Constants for the datasets used
    area = 1e6
    Ptx = 10
    
    for scenario in scenarios:
        if scenario == 'suburban_raytrace':
            path_without_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT33/obs_time_1/'
            path_with_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT34/obs_time_1/'
        elif scenario == 'urban_raytrace':
            path_without_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT35/obs_time_1/'
            path_with_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSSjamLoc/RT36/obs_time_1/'
        elif scenario == 'pathloss':
            path_without_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSS-FL/datasets/dataPLANS/4.definitive/PL2/'
            path_with_posEstVar = '/Users/marionajaramillocivill/Documents/GitHub/GNSS-FL/datasets/dataPLANS/4.definitive/PL10/'

        for model_type in model_folder:
            if model_type == 'PL':
                config['model_type'] = 'PL'
            elif model_type == 'APBM':
                config['model_type'] = 'APBM'
                
            for experiment in experiments:
                if experiment == 'numNodes':
                    values_to_iterate = numNodes
                    x_axis_values_boxplot = numNodes
                    x_label_boxplot = "Number of Nodes"
                elif experiment == 'posEstVar':
                    values_to_iterate = posEstVar
                    x_axis_values_boxplot = posEstVar
                    x_label_boxplot = r"Position Estimation Variance (m$^2$)"
                elif experiment == 'num_obs':
                    values_to_iterate = num_obs
                    x_axis_values_boxplot = num_obs/area
                    x_label_boxplot = r"obs. density (obs/(m$^2$)"
                elif experiment == 'meas_noise_var':
                    values_to_iterate = meas_noise_var
                    x_axis_values_boxplot = 10*np.log10(Ptx/meas_noise_var)
                    x_label_boxplot = "INR (dB)"
                elif experiment == 'show_figures':
                    values_to_iterate = ['show_figures']
                    
                
                experiment_value_folder = []
                results_mc_runs_all_values = [] # To store results for each value
                for value in values_to_iterate:
                    results_mc_run = []
                    for mc_run in range(N_mc):
                        # Generate a unique seed for this run
                        experiment_folder, experiment_value_folder, mc_run_folder = create_directories(execution_folder, scenario, model_type, experiment, value, mc_run)
                    
                        run_seed = base_seed + mc_run
                        
                        if experiment == 'numNodes':
                            config['num_nodes'] = value
                            config['path'] = path_without_posEstVar
                            config['num_obs'] = 1000
                            config['meas_noise_var'] = 1
                        elif experiment == 'posEstVar':
                            config['num_nodes'] = 10
                            if value == 0:
                                config['path'] = path_without_posEstVar
                            elif value == 36:
                                config['path'] = path_with_posEstVar
                            config['num_obs'] = 1000
                            config['meas_noise_var'] = 1
                        elif experiment == 'num_obs':
                            config['num_nodes'] = 10
                            config['path'] = path_without_posEstVar
                            config['num_obs'] = value
                            config['meas_noise_var'] = 1
                        elif experiment == 'meas_noise_var':
                            config['num_nodes'] = 10
                            config['path'] = path_without_posEstVar
                            config['num_obs'] = 1000
                            config['meas_noise_var'] = value
                        elif experiment == 'show_figures':
                            config['num_nodes'] = numNodes
                            if posEstVar == 0:
                                config['path'] = path_without_posEstVar
                            elif posEstVar == 36:
                                config['path'] = path_with_posEstVar
                            config['num_obs'] = num_obs
                            config['meas_noise_var'] = meas_noise_var
                        
                        # Redirect print statements to a file
                        log_file = os.path.join(mc_run_folder, "log.txt")
                        with open(log_file, "w") as f:
                            with redirect_stdout(f):
                                print(f"Monte Carlo Run {mc_run + 1}/{N_mc} with Seed: {run_seed}")
                                
                                print("Configuration:")
                                for key, val in config.items():
                                    print(f"{key}: {val}")
                                    
                                # Run train-test cycle
                                global_test_loss, jam_init_loc_error, jam_loc_error, true_jam_loc, predicted_jam_loc, learnt_P0, learnt_gamma = train_test(config, run_seed, mc_run_folder, show_figures, mc_run)

                                # Store results
                                results_mc_run.append({
                                    'run': mc_run + 1,
                                    'seed': run_seed,
                                    'global_test_loss': global_test_loss,
                                    'jam_init_loc_error': jam_init_loc_error,
                                    'jam_loc_error': jam_loc_error,
                                    'true_jam_loc': true_jam_loc,
                                    'predicted_jam_loc': predicted_jam_loc,
                                    'learnt_P0': learnt_P0,
                                    'learnt_gamma': learnt_gamma,
                                })

                                print(f"  Global Test Loss: {global_test_loss:.4f}")
                                print(f"  Jammer Initial Localization Error: {jam_init_loc_error:.4f}")
                                print(f"  Jammer Localization Error: {jam_loc_error:.4f}")

                    plot_ECDF(results_mc_run, experiment_value_folder)  
                    results_mc_runs_all_values.append(results_mc_run)
                                        
                    # Save aggregate results to a text file
                    results_file = os.path.join(experiment_value_folder, "results.txt")
                    with open(results_file, "w") as f:
                        f.write(f"Scenario: {scenario}\n")
                        f.write(f"Model Type: {model_type}\n")
                        f.write(f"Experiment: {experiment}\n")
                        f.write(f"Value: {value}\n")
                        f.write("Configuration:\n")
                        for key, val in config.items():
                            f.write(f"  {key}: {val}\n")
                        test_losses = [res['global_test_loss'] for res in results_mc_run]
                        jam_init_loc_errors = [res['jam_init_loc_error'] for res in results_mc_run]
                        jam_loc_errors = [res['jam_loc_error'] for res in results_mc_run]

                        print("\nMonte Carlo Results Summary:")
                        print(f"Average Global Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
                        print(f"Average Jammer Inital Localization Error: {np.mean(jam_init_loc_errors):.4f} ± {np.std(jam_init_loc_errors):.4f}")
                        print(f"Average Jammer Localization Error: {np.mean(jam_loc_errors):.4f} ± {np.std(jam_loc_errors):.4f}")
                        
                        f.write(f"Average Global Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}\n")
                        
                        f.write(f"Average Jammer Localization Error: {np.mean(jam_loc_errors):.4f} ± {np.std(jam_loc_errors):.4f}\n")
                        f.write(f"Median Jammer Localization Error: {np.median(jam_loc_errors):.4f}\n")
                        f.write(f"Interquartile Range of Jammer Localization Error: {np.percentile(jam_loc_errors, 75) - np.percentile(jam_loc_errors, 25):.4f}\n")
                        f.write(f"Minimum Jammer Localization Error: {np.min(jam_loc_errors):.4f}\n")
                        f.write(f"Maximum Jammer Localization Error: {np.max(jam_loc_errors):.4f}\n")
                        
                        f.write(f"Average Jammer Initial Localization Error: {np.mean(jam_init_loc_errors):.4f} ± {np.std(jam_init_loc_errors):.4f}\n")
                        f.write(f"Median Jammer Initial Localization Error: {np.median(jam_init_loc_errors):.4f}\n")
                        f.write(f"Interquartile Range of Jammer Initial Localization Error: {np.percentile(jam_init_loc_errors, 75) - np.percentile(jam_init_loc_errors, 25):.4f}\n")
                        f.write(f"Minimum Jammer Initial Localization Error: {np.min(jam_init_loc_errors):.4f}\n")
                        f.write(f"Maximum Jammer Initial Localization Error: {np.max(jam_init_loc_errors):.4f}\n")  
                
                if experiment != 'show_figures':        
                    plot_boxplot(x_axis_values_boxplot, results_mc_runs_all_values, experiment_folder, x_label_boxplot)

    