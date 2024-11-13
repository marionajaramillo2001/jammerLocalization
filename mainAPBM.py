import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import datetime as date
from apbm.data_loader_APBM import CustomDataset, data_process
from apbm.crossval import CrossVal
from apbm.model import Net_augmented
from functools import partial
import random

# random seed configurations
seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

os.chdir(os.getcwd()) # set the current path as the working directory

def main(data_args, model_args, alg_args, theta_init):
    # data
    d_p = data_process(**data_args)
    indices_folds, crossval_dataset, test_loader = d_p.split_dataset()
    
    ## Define the model
    if theta_init == 'fix':
        model = Net_augmented(**model_args, theta0=torch.tensor([50.0,50.0]))
    crossval = CrossVal(model, **alg_args)
    train_losses, val_losses, global_test_loss = crossval.train(indices_folds, crossval_dataset,test_loader, d_p.trueJloc)
    
    return train_losses, val_losses, global_test_loss


if __name__ == '__main__':
    
    data_name = 'RT2'
    if data_name == 'RT2': # 100
        path = 'datasets/dataPLANS/4.definitive/RT2/'
    elif data_name == 'PL2': # 10000
        path = 'datasets/dataPLANS/4.definitive/PL2/'
    elif data_name == 'R11': # 100
        path = 'datasets/dataPLANS/4.definitive/RT11/'
    elif data_name == 'PL11': # 10000
        path = 'datasets/dataPLANS/4.definitive/PL11/'
    elif data_name == 'R12': # 10000
        path = 'datasets/dataPLANS/4.definitive/RT12/'
    elif data_name == 'R13': # 10000
        path = 'datasets/dataPLANS/4.definitive/RT13/'

    
    config = {
        "theta_init": 'fix', # 'fix' or 'random' or 'max_loc' or 'None'
        "runs": 1,
        "monte_carlo_runs": 5,
        "betas": True,
        "model_args": {
            'input_dim': 2, 
            'layer_wid': [500,1],
            'nonlinearity': 'relu', 
            'gamma': 2, 
            'model_mode': 'both'
            },
        "max_epochs": 200,
        "batch_size": 16,
        "lr_optimizer_nn": 0.1,
        "lr_optimizer_theta": 0.1,
        "weight_decay_optimizer_nn": 1e-5,
        "weight_decay_optimizer_theta": 1e-5,
        "mu": 0.1,
    }
    
    # model
    
    # optimizer = partial(optim.Adam,lr=0.1,weight_decay=1e-5)    # monte carlo runs
    
    optimizer_nn = partial(optim.Adam, lr=config['lr_optimizer_nn'], weight_decay=config['weight_decay_optimizer_nn'])
    # optimizer_nn = partial(optim.Adam,lr=0.1)
    optimizer_theta = partial(optim.Adam, lr=config['lr_optimizer_theta'], weight_decay=config['weight_decay_optimizer_theta'])
    # optimizer_theta = partial(optim.Adam,lr=4)
    # optimizer_theta = partial(optim.SGD,lr=1)
    # torch.optim.SGD(model.parameters(), lr=0.5)
    alg_args = {'max_epochs': config['max_epochs'], 'batch_size': config['batch_size'], 'optimizer_nn': optimizer_nn, 'optimizer_theta': optimizer_theta, 'mu': config['mu'], 'betas': config['betas']}
    alg_args = {'max_epochs': config['max_epochs'], 'batch_size': config['batch_size'], 'optimizer_nn': optimizer_nn, 'optimizer_theta': optimizer_theta, 'mu': config['mu'], 'betas': config['betas']}
    model_args = config['model_args']
                
    r_mc_test_loss = 0
    

    for r in range(config['runs']):
        data_args = {'path': path, 'time_t': r, 'test_ratio': 0.2, 'data_preprocesing': 1, 'batch_size': config['batch_size'], 'bins_num': 10}
        for m in range(config['monte_carlo_runs']):
            train_losses, val_losses, global_test_loss = main(data_args, model_args, alg_args, config['theta_init'])
            # take the average of the global_losses and global_loc
            r_mc_test_loss += global_test_loss
        r_mc_test_loss = r_mc_test_loss/(config['runs']*config['monte_carlo_runs'])
        print(r_mc_test_loss)
        # save the results
        path = 'results7/'
        if os.path.exists(path) == False:
            os.mkdir(path)
        # save the results separately
    # df_loss = pd.DataFrame(loss_temp1)
    # df_loss.to_csv(f'{path}{data_name}_results_loss.csv', index=False)
    # df_loc = pd.DataFrame(loc_temp2)
    # df_loc.to_csv(f'{path}{data_name}_loc_nodes{num_nodes}_theta({theta_init})_weight({weight_method})_{date.datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)
