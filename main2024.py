# main

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from fedbase.fedavg import fedavg
from fedbase.data_loader import data_process
from fedbase.model import Net_augmented, Net_augmented_gate
import torch.optim as optim
from functools import partial
from fedbase.fedavg import FedAvg
from fedbase.fedavg_online import FedAvg as FedAvg_oneline
import datetime as date

os.chdir(os.path.dirname(os.path.abspath(__file__))) # set the current path as the working directory


# parameters control the data
## data itself: path, noise, point_section

## data split: test_ratio, client_num, division_method(iid, non-iid)

# parameters control the model
## model itself: model, activation, loss function

# def main function run 1 time
def main(seeds, num_nodes, data_args, model_args, alg_name, alg_args, theta_init='max_loc'):
    np.random.seed(seeds)
    # data
    d_p = data_process(**data_args)
    dataset_splited = d_p.split_dataset(num_nodes)
    # model
    if theta_init == 'None':
        model = Net_augmented(**model_args)
    elif theta_init == 'max_loc':
        model = Net_augmented(**model_args, theta0=dataset_splited[-1]+torch.randn(2))
    elif theta_init == 'random':
        model = Net_augmented(**model_args, theta0=99*torch.rand(2))
    elif theta_init == 'fix':
        model = Net_augmented(**model_args, theta0=torch.tensor([50.0,50.0]))
    # algorithm
    if alg_name == "FedAvg" or alg_name=='Fedprox':
        fedavg = FedAvg(model, **alg_args)
        df = fedavg.train(dataset_splited, d_p.trueJloc)
        # test_metrics = fedavg(dataset_splited, batch_size, num_nodes, model, alg_args)
        # return global_losses, global_loc
        # return df['loss'], df['loc_error']
    elif alg_name == 'FedAvg_online':
        fedavg = FedAvg_oneline(model, **alg_args)
        df = fedavg.train(dataset_splited, d_p.trueJloc)
        # return df['loss'], df['loc_error']
    return df

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

    # model
    
    # optimizer = partial(optim.Adam,lr=0.1,weight_decay=1e-5)    # monte carlo runs
    optimizer_nn = partial(optim.Adam,lr=0.1,weight_decay=1e-5)
    # optimizer_nn = partial(optim.Adam,lr=0.1)
    optimizer_theta = partial(optim.Adam,lr=0.1, weight_decay=1e-5)
    # optimizer_theta = partial(optim.Adam,lr=4)
    # optimizer_theta = partial(optim.SGD,lr=1)
    # torch.optim.SGD(model.parameters(), lr=0.5)
    
    num_nodes = 10 #[2,5] #10 #[2,4] #[1,1] #10 #[1,1] # #1 [2,4]
    split_method= 'random' #'random' #'area_grid' # divie the data method
    theta_init = 'fix' #'max_loc' #'random', 'fix', 'None'
    weight_method = 'global_power_mean' #'normal' #'global_power_mean' #'local_power_mean' #'bins' # 'batch_power_mean','batch_power_mean2', 'local_power_mean'
    weight_ratio = 'normal' # mean_max, mean, max, power_max, normal
    layer_wid = [500,1] #[200,100,1] #[500,1]
    runs = 1 # max 50, min 1
    monte_carlo_runs = 5
    loss_temp1 = 0
    loc_temp2 = 0
    betas = True # control the regularization term
    algorithm = 'FedAvg' # 'FedAvg_oneline' #'FedAvg' #'Fedprox','Fedprox-PL'
    model_args = {'input_dim': 2, 'layer_wid': layer_wid, 'nonlinearity': 'relu', 'gamma': 2, 'data_max': None, 'data_min': None, 'model_mode': 'both'}
    alg_args = {'num_rounds': 10, 'local_epochs': 100, 'optimizer_nn': optimizer_nn, 'optimizer_theta': optimizer_theta, 'algorithm':algorithm, 'weight_ratio': weight_ratio, 'mu': 0.1, 'weight_method':weight_method, 'betas':betas} 
    alg_name = 'FedAvg'
    local_batch = 16
    online = False
    if online:
        alg_name = 'FedAvg_online' #'Fedprox' #'FedAvg'
        alg_args = {'num_rounds': 5, 'optimizer_nn': optimizer_nn, 'optimizer_theta': optimizer_theta, 'local_epochs': 100, 'algorithm':algorithm, 'weight_ratio': weight_ratio, 'mu': 0.1, 'weight_method':weight_method, 'online': True}
        local_batch=1

    for r in range(runs):
        seeds = 2024 #np.random.choice(10**3)
        data_args = {'path': path, 'time_t': r, 'test_ratio': 0.2, 'data_preprocesing': 1, 'noise': False, 'noise_std': 3, 'data_selection': False, 'point_section': 100, 'split_method': split_method, 'local_bs': local_batch, 'bins_num': 10}
        for m in range(monte_carlo_runs):
            df = main(seeds, num_nodes, data_args, model_args, alg_name,alg_args,theta_init)
            # take the average of the global_losses and global_loc
            loss_temp1+=df['loss']
            loc_temp2+=df['loc_error']
    loss_temp1 = loss_temp1/(runs*monte_carlo_runs)
    loc_temp2 = loc_temp2/(runs*monte_carlo_runs)
    # save the results
    path = 'results7/'
    if os.path.exists(path) == False:
        os.mkdir(path)
    # save the results separately
    df_loss = pd.DataFrame(loss_temp1)
    df_loss.to_csv(f'{path}{data_name}_results_loss.csv', index=False)
    df_loc = pd.DataFrame(loc_temp2)
    df_loc.to_csv(f'{path}{data_name}_loc_nodes{num_nodes}_theta({theta_init})_weight({weight_method})_{date.datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)
    
    


        
    