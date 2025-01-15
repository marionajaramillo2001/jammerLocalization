# main

import torch
import numpy as np
import pandas as pd
import os
# from fedbase.fedavg import fedavg
from fedbase_v2.data_loader import data_process
from fedbase_v2.model import Net_augmented
import torch.optim as optim
from functools import partial
from fedbase_v2.fedavg import FedAvg
from fedbase_v2.fedavg_online import FedAvg as FedAvg_oneline

os.chdir(os.path.dirname(os.path.abspath(__file__))) # set the current path as the working directory


# parameters control the data
## data itself: path, noise, point_section

## data split: test_ratio, client_num, division_method(iid, non-iid)

# parameters control the model
## model itself: model, activation, loss function

def main(seeds, num_nodes, data_args, model_args, alg_name, alg_args):
    np.random.seed(seeds)
    # data
    d_p = data_process(**data_args)
    dataset_splited = d_p.split_dataset(num_nodes)

    model = Net_augmented(**model_args, theta0=torch.tensor([50.0,50.0]))
    # algorithm
    if alg_name == "FedAvg" or alg_name=='Fedprox':
        fedavg = FedAvg(model, **alg_args)
        df = fedavg.train(dataset_splited, d_p.trueJloc)

    # elif alg_name == 'FedAvg_online':
    #     fedavg = FedAvg_oneline(model, **alg_args)
    #     df = fedavg.train(dataset_splited, d_p.trueJloc)
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
    
    optimizer_nn = partial(optim.Adam,lr=0.1,weight_decay=1e-5)
    optimizer_theta = partial(optim.Adam,lr=0.1, weight_decay=1e-5)
    
    num_nodes = 10 #[2,5] #10 #[2,4] #[1,1] #1
    split_method= 'random' #'random' #'area_grid' # divie the data method
    weight_ratio = 'normal' # mean_max, mean, max, power_max, normal
    layer_wid = [500,1] #[200,100,1] #[500,1]
    betas = True # control the regularization term
    algorithm = 'FedAvg' # 'FedAvg_oneline' #'FedAvg' #'Fedprox','Fedprox-PL'
    model_args = {
        'input_dim': 2,
        'layer_wid': layer_wid,
        'nonlinearity': 'relu',
        'gamma': 2,
        'model_mode': 'both'
    }
    alg_args = {
        'num_rounds': 10,
        'local_epochs': 100,
        'optimizer_nn': optimizer_nn,
        'optimizer_theta': optimizer_theta,
        'algorithm': algorithm,
        'weight_ratio': weight_ratio,
        'mu': 0.1,
        'betas': betas
    }
    alg_name = 'FedAvg'
    local_batch = 16
    # online = False
    # if online:
    #     alg_name = 'FedAvg_online' #'Fedprox' #'FedAvg'
    #     alg_args = {'num_rounds': 5, 'optimizer_nn': optimizer_nn, 'optimizer_theta': optimizer_theta, 'local_epochs': 100, 'algorithm':algorithm, 'weight_ratio': weight_ratio, 'mu': 0.1, 'online': True}
    #     local_batch=1

    seeds = 2024
    data_args = {
        'path': path,
        'time_t': 1,
        'test_ratio': 0.2,
        'data_preprocessing': 1,
        'noise': False,
        'noise_std': 3,
        'split_method': split_method,
        'local_bs': local_batch
    }
    df = main(seeds, num_nodes, data_args, model_args, alg_name,alg_args)
            
    