#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
library of functions
"""
import numpy as np
import torch
import sys
sys.path.insert(1, 'BFNN')
from kalman import KalmanFilter, StateModel, ObservationModel, NNEKF
import plotly.io as pio
pio.renderers.default='browser'
import copy
sys.path.insert(1, 'pykalman')
from pykalman import KalmanFilter
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def normalize(tensor_list):
    
  # manage inf values
  # -inf means no sig. Should they be removed?
  
  # set them lower than the non-inf minimum, but not too much to avoid disrupting the value range
  coef = 1.5
  tensor_list[tensor_list.isneginf()] = np.nan # removing the -inf
  tmp = tensor_list.topk(1,dim=0,largest=False) # finding the min omitting the NaN
  tmpmin = tmp.values.min()
  tensor_list[tensor_list.isnan()] = coef*tmpmin # set the NaN to a new minimum
  
  tensor_list[tensor_list.isinf()] = np.nan # removing the inf
  tmp = tensor_list.topk(1,dim=0,largest=True) # finding the max omitting the NaN
  tmpmax = tmp.values.max()
  tensor_list[tensor_list.isnan()] = coef*tmpmax # set the NaN to a new minimum

  mins = tensor_list.min(dim=0, keepdim=True)[0]
  maxs = tensor_list.max(dim=0, keepdim=True)[0]
  
  tensor_list = (tensor_list - mins) / (maxs - mins)
  # tensor_list = tensor_list.float()
  return tensor_list,maxs,mins

def normalize_maxmin(tensor_list,maxs,mins):
    """
    normalize wrt to external max and min
    """
    tensor_list = (tensor_list - mins) / (maxs - mins)
    # tensor_list = tensor_list.float()
    return tensor_list

def initEKF(layers, noise_pw, nonlinearity, init_weights, bias):
    """
    Init EKF (with state and meas dimensions, initial state, and cov matrices)
    """
    
    # x_dim = layers[0]
    # s_dim = layers[1:-1]
    y_dim = layers[-1]
    
   
    ## state_model = StateModel(x_dim, s_dim, nonlinearity)
    ## observation_model = ObservationModel(s_dim[-1], y_dim)
    
    observation_model = ObservationModel(layers,nonlinearity,init_weights, bias)
    
    # init initial internal state (instant k-1) (all hidden layers)
    ##s0 = torch.zeros((sum(s_dim), 1))

    # total no of param (state+obs)
    ## w_dim = state_model.get_param().data.shape[0] + observation_model.get_param().data.shape[0]
    w_dim = observation_model.get_param().data.shape[0]



    ##P = np.sqrt(noise_pw) * torch.eye(w_dim + s_dim)       
    P = np.sqrt(noise_pw) * torch.eye(w_dim)

    
    
    Q = np.sqrt(noise_pw) * torch.eye(y_dim) # measurement cov
    R = 0.1 * np.sqrt(noise_pw) * torch.eye(len(P)) # process cov

    nnekf = NNEKF(observation_model, P, R, Q)
    return nnekf, w_dim

def getSampleGrid(test_y_predict,test_x, radius = 20, Npoints = 80):
    #--- based on test data
    # sample the space around test_Jloc
    # radius: sample space area (m)
    # Npoints: sample points within the radius
    MM = test_y_predict.max(dim=0,keepdim=True)[0]
    Idx = test_y_predict.argmax(dim=0,keepdim=True)[0]
    test_Jloc = test_x.numpy()[Idx,:]
    
    x = np.linspace(test_Jloc[0]-radius,test_Jloc[0]+radius,Npoints*2)
    y = np.linspace(test_Jloc[1]-radius,test_Jloc[1]+radius,Npoints*2)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    sampGrid_mini = np.array([xv,yv]).T
    sampGrid_x = torch.from_numpy(np.array(sampGrid_mini)).float()
    return sampGrid_x, sampGrid_mini, MM

def grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc):
    MM = sampGrid_y.max(dim=0,keepdim=True)[0]
    Idx = sampGrid_y.argmax(dim=0,keepdim=True)[0]
    Jloc = sampGrid_mini[Idx,:]
    err = Jloc - trueJloc
    return Jloc, err, MM

def jam_pos_estimation_grid(test_y_predict,test_x,model_NN,trueJloc):
    sampGrid_x, sampGrid_mini, _ = getSampleGrid(test_y_predict,test_x)
    sampGrid_y = model_NN(sampGrid_x).detach()
    Jloc, err, _ = grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc)  
    return Jloc, err

def jam_pos_estimation_grad(test_y_predict,test_x,model_NN,trueJloc):
    model = copy.deepcopy(model_NN)
    # model = type(model_NN)() # get a new instance
    # model.load_state_dict(model_NN.state_dict()) # copy weights and stuff
    Idx = test_y_predict.argmax(dim=0,keepdim=True)[0]
    x0 = test_x[Idx,:]
    x0.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False
    
    # gradient ascent loop
    lr = 1e-2
    
    # optim = torch.optim.SGD(x0, lr)
    
    #y0 = -y0 # ascent
    for i in range(100):
        # optim.zero_grad()
        y0 = model(x0)
        #compute gradient
        y0.backward()
        with torch.no_grad():
            x0 = x0 + lr * x0.grad  # + for the ascent
        # optim.step()
        x0.requires_grad = True
    
    Jloc = x0.detach().numpy().reshape(-1)
    err = Jloc-trueJloc
    return Jloc, err

def dist(X,xj):
    d = np.sqrt((X[:,0]-xj[0])**2+(X[:,1]-xj[1])**2)
    return d

def crb_Jloc_power_obs(X,xj,noise_var,gamma):
    """
    Computes CRB for Pn = Ptx -10*gamma*log10(d) + wn
    where Pn is n-th power observation, d distance between xj and power obs 
    location, wn n-th independent noise sample ~N(0,noise_var)

    Parameters
    ----------
    X : power observation locations [N measurements * 2 spatial dimensions]
    xj: true jammer location (parameter to be estimated)
    noise_var: measuerents gaussian noise variance
    gamma: path loss exponent

    Returns
    -------
    crb: Cramer Rao Bound [array of variances of parameter vector]

    """
    d = dist(X,xj)
    a = np.sum( (xj[0]-X[:,0])**2/d**4, axis=0)
    b = np.sum( (xj[1]-X[:,1])**2/d**4 , axis=0)
    c = np.sum( ((xj[0]-X[:,0])*(xj[1]-X[:,1]))/d**4 , axis=0)
    
    coef = (noise_var*np.log(10)**2)/(100*gamma**2)
    
    crb = np.zeros((2,2))
    
    # var(xj)
    crb[0,0] = coef * b/(a*b-c**2) 
    # var(yj)
    crb[1,1] = coef * a/(a*b-c**2) 
    # covar(xj,yj)
    crb[0,1] = coef * -c/(a*b-c**2) 
    crb[1,0] = crb[0,1]

    return crb
    

def mle(x,y,Ptx,sigma,gamma,trueJloc):
    """
    MLE for pathloss
    computed through a coarse grid search and refined through a scipy optimizater
    
    # log-likelihood formula:
    # l = -N/2*np.log(2*np.pi*sigma**2) + ((-1/(2*sigma**2)* tot  ))
    # use only maximizable part
    """
    Nss = 40 # search space size
    
    # define search space (could be enlarged to be safer)
    min_ss = x.min(dim=0) # search space
    min_ss_x = min_ss[0][0] 
    min_ss_y = min_ss[0][1]
    max_ss = x.max(dim=0) # search space
    max_ss_x = max_ss[0][0] 
    max_ss_y = max_ss[0][1]
    
    # search for the max
    xj_v = np.linspace(min_ss_x,max_ss_x,Nss)
    yj_v = np.linspace(min_ss_y,max_ss_y,Nss)
    
    l_mat = np.zeros([xj_v.shape[0],yj_v.shape[0]])

    lmax = -float("inf")
    for ii, xj in enumerate(xj_v):
        for jj, yj in enumerate(yj_v):
            
            l = -pl_logLH([xj,yj],x,y,gamma,Ptx,sigma)           
            l_mat[ii,jj] = l
            
            if l > lmax:
                lmax = l
                xj_mle = xj; yj_mle = yj
    xj_vec_coarse = [xj_mle, yj_mle]
    
    # res  = minimize(pl_logLH,xj_vec_coarse,args=(x,y,gamma,Ptx,sigma),
    #                    method='BFGS')
    # res  = minimize(pl_logLH,xj_vec_coarse,args=(x,y,gamma,Ptx,sigma),
    #                     method='Nelder-Mead',tol=1e-6)
    # xj_vec = res.x
    xj_vec = xj_vec_coarse
    
    err = xj_vec-trueJloc
    
    # import plotly.graph_objects as go
    # import plotly.io as pio
    # pio.renderers.default='browser'
    # #pio.renderers.default='svg'
    # # plot the train data 3D figure
    # fig = go.Figure(data=[go.Surface(x = xj_v,y=yj_v,z=l_mat)])
    # fig.update_layout(title_text='Train data', title_x=0.5)
    # fig.show()
       
    return xj_vec, err

def pl_logLH(xj,x,y,gamma,Ptx,sigma):
    """
    MINUS Path loss log likelihood function 
    logLH function is inverted before return to allow minimization
    (only maximizable part, a constant term has been neglected)
    """
    
    N = x.shape[0]
    tot = 0
    for i in range(N):
        # maximizable part of log-likelihood
        d = np.sqrt((xj[0]-x[i,0])**2 + (xj[1]-x[i,1])**2)
        # compute path loss
        if d<=np.pi:
        # manage log10(0)
            L = 9.942997453882677# Ptx-y[i] # when too close to a datapoint the power is the power of the datapoint
        else:
            L = 10*gamma*np.log10(d)  
            
            # f = 1575.42e6
            # gamma = 2
            # c = 299792458
            # L = 10*gamma*np.log10(4*np.pi*f*d/c) 

        tot += (y[i]-(Ptx-L))**2
    l = -1/(2*sigma**2) * tot 
    return -l


# TO BE PORTED FROM MATLAB
# def llh_grad(theta,Pn_vec,xn_vec,P0,gamma,sigma):
#     """
#     evaluate the function in theta
#     Pn_vec is an array of N observations
#     xn_vec is a NxD matrix of coorinates where observations are taken
#     """
    
#     N = Pn_vec.shape[0]
#     const = -10*gamma/(sigma**2*np.log(10))
#     tot_x = 0
#     tot_y = 0
#     for ii in range(N):
#         obs_error = Pn_vec(ii) -obs_func(theta,xn_vec(ii,:),P0,gamma)
#         d = dist(theta,xn_vec(ii,:))
    
#         tot_x = tot_x+ (obs_error) * (theta(1)-xn_vec(ii,1))/d**2
#         tot_y = tot_y+ (obs_error) * (theta(2)-xn_vec(ii,2))/d**2

#     grad_x = const*tot_x
#     grad_y = const*tot_y
#     grad = [grad_x, grad_y]
    
#     return grad


def initialize_kf(partial_initial_state, time_step, kf_init = None):
    initial_vel = [0,0]
        
    # init KF
    Del = time_step # time step (s)
    transition_matrix = np.array([[1,0,Del,0],[0,1,0,Del],[0,0,1,0],[0,0,0,1]])
    observation_matrix = np.array([[1,0,0,0],[0,1,0,0]])
    n_dim_obs = observation_matrix.shape[0]
    n_dim_state = transition_matrix.shape[0]
    
    # kf_init passed to initialize the matrices
    if kf_init == None:    
        initial_transition_covariance = np.eye(n_dim_state)
        initial_observation_covariance = np.eye(n_dim_obs)
        initial_state_covariance = 30*np.eye(n_dim_state)
        initial_state_mean = np.append(partial_initial_state,initial_vel)
    else:
        initial_transition_covariance = kf_init.transition_covariance
        initial_observation_covariance = kf_init.observation_covariance
        initial_state_covariance = kf_init.initial_state_covariance
        initial_state_mean = kf_init.initial_state_mean
    
    kf = KalmanFilter(
        transition_matrices = transition_matrix, 
        observation_matrices = observation_matrix,
        transition_covariance = initial_transition_covariance,
        observation_covariance = initial_observation_covariance,
        initial_state_mean = initial_state_mean,
        initial_state_covariance = initial_state_covariance
    )
    
    return kf, initial_state_mean, initial_state_covariance
        
def learn_kf_param(observations, partial_initial_state, time_step):        
    
    initial_vel = [0,0]
        
    # init KF
    Del = time_step # time step (s)
    transition_matrix = np.array([[1,0,Del,0],[0,1,0,Del],[0,0,1,0],[0,0,0,1]])
    observation_matrix = np.array([[1,0,0,0],[0,1,0,0]])
    
    # if not specified, transition_covariance, observation_covariance, initial_state_mean, and initial_state_covariance are learned
    initial_state_mean = np.append(partial_initial_state,initial_vel)
    
    kf = KalmanFilter(
        transition_matrices = transition_matrix, 
        observation_matrices = observation_matrix,
        initial_state_mean = initial_state_mean,
    )
    
    kf_learned = kf.em(observations)

    return kf_learned

def compute_cdf(data):
    bins = 15
    # getting data of the histogram
    count, bins_count = np.histogram(data.ravel(), bins=bins)
      
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
      
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return cdf, pdf, bins_count  

def highest_values_based_selection(train_y,test_y,train_x,test_x,
                    constant_data_select_size,ttSplit):
    
    test_size = round(constant_data_select_size*(1/ttSplit-1))
    idx = train_y.argsort(dim=0,descending=True)
    train_y[:] = train_y[idx]            
    train_x[:,:] = train_x[idx,:]            
    idx = test_y.argsort(dim=0,descending=True)
    test_y[:] = test_y[idx]
    test_x[:,:] = test_x[idx,:]                     
    # trim
    train_y = train_y[:constant_data_select_size].detach().clone() 
    train_x = train_x[:constant_data_select_size,:].detach().clone()
    # keep the same training testing ratio        
    test_y = test_y[:test_size].detach().clone()
    test_x = test_x[:test_size,:].detach().clone()
        
    return train_y,test_y,train_x,test_x 


def cov_based_selection(filt_state_cov,filt_state_mean,train_y,test_y,train_x,
                        test_x,sigma_fact,min_data_size,max_data_size):
    # dim = train_x.shape[1]
    dim = 2
    N_train = train_x.shape[0]
    N_test = test_x.shape[0]
   
    stdx = np.sqrt(filt_state_cov[0,0])
    stdy = np.sqrt(filt_state_cov[1,1])
    
    th_x = sigma_fact*stdx
    th_y = sigma_fact*stdy
    
    train_x_select = torch.tensor([]) 
    train_y_select = torch.tensor([])
    first_time = 1
    for i in range(N_train):
        dx = np.abs(train_x[i,0]-filt_state_mean[0])
        dy = np.abs(train_x[i,1]-filt_state_mean[1])
        if dx<th_x and dy<th_y:
            if first_time:
                train_x_select = torch.cat((train_x_select.reshape([-1,1]),train_x[i,:].reshape([-1,1])),0)
                first_time = 0
            else:
                train_x_select = torch.cat((train_x_select,train_x[i,:].reshape([-1,1])),1)                
            train_y_select = torch.cat((train_y_select,train_y[i].unsqueeze(0)),0)
            
    test_x_select = torch.tensor([]) 
    test_y_select = torch.tensor([])      
    first_time = 1
    for i in range(N_test):
        dx = np.abs(test_x[i,0]-filt_state_mean[0])
        dy = np.abs(test_x[i,1]-filt_state_mean[1])
        if dx<th_x and dy<th_y:
            if first_time:
                test_x_select = torch.cat((test_x_select.reshape([-1,1]),test_x[i,:].reshape([-1,1])),0)
                first_time = 0
            else:
                test_x_select = torch.cat((test_x_select,test_x[i,:].reshape([-1,1])),1)                
            test_y_select = torch.cat((test_y_select,test_y[i].unsqueeze(0)),0)
    
    # limit data size
    if test_y_select.shape[0] > max_data_size:
        test_x_select = test_x_select[:,:max_data_size]
        test_y_select = test_y_select[:max_data_size]
    if train_y_select.shape[0] > max_data_size:
        train_x_select = train_x_select[:,:max_data_size]
        train_y_select = train_y_select[:max_data_size]      
        
    # selection failed, return my original tensors        
    if test_y_select.shape[0] < min_data_size or train_y_select.shape[0] < min_data_size:
        return False, train_y, test_y, train_x, test_x
            
    return True, train_y_select.detach().clone(), test_y_select.detach().clone(), train_x_select.transpose(0,1).detach().clone(), test_x_select.transpose(0,1).detach().clone()


def estimate_param_rate(reg_weights_rate_type,param,param_old,fixed_weight_rate = 0.1):
    # can be modified with linear prediction method given the set of past parameters
    
    if reg_weights_rate_type == 'time_instants':
        param_rate = param-param_old
        DeltaT = 1 # time instnts elapsed (over whch rate is computed)(batches don't matter because the dataset is not related to a different jammer location)
    elif reg_weights_rate_type == 'fixed':
        param_rate = fixed_weight_rate
        DeltaT = 1
    else:
        print('unknown parameter rate estimation method')
    return param_rate, DeltaT
