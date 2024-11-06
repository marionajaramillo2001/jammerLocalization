# -*- coding: utf-8 -*-
"""
Colaboratory file is located at
    https://colab.research.google.com/drive/1HAhKX8jHmMtS5jDPrlF9CkBVWaE7HKBk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as io
import numpy as np
import torch
import model_nn
import sys
sys.path.insert(1, 'BFNN')
import cubature
from kalman import KalmanFilter, StateModel, ObservationModel, NNEKF
import time
import functions as f

# import random
# random.seed(42)
# torch.manual_seed(42)

tic = time.time()

#%% SETTINGS
plot_3D_training_output_flag = 0
toy_example = 0
scramble_data = 0 
ttSplit = 2/3 # train and test split ratio

nn_type = 'ffnn'
# nn_type = 'ekf'
# nn_type = 'ckf'

# model_nn hyper-parameters
layer_wid = [200,100,1]
nonlinearity = 'tanh'
epochs = 500
lr = 0.003

#EKF parameters
layer_wid_ekf = [5,5,1]
noise_pw = 1e-16
nonlinearity_ekf = 'tanh'
init_weights = "rand" # "zeros" #
bias = "independent"#"shared" #   shared bias for each layer or not


#%% --- LOAD DATA

if toy_example:
    T = 50
    N = T*100
    
    train_x_0 = torch.linspace(0, T, steps=N).reshape(-1,1)
    # train_x += 0.01* torch.randn_like(train_x)
    # train_y_0 = torch.sin(10*train_x_0) + torch.sin(5*train_x_0)
    train_y_0 = 3*train_x_0**2 
    # train_y += 0.01* torch.randn_like(train_y)
    
    # test_x_0 = torch.linspace(0,T, steps=round(2/3*N)).reshape(-1,1)
    test_x_0 = torch.linspace(0,T, N).reshape(-1,1)

    # test_x += 0.01* torch.randn_like(test_x)
    # test_y_0 = torch.sin(10*test_x_0) + torch.sin(5*test_x_0)
    test_y_0 = 3*test_x_0**2
    # test_y += 0.01* torch.randn_like(test_y)
    
    if scramble_data:
        # shuffle training datasets
        Idx = list(range(train_x_0.shape[0]))
        np.random.shuffle(Idx)
        Xsh = [train_x_0[i] for i in Idx]
        train_x = torch.tensor(Xsh).reshape(-1,1)
        Ysh = [train_y_0[i] for i in Idx]
        train_y = torch.tensor(Ysh).reshape(-1,1)
        
        # shuffle test datasets
        # Idx = list(range(test_x_0.shape[0]))
        # np.random.shuffle(Idx)
        # Xsh = [test_x_0[i] for i in Idx]
        # test_x = torch.tensor(Xsh).reshape(-1,1)
        # Ysh = [test_y_0[i] for i in Idx]
        # test_y = torch.tensor(Ysh).reshape(-1,1)
        
        test_x = test_x_0
        test_y = test_y_0
    else:
        train_x = train_x_0; test_x = test_x_0
        train_y = train_y_0; test_y = test_y_0
        
else:
    # load true position
    fname_Jloc = 'trueJamLoc.mat'
    data = io.loadmat(fname_Jloc)
    trueJloc = data['Jloc']
    # load matlab data
    fname_x = 'X.mat'
    data = io.loadmat(fname_x)
    X = data['XX']
    fname_y = 'Y.mat'
    data = io.loadmat(fname_y)
    Y = data['YY']
    q = round(X.shape[0]*ttSplit)
    
    # prepare training and test datasets
    # is data a time sequence?
    if len(X.shape)>2:
        T = X.shape[-1]
    else:
        T = 1
    # scramble data
    if scramble_data:
        Idx = list(range(X.shape[0]))
        np.random.shuffle(Idx)
        Xsh = [X[i] for i in Idx]
        Ysh = [Y[i] for i in Idx]
        X = Xsh; Y = Ysh;
    Xtrain = X[:q]; Ytrain = Y[:q]
    Xtest  = X[q:]; Ytest = Y[q:]
    train_x = torch.from_numpy(np.array(Xtrain)).float()
    train_y = torch.from_numpy(np.array(Ytrain)).float()
    test_x = torch.from_numpy(np.array(Xtest)).float()
    test_y = torch.from_numpy(np.array(Ytest)).float()

# normalize the data
train_y = f.normalize(train_y)
test_y = f.normalize(test_y)


#%% --- TRAIN & TEST
if nn_type == 'ffnn':
    # learning
    train_loss, test_loss, model_NN = model_nn.main(train_x,train_y,test_x,test_y,layer_wid,nonlinearity,epochs,lr)
    # train predict
    train_y_predict = model_NN(train_x).detach()
    # test predict
    test_y_predict = model_NN(test_x).detach()
    
    # if toy_example, we're done
    if toy_example:
        import matplotlib.pyplot as plt
        plt.close('all')
        plt.figure(1)
        plt.plot(train_loss,label='train loss')
        plt.plot(test_loss,label='test loss')
        plt.legend()
        
        # train
        plt.figure(2)
        plt.plot(train_y_predict,label='train predict')    
        plt.plot(train_y,'--',label='train target')
        plt.legend()
        
        # test
        plt.figure(3)
        plt.plot(test_y_predict,label='test predict')    
        plt.plot(test_y,'--',label='test target')
        plt.legend()
        sys.exit()
        
    # --- JAMMER POS ESTIMATION
    sampGrid_x, sampGrid_mini, _ = f.getSampleGrid(test_y_predict,test_x, radius = 20, Npoints = 40)
    sampGrid_y = model_NN(sampGrid_x).detach()
    Jloc, err, _ = f.grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc)    

# Not adapted to continual learning!
if nn_type == 'ekf':
    # Init EKF
    input_sz = train_x.size(1)
    nnekf, w_dim = f.initEKF([input_sz]+layer_wid_ekf,noise_pw,nonlinearity_ekf,
                           init_weights,bias)
    print('I1')
    
    # Train EKF
    train_ekf_e = torch.zeros((len(train_x), 1))
    train_ekf_d = torch.zeros_like(train_y)
    for i in range(len(train_x)):
        if not i%100:
            print(i)
        xi = train_x[i].reshape(-1, 1)
        yi, ei = nnekf.forward_backward(xi, train_y[i].reshape(-1, 1))
        train_ekf_d[i] = yi.reshape(train_y.size(dim=1),)
        train_ekf_e[i] = torch.norm(ei)**2
    train_loss = train_ekf_e
    train_y_predict = train_ekf_d.detach()
       
    # Test EKF
    test_ekf_e = torch.zeros((len(test_x), 1))
    test_ekf_d = torch.zeros_like(test_y)
    for i in range(len(test_x)):
        if not i%100:
            print(i)
        xi = test_x[i].reshape(-1, 1)
        # yi, ei = nnekf.forward_backward(xi, test_y[i].reshape(-1, 1))
        yi = nnekf.forward(xi); ei = test_y[i]-yi
        test_ekf_d[i] = yi.reshape(test_y.size(dim=1),)
        test_ekf_e[i] = torch.norm(ei)**2
    test_y_predict = test_ekf_d.detach()    
    test_loss = test_ekf_e
    
    # if toy_example, we're done
    if toy_example:
        import matplotlib.pyplot as plt
        plt.close('all')
        plt.figure(1)
        plt.plot(train_loss,label='train loss')
        plt.plot(test_loss,label='test loss')
        plt.legend()
        
        # train
        plt.figure(2)
        plt.plot(train_y_predict,label='train predict')    
        plt.plot(train_y,'--',label='train target')
        plt.legend()
        
        # test
        plt.figure(3)
        plt.plot(test_y_predict,label='test predict')    
        plt.plot(test_y,'--',label='test target')
        plt.legend()
        sys.exit()
    
    # --- JAMMER POS ESTIMATION
    sampGrid_x, sampGrid_mini, _ = f.getSampleGrid(test_y_predict,test_x, radius = 20, Npoints = 40)
    # Sample EKF
    #sample_ekf_e = torch.zeros((len(sampGrid_x), 1))
    sample_ekf_d = torch.zeros(len(sampGrid_x),layer_wid_ekf[-1])
    for i in range(len(test_x)-1):
        if not i%100:
            print(i)
        xi = sampGrid_x[i].reshape(-1, 1)
        #yi, ei = nnekf.forward_backward(xi, sample_ekf_d[i].reshape(-1, 1))
        yi = nnekf.forward(xi)
        sample_ekf_d[i+1] = yi.reshape(test_y.size(dim=1),)
        #sample_ekf_e[i+1] = torch.norm(ei)**2
    sampGrid_y = sample_ekf_d.detach()
    
    Jloc, err, _ = f.grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc)    

print('Jammer location estimation error (test dataset): ', err)

toc = time.time()
print("%.2f seconds elapsed." % (toc-tic))

#%% --- PLOTS
import matplotlib.pyplot as plt
plt.close('all')
plt.plot(train_loss,label='train')
plt.plot(test_loss,label='test')
plt.legend()
plt.savefig('figs/learning_curve.png')

# ax = plt.axes(projection='3d')
# ax.scatter(train_x[:,0], train_x[:,1], train_y[:,0], c=train_y[:,0], cmap='viridis', linewidth=0.5);

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'

# plot the train data 3D figure
fig = go.Figure(data=[go.Scatter3d(x = train_x[:,0], y = train_x[:,1], z = train_y[:,0], mode='markers', marker=dict(
        size=3,
        color = train_y[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Train data', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/train_data.html")

# plot the train predict data
fig = go.Figure(data=[go.Scatter3dSurface(x = train_x[:,0], y = train_x[:,1], z = train_y_predict[:,0], mode='markers', marker=dict(
        size=3,
        color = train_y_predict[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Train predict', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/train_predict.html")

# plot the test predict data
fig = go.Figure(data=[go.Scatter3d(x = test_x[:,0], y = test_x[:,1], z = test_y_predict[:,0], mode='markers', marker=dict(
        size=3,
        color = train_y_predict[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Test predict', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/test_predict.html")

# plot the  3D figure of train data with predict one MSE
mse_train = (train_y - train_y_predict)**2
mse_train = mse_train.detach().numpy()
fig = go.Figure(data=[go.Scatter3d(x = train_x[:,0], y = train_x[:,1], z = mse_train[:,0], mode='markers', marker=dict(
        size=3,
        color = mse_train[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Train MSE', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/train_mse.html")

# plot the  3D figure of test data with predict one MSE
mse_test = (test_y - test_y_predict)**2
mse_test = mse_test.detach().numpy()
fig = go.Figure(data=[go.Scatter3d(x = test_x[:,0], y = test_x[:,1], z = mse_test[:,0], mode='markers', marker=dict(
        size=3,
        color = mse_test[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Test MSE', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/test_mse.html")

# # sample the model
# fname_grid = 'sampGrid.mat'
# data = io.loadmat(fname_grid)
# sampGrid = data['sampGrid']
# sampGrid_x = torch.from_numpy(np.array(sampGrid)).float()
# sampGrid_y = model_NN(sampGrid_x).detach()
# # plot the train data 3D figure
# fig = go.Figure(data=[go.Scatter3d(x = sampGrid_x[:,0], y = sampGrid_x[:,1], z = sampGrid_y[:,0], mode='markers', marker=dict(
#         size=3,
#         color = train_y[:,0],                # set color to an array/list of desired values
#         colorscale='delta',   # choose a colorscale
#         opacity=0.8
#     ))])
# fig.update_layout(title_text='Model sampling', title_x=0.5)
# fig.show()

# # Print Jammer location estimation error
# # based on oversampled grid
# MM = sampGrid_y.max(dim=0,keepdim=True)[0]
# Idx = sampGrid_y.argmax(dim=0,keepdim=True)[0]
# Jloc = sampGrid[Idx,:]
# fname_Jloc = 'trueJamLoc.mat'
# data = io.loadmat(fname_Jloc)
# trueJloc = data['Jloc']
# err = Jloc - trueJloc
# print('Jammer location estimation error (sample grid): ', err)

# plot the train data 3D figure
fig = go.Figure(data=[go.Scatter3d(x = sampGrid_x[:,0], y = sampGrid_x[:,1], z = sampGrid_y[:,0], mode='markers', marker=dict(
        size=3,
        color = train_y[:,0],                # set color to an array/list of desired values
        colorscale='delta',   # choose a colorscale
        opacity=0.8
    ))])
fig.update_layout(title_text='Model sampling (small area)', title_x=0.5)
if plot_3D_training_output_flag:
    fig.show()
fig.write_html("figs/model_samp.html")



