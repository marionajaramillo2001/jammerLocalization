import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
import numpy as np
import functions as f
import copy


class Net(nn.Module):
    
    def __init__(self, input_dim, layer_wid, nonlinearity):
        """
        """
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        
        self.fc_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(in_features=input_dim, 
                                        out_features=layer_wid[0]))
        
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], 
                                            out_features=layer_wid[i + 1]))

        # Set the nonlinearity
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x) #F.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)
        elif nonlinearity == 'softplus':
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)

    def forward(self, x):
        """
        :param x: input with dimensions N x input_dim where N is number of
            inputs in the batch.
        """
        
        for fc_layer in self.fc_layers[:-1]:
            x = self.nonlinearity(fc_layer(x))
            
        return self.fc_layers[-1](x)
    
    def get_layers(self):
        L = len(self.fc_layers)
        layers = (L+1)*[0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i+1] = self.fc_layers[i].out_features
        return layers
    
    def get_param(self):
        """
        Returns a tensor of all the parameters
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P,a))
        return P
    
    
class Polynomial3(torch.nn.Module):
    def __init__(self,gamma=2,theta0=None,data_max=None,data_min=None):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        if theta0 == None:
            self.theta = nn.Parameter(torch.zeros((2))) # 2-dimensional jammer position. Param to be learned
        else:
            self.theta = nn.Parameter(torch.tensor(theta0))
                        
        self.gamma = gamma
        #self.gamma = nn.Parameter(torch.randn(())) #try to learn it if it can converge
        
        self.P0 = nn.Parameter(torch.randn(())) # tx power. It should be learned
        # self.P0 = 10
        
        self.data_max = data_max
        self.data_min = data_min
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        L = self.gamma*10*torch.log10(torch.norm(x-self.theta,p=2,dim=1))
        
        # when points are too close L<0 are found. Workaround set L=0
        # It works for -inf as well
        nearfield_loss =  self.gamma*10*np.log10(np.pi)
        if torch.sum(L<nearfield_loss):
            #L[L<0] = 10  
            # try to smooth singularities
            i = (L<nearfield_loss).nonzero()
            L[i] = nearfield_loss
        
        fobs = self.P0 - L.unsqueeze(1)
        
        #fobs = f.normalize_maxmin(fobs,self.data_max,self.data_min)
        
        return fobs

    def get_theta(self):
        """
        Get the theta parameter leanred by the PL model
        """
        return self.theta
    
    
# Build a augmented model class    
class Net_augmented(torch.nn.Module):
    def __init__(self,input_dim, layer_wid, nonlinearity,gamma=2,theta0=None,
                 data_max=None,data_min=None):
        super().__init__()
        #super(Net, self).__init__()

        # build the pathloss model
        self.model_PL = Polynomial3(gamma,theta0,data_max,data_min)

        # build the NN network
        self.model_NN = Net(input_dim, layer_wid, nonlinearity)  
    
        
    def forward(self,x):
    
        y = self.model_PL(x) + self.model_NN(x)
        # y = self.model_PL(x)
        # y = self.model_NN(x)

        
        return y
    def get_NN_param(self):
        """
        Returns a tensor of parameter of NN layers
        """
        return self.model_NN.get_param()
    
    def get_theta(self):
        """
        Get the theta parameter leanred by the PL model
        """
        return self.model_PL.get_theta()
    
# define main function, return the train loss and test loss
def main(train_x,train_y,test_x,test_y,layer_wid,nonlinearity,epochs,lr,data_max,data_min):
    """
    layer_wid: NN strucutre. [#Neurons1,#neurons2,output_dimension]
    nonlinearity: activation function. e.g. 'relu'
    epochs: training epochs
    lr: learning rate of neural network
    """
    # prepare the train data and test data
    train_data = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=400, shuffle=True)
    # test_data = TensorDataset(test_x, test_y)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=400, shuffle=True)
    # get the input dimension
    input_dim = train_x.shape[1]
    
    # initialize theta in the hybrid model
    idx = train_y.detach().clone().argmax()
    theta0 = train_x.detach().clone()[idx,:] + torch.randn(2) # avoid singularities
    theta0 = theta0.tolist()
    
    # build the Augmented NN model
    gamma = 2  
    model_aug = Net_augmented(input_dim, layer_wid, nonlinearity,gamma,theta0,data_max,data_min)
    
    # define the loss funciton and optimizer
    loss_function = nn.MSELoss()
    #reg_func = nn.MSELoss()   
    optimizer = torch.optim.Adam(model_aug.parameters(), lr=lr)

    train_mse = torch.zeros(epochs)
    test_mse = torch.zeros(epochs)
    # training
    for epoch in range(epochs):
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            
            y_predict = model_aug(x_batch)

            loss = loss_function(y_predict, y_batch)
            
            # NN model regularization
            beta = 1
            l2_reg = torch.tensor(0.)
            for param in model_aug.model_NN.parameters():
                l2_reg += torch.linalg.norm(param)**2            
            loss += beta * l2_reg    
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_dataloader)
        train_mse[epoch] = train_loss
        # test loss
        
        y_predict_test = model_aug(test_x)
        
        test_loss = loss_function(y_predict_test, test_y)
        test_mse[epoch] = test_loss.detach()

    return train_mse, test_mse, model_aug

    # define main function, return the train loss and test loss
def continual_learn(train_x,train_y,test_x,test_y,model_aug,epochs,lr,data_max,data_min,
                    reg_flag=0,pass_last_weights=0,reg_weights_rate=0,
                    reg_pred_jloc=0,reg_pred_P0=0,*args):
    """
    layer_wid: NN strucutre. [128,64,output_dimension]
    nonlinearity: activation function. 'relu'
    epochs: training epochs
    lr: learning rate of neural network
    """
    # prepare the train data and test data
    train_data = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)
    # test_data = TensorDataset(test_x, test_y)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=400, shuffle=True)
    # get the input dimension
    input_dim = train_x.shape[1]

    # Build new model, randomly init weights, save previously trained param
    param_old = model_aug.get_NN_param()
    param_old_gen = model_aug.model_NN.parameters()
    model_old = copy.deepcopy(model_aug)

    if reg_flag:
        lam = args[0]
        reg_func = nn.MSELoss()
    if reg_weights_rate:
        lam2 = args[1]
        reg_func = nn.MSELoss()
    if not(pass_last_weights):
        layers = model_aug.model_NN.get_layers()    
        nonlinearity = args[2]
        # re init the whole model except theta (NNparam, P0)
        # initialize theta from last theta
        theta0 = model_aug.get_theta().detach().clone().numpy() 
        theta0 = theta0.tolist()
        gamma = 2  
        model_aug = Net_augmented(input_dim, layers[1:], nonlinearity,gamma,theta0,data_max,data_min)
    if reg_pred_jloc:
        reg_func = nn.MSELoss()
        theta_pred = args[5]
        lam3 = args[6]
        
    
    
    
    # idx = train_y.detach().clone().argmax()
    # theta0 = train_x.detach().clone()[idx,:] + torch.randn(2) # avoid singularities
    # theta0 = theta0.tolist()
    # model_aug.model_PL = Polynomial3(2,theta0,data_max,data_min)
    
    
    
    # define the loss funciton and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model_aug.parameters(), lr=lr)

    train_mse = torch.zeros(epochs)
    test_mse = torch.zeros(epochs)
    
    # saving loss and regularization part
    loss_part1 = list()
    loss_part2 = list()
    loss_part3 = list()
    loss_part4 = list()

    param_sequence = torch.tensor(())
    
    
    # training
    for epoch in range(epochs):
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_predict = model_aug(x_batch)
            loss = loss_function(y_predict, y_batch)
            loss_part1.append(loss.detach().clone().numpy())
            
            # NN model regularization wrt APBM
            beta = 1
            l2_reg = torch.tensor(0.)
            for param in model_aug.model_NN.parameters():
                l2_reg += torch.linalg.norm(param)**2            
            loss += beta * l2_reg    

            if reg_flag or reg_weights_rate:
                # # UPDATE WITH SOMETHING WITH GRAD TRUE
                # E.G.:
                # l2_reg = torch.tensor(0.)
                # for param in model_aug.model_NN.parameters():
                #     l2_reg += torch.linalg.norm(param)**2            
                # loss += beta * l2_reg
                param = model_aug.get_NN_param()

                #DEBUG
                param_sequence = torch.cat((param_sequence,param.unsqueeze(-1).detach().clone()),-1)

            if reg_flag:           
                #reg_term = lam*reg_func(param,param_old)
                
                reg_term = torch.tensor(0.)
                for param_old_i,param_i in zip(param_old_gen,model_aug.model_NN.parameters()):
                    reg_term += torch.linalg.norm(param_i-param_old_i)**2   
                reg_term = lam2*reg_term
                
                
                loss_part2.append(reg_term.detach().clone().numpy())
                loss += reg_term
                
            if reg_weights_rate:
                # the last set of weights is ideally the best. So every time I can estimate the rate using the last

                weights_rate = args[3]
                DeltaT = args[4]
                # if Delta_T (and weights_rate) is not empty (i.e. no rate estimations available)
                if DeltaT: 
                    
                    
                    
                    #reg_term = lam2*reg_func(param,param_old+weights_rate*DeltaT)
                    
                    
                    reg_term = torch.tensor(0.)
                    for param_old_i,param_i, w_rates_i in zip(param_old_gen,model_aug.model_NN.parameters(),weights_rate):
                        reg_term += torch.linalg.norm(param_i-(param_old_i+w_rates_i*DeltaT))**2   
                    reg_term = lam2*reg_term
                    
                        
                    loss_part3.append(reg_term.detach().clone().numpy())
                    loss += reg_term
                    
            if reg_pred_jloc:
                theta_now = model_aug.get_theta() #model_aug.get_theta().detach().clone()
                theta_pred = torch.tensor(theta_pred) + torch.randn(2)
                #reg_term = lam3*reg_func(theta_pred,theta_now)
                reg_term = lam3*sum((theta_now-theta_pred)**2)
                loss_part4.append(reg_term.detach().clone().numpy())
                loss += reg_term
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_dataloader)
        train_mse[epoch] = train_loss
        # test loss
        y_predict_test = model_aug(test_x)
        test_loss = loss_function(y_predict_test, test_y)
        if reg_flag:
            param = model_aug.get_NN_param()
            test_loss += lam*reg_func(param,param_old)
        test_mse[epoch] = test_loss.detach()
        
        
    # DEBUG: parameters inspection    
    if False:    
        import matplotlib.pyplot as plt
        plt.close('all')
        A = np.random.uniform(0,param_sequence.shape[0],20)   
        fig, axs = plt.subplots(2)
        fig.suptitle('Parameter dynamics (random selection)')
        axs[0].plot(param_sequence[A,:].transpose(0,1))
        axs[0].set_ylabel('parameter')
        axs[1].plot(param_sequence[A,:].diff(dim=1).transpose(0,1))
        axs[1].set_xlabel('batches through epochs')
        axs[1].set_ylabel('parameter rate')
    
        fig, axs = plt.subplots(2)
        fig.suptitle('Parameter dynamics (average)')
        axs[0].plot(param_sequence.mean(0))
        axs[0].set_ylabel('average parameter')
        axs[1].plot(param_sequence.mean(0).diff())
        axs[1].set_xlabel('batches through epochs')
        axs[1].set_ylabel('average parameter rate')
        
        
    loss_parts = [loss_part1,loss_part2,loss_part3, loss_part4]
    
    return train_mse, test_mse, model_aug, loss_parts, param_old, model_old


'''
example
'''
if __name__ == "__main__":
    # build the NN netwokr
    input_dim=2
    layer_wid=[128,32,1]
    nonlinearity='relu'
    model_NN = Net(input_dim, layer_wid, nonlinearity)
    # data processing
    #train_x = [200,2,10], train_y = [200,1,10]
    times=10
    epochs=100
    lr=0.002
    # continual train
    train_mses = []
    test_mses = []
    for i in range(times):
        # every_epoch, model_NN will update
        train_mse, test_mse, model_NN = continual_learn(train_x[:,:,i],train_y[:,:,i],test_x[:,:,i],test_y[:,:,i],model_NN,epochs,lr)    
        train_mses.append(train_mse)
        test_mses.append(test_mse)
        

# example and tutorials
"""
https://www.machinecurve.com/index.php/2021/07/20/how-to-create-a-neural-network-for-regression-with-pytorch/
https://github.com/yunjey/pytorch-tutorial
https://github.com/jcjohnson/pytorch-examples
https://github.com/kevinzakka/pytorch-goodies
"""
