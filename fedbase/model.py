import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                 data_max=None,data_min=None,model_mode='both'):
        super().__init__()
        #super(Net, self).__init__()

        # build the pathloss model
        self.model_PL = Polynomial3(gamma,theta0,data_max,data_min)

        # build the NN network
        self.model_NN = Net(input_dim, layer_wid, nonlinearity)  
        self.model_mode = model_mode
        # self.w = nn.Parameter(torch.zeros((2)))
        # self.w = nn.Parameter(torch.rand(2))
        self.w = nn.Parameter(torch.tensor([0.5,0.5]))
        
    def forward(self,x):
        if self.model_mode == 'NN':
            y = self.model_NN(x)
        elif self.model_mode == 'PL':
            y = self.model_PL(x)
        else:
            # y = self.w[0]*self.model_PL(x) + self.w[1]*self.model_NN(x)
            # y = self.w * self.model_PL(x) + self.model_NN(x)
            y = self.model_PL(x) + self.model_NN(x)
            # y = self.w * self.model_PL(x) + (1-self.w)*self.model_NN(x)
            # # normalize w
            # self.w.data = self.w.data/torch.sum(self.w.data)
            # y = self.w[0]*self.model_PL(x) + self.w[1]*self.model_NN(x)
            # y=0.0001*self.model_PL(x)+self.model_NN(x)
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
    
# Define the gating network
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    

# Build a augmented model class    
class Net_augmented_gate(torch.nn.Module):
    def __init__(self,input_dim, layer_wid, nonlinearity,gamma=2,theta0=None,
                 data_max=None,data_min=None,model_mode='both',num_experts=2):
        super().__init__()
        #super(Net, self).__init__()

        # build the pathloss model
        self.model_PL = Polynomial3(gamma,theta0,data_max,data_min)

        # build the NN network
        self.model_NN = Net(input_dim, layer_wid, nonlinearity) 
        self.experts = nn.ModuleList([Net(input_dim, layer_wid, nonlinearity) for _ in range(num_experts)]) 
        # add the self.model_PL as the last expert
        self.experts.append(self.model_PL)
        self.gating_network = GatingNetwork(input_dim, num_experts+1)
        self.model_mode = model_mode
        # self.w = nn.Parameter(torch.zeros((2)))
        # self.w = nn.Parameter(torch.rand(2))
        self.w = nn.Parameter(torch.tensor([0.5,0.5]))
        
    def forward(self,x):
        if self.model_mode == 'NN':
            y = self.model_NN(x)
        elif self.model_mode == 'PL':
            y = self.model_PL(x)
        else:
            gating_weights = self.gating_network(x)
            expert_outputs = torch.stack([expert(x) for expert in self.experts[:-1]], dim=2)
            expert_outputs = torch.cat([expert_outputs, self.model_PL(x).unsqueeze(2)], dim=2)
            y = torch.sum(gating_weights.unsqueeze(1) * expert_outputs, dim=2)

            # y = self.w[0]*self.model_PL(x) + self.w[1]*self.model_NN(x)
            # y = self.w * self.model_PL(x) + self.model_NN(x)
            # y = self.model_PL(x) + self.model_NN(x)
            # y = self.w * self.model_PL(x) + (1-self.w)*self.model_NN(x)
            # normalize w
            # self.w.data = self.w.data/torch.sum(self.w.data)
            # y = self.w[0]*self.model_PL(x) + self.w[1]*self.model_NN(x)
            # y=0.0001*self.model_PL(x)+self.model_NN(x)
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