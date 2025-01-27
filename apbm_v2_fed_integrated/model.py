import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """
    A neural network class with customizable architecture and nonlinearity.

    Attributes:
    - input_dim (int): The input dimension of the data.
    - layer_wid (list): A list defining the width of each layer.
    - nonlinearity (str): The type of nonlinearity to apply between layers.
    """
    def __init__(self, input_dim, layer_wid, nonlinearity):
        """
        Initializes the Net class by building a feed-forward neural network.

        Parameters:
        - input_dim (int): The number of input features.
        - layer_wid (list): List of integers defining the number of neurons in each layer.
        - nonlinearity (str): The nonlinearity function to use ('relu', 'sigmoid', etc.).

        Outputs:
        - Initializes the network with specified architecture and activation functions.
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        self.normalization = nn.LayerNorm(input_dim)  # Use LayerNorm for input normalization
        self.batch_norm = nn.BatchNorm1d(input_dim, track_running_stats=False)  # Add BatchNorm for input normalization
        self.dropout = nn.Dropout(p=0.2)  # Add Dropout layer
        self.fc_layers = nn.ModuleList()

        # Create input layer
        self.fc_layers.append(nn.Linear(in_features=input_dim, out_features=layer_wid[0]))

        # Create hidden layers
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], out_features=layer_wid[i + 1]))

        # Apply Xavier Initialization
        self.initialize_weights()

    
        # Set the activation function
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
    
    def initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                torch.nn.init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:  # Initialize bias as zero
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for the neural network.

        Parameters:
        - x (Tensor): Input tensor with dimensions N x input_dim (N = batch size).

        Outputs:
        - (Tensor): Output tensor after passing through the network.
        """
        # x = self.batch_norm(x)
        for fc_layer in self.fc_layers[:-1]:
            batch_normv2 = nn.BatchNorm1d(fc_layer.in_features, track_running_stats=False).to(x.device)
            x = batch_normv2(x)  # Apply batch normalization
            x = self.nonlinearity(fc_layer(x))
            x = self.dropout(x)
        return self.fc_layers[-1](x)

    def get_layers(self):
        """
        Retrieves the input and output dimensions of all layers.

        Outputs:
        - (list): A list containing the input and output dimensions of all layers.
        """
        L = len(self.fc_layers)
        layers = (L + 1) * [0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i + 1] = self.fc_layers[i].out_features
        return layers

    def get_param(self):
        """
        Returns a flattened tensor of all parameters in the network.

        Outputs:
        - (Tensor): A concatenated tensor of all network parameters.
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P, a))
        return P


class Polynomial3(nn.Module):
    """
    A class representing a pathloss model for predicting signal strength based on position.

    Attributes:
    - gamma (float): A learnable parameter representing the path loss exponent.
    - theta (Parameter): A learnable parameter representing the jammer's position.
    - P0 (Parameter): The transmit power to be learned.
    - data_max (float, optional): Maximum value for data normalization.
    - data_min (float, optional): Minimum value for data normalization.
    """
    def __init__(self, gamma=2, theta0=None, data_max=None, data_min=None):
        """
        Initializes the Polynomial3 class with the specified parameters.

        Parameters:
        - gamma (float): Path loss exponent (default is 2).
        - theta0 (list or None): Initial value for theta (position).
        - data_max (float or None): Maximum value for normalization.
        - data_min (float or None): Minimum value for normalization.

        Outputs:
        - Initializes the polynomial model with learnable parameters.
        """
        super().__init__()
        self.theta = nn.Parameter(torch.zeros((2))) if theta0 is None else nn.Parameter(torch.tensor(theta0))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.P0 = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        # self.P0 = nn.Parameter(torch.randn(()))  # Transmit power parameter
        self.data_max = data_max
        self.data_min = data_min
    
    def forward(self, x):
        """
        Forward pass for the polynomial model.

        Parameters:
        - x (Tensor): Input tensor containing position data.

        Outputs:
        - (Tensor): Predicted signal strength.
        """
        L = self.gamma * 10 * torch.log10(torch.norm(x - self.theta, p=2, dim=1))

        # Handle near-field loss
        nearfield_loss = np.log10(np.pi) * self.gamma * 10
        nearfield_loss = nearfield_loss.clone().detach().to(dtype=L.dtype, device=L.device).requires_grad_(True)        
       
        if torch.sum(L < nearfield_loss):
            i = (L < nearfield_loss).nonzero(as_tuple=True)  # Use as_tuple=True for modern PyTorch
            L[i] = nearfield_loss

        fobs = self.P0 - L.unsqueeze(1)
        return fobs

    def get_theta(self):
        """
        Retrieves the learned theta (position parameter).

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.theta
    
    def get_P0(self):
        """
        Retrieves the learned theta P0 (transmit power).

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.P0
    
    def get_gamma(self):
        """
        Retrieves the learned gamma (path loss exponent).

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.gamma

class Net_augmented(nn.Module):
    """
    A class that combines a neural network model and a pathloss model.

    Attributes:
    - model_PL (Polynomial3): Pathloss model instance.
    - model_NN (Net): Neural network model instance.
    - model_mode (str): The mode in which the model operates ('NN', 'PL', 'both').
    """
    def __init__(self, input_dim, layer_wid, nonlinearity, gamma=2, theta0=None):
            """
            Initializes the Net_augmented class with a neural network and pathloss model.

            Parameters:
            - input_dim (int): The input dimension of the data.
            - layer_wid (list): List of integers for the number of neurons in each layer.
            - nonlinearity (str): Nonlinearity function ('relu', 'sigmoid', etc.).
            - gamma (float): Path loss exponent (default is 2).
            - theta0 (list or None): Initial theta value.
            - data_max (float or None): Maximum value for normalization.
            - data_min (float or None): Minimum value for normalization.
            - model_mode (str): The mode of operation ('NN', 'PL', 'both').

            Outputs:
            - Initializes the model with specified parameters.
            """
            super().__init__()
            self.theta = nn.Parameter(torch.zeros((2))) if theta0 is None else nn.Parameter(torch.tensor(theta0))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
            self.P0 = nn.Parameter(torch.tensor(10, dtype=torch.float32))
            # self.P0 = nn.Parameter(torch.randn(()))  # Transmit power parameter
            
            self.input_dim = input_dim
            self.output_dim = layer_wid[-1]
            self.normalization = nn.LayerNorm(input_dim)  # Use LayerNorm for input normalization
            self.batch_norm = nn.BatchNorm1d(input_dim, track_running_stats=False)  # Add BatchNorm for input normalization
            self.dropout = nn.Dropout(p=0.2)  # Add Dropout layer
            self.fc_layers = nn.ModuleList()

            # Create input layer
            self.fc_layers.append(nn.Linear(in_features=input_dim, out_features=layer_wid[0]))

            # Create hidden layers
            for i in range(len(layer_wid) - 1):
                self.fc_layers.append(nn.Linear(in_features=layer_wid[i], out_features=layer_wid[i + 1]))

            # Apply Xavier Initialization
            self.initialize_weights()
            
            # Set the activation function
            if nonlinearity == "sigmoid":
                self.nonlinearity = lambda x: torch.sigmoid(x)
            elif nonlinearity == "relu":
                self.nonlinearity = lambda x: F.relu(x)
            elif nonlinearity == "softplus":
                self.nonlinearity = lambda x: F.softplus(x)
            elif nonlinearity == 'tanh':
                self.nonlinearity = lambda x: torch.tanh(x)
            elif nonlinearity == 'leaky_relu':
                self.nonlinearity = lambda x: F.leaky_relu(x)
            else:
                raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
            
            self.w = nn.Parameter(torch.tensor([0.8, 0.2], requires_grad=True))  # Initialize logits for w_PL and w_NN

    
            
    def initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                torch.nn.init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:  # Initialize bias as zero
                    torch.nn.init.zeros_(layer.bias)
            
    def forward(self, x):
        """
        Forward pass for the augmented model.

        Parameters:
        - x (Tensor): Input tensor.

        Outputs:
        - (Tensor): Combined output based on the model mode.
        """
        w_PL, w_NN = torch.softmax(self.w, dim=0)
        
        return w_PL*self.forward_PL(x) + w_NN*self.forward_NN(x)
    
    def forward_NN(self, x):
        """
        Forward pass for the neural network.

        Parameters:
        - x (Tensor): Input tensor with dimensions N x input_dim (N = batch size).

        Outputs:
        - (Tensor): Output tensor after passing through the network.
        """
        # x = self.batch_norm(x)
        for fc_layer in self.fc_layers[:-1]:
            batch_normv2 = nn.BatchNorm1d(fc_layer.in_features, track_running_stats=False).to(x.device)
            x = batch_normv2(x)  # Apply batch normalization
            x = self.nonlinearity(fc_layer(x))
            x = self.dropout(x)
        return self.fc_layers[-1](x)

    def forward_PL(self, x):
        """
        Forward pass for the polynomial model.

        Parameters:
        - x (Tensor): Input tensor containing position data.

        Outputs:
        - (Tensor): Predicted signal strength.
        """
        L = self.gamma * 10 * torch.log10(torch.norm(x - self.theta, p=2, dim=1))

        # Handle near-field loss
        nearfield_loss = np.log10(np.pi) * self.gamma * 10
        nearfield_loss = nearfield_loss.clone().detach().to(dtype=L.dtype, device=L.device).requires_grad_(True)        
       
        if torch.sum(L < nearfield_loss):
            i = (L < nearfield_loss).nonzero(as_tuple=True)  # Use as_tuple=True for modern PyTorch
            L[i] = nearfield_loss

        fobs = self.P0 - L.unsqueeze(1)
        return fobs

    def get_layers(self):
        """
        Retrieves the input and output dimensions of all layers.

        Outputs:
        - (list): A list containing the input and output dimensions of all layers.
        """
        L = len(self.fc_layers)
        layers = (L + 1) * [0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i + 1] = self.fc_layers[i].out_features
        return layers

    def get_param(self):
        """
        Returns a flattened tensor of all parameters in the network.

        Outputs:
        - (Tensor): A concatenated tensor of all network parameters.
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P, a))
        return P

    def get_theta(self):
        """
        Retrieves the learned theta (position parameter).

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.theta
    
    def get_P0(self):
        """
        Retrieves the learned theta P0 (transmit power).

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.P0
    
    def get_gamma(self):
        """
        Retrieves the learned gamma (path loss exponent).

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.gamma