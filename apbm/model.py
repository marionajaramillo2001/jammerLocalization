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
        for fc_layer in self.fc_layers[:-1]:
            x = self.nonlinearity(fc_layer(x))
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
        self.P0 = nn.Parameter(torch.tensor(0, dtype=torch.float32))
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
        nearfield_loss = self.gamma * 10 * np.log10(np.pi)
        if torch.sum(L < nearfield_loss):
            i = (L < nearfield_loss).nonzero()
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
    def __init__(self, input_dim, layer_wid, nonlinearity, gamma=2, theta0=None, data_max=None, data_min=None, model_mode='both'):
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
        self.model_PL = Polynomial3(gamma, theta0, data_max, data_min)
        self.model_NN = Net(input_dim, layer_wid, nonlinearity)
        self.model_mode = model_mode
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        """
        Forward pass for the augmented model.

        Parameters:
        - x (Tensor): Input tensor.

        Outputs:
        - (Tensor): Combined output based on the model mode.
        """
        if self.model_mode == 'NN':
            y = self.model_NN(x)
        elif self.model_mode == 'PL':
            y = self.model_PL(x)
        else:
            w_PL, w_NN = torch.softmax(self.w, dim=0)
            y = w_PL * self.model_PL(x) + w_NN * self.model_NN(x)
        return y

    def get_NN_param(self):
        """
        Retrieves the parameters of the neural network model.

        Outputs:
        - (Tensor): Flattened tensor of the neural network parameters.
        """
        return self.model_NN.get_param()

    def get_theta(self):
        """
        Retrieves the learned theta from the pathloss model.

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.model_PL.get_theta()
    
    def get_P0(self):
        """
        Retrieves the learned P0 from the pathloss model.

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.model_PL.get_P0()
    
    def get_gamma(self):
        """
        Retrieves the learned gamma from the pathloss model.

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.model_PL.get_gamma()


class GatingNetwork(nn.Module):
    """
    A gating network that produces weights for combining expert model outputs.

    Attributes:
    - input_size (int): The input dimension.
    - num_experts (int): The number of experts (models) to combine.
    """
    def __init__(self, input_size, num_experts):
        """
        Initializes the GatingNetwork class.

        Parameters:
        - input_size (int): The input dimension.
        - num_experts (int): Number of expert models.

        Outputs:
        - Initializes a fully connected layer followed by softmax activation.
        """
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass for the gating network.

        Parameters:
        - x (Tensor): Input tensor.

        Outputs:
        - (Tensor): Weights for combining expert outputs.
        """
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class Net_augmented_gate(nn.Module):
    """
    An augmented model that combines neural networks and a pathloss model using a gating network.

    Attributes:
    - model_PL (Polynomial3): The pathloss model instance.
        - model_NN (Net): The main neural network instance.
    - experts (ModuleList): List of expert models, including multiple neural networks and the pathloss model.
    - gating_network (GatingNetwork): The network that generates weights for combining the expert outputs.
    - model_mode (str): The mode in which the model operates ('NN', 'PL', 'both').

    Methods:
    - __init__: Initializes the model components.
    - forward: Combines outputs from experts based on weights from the gating network.
    - get_NN_param: Retrieves the parameters of the neural network.
    - get_theta: Retrieves the learned theta from the pathloss model.
    - get_P0: Retrieves the learned P0 from the pathloss model.
    - get_gamma: Retrieves the learned gamma from the pathloss model.
    """
    def __init__(self, input_dim, layer_wid, nonlinearity, gamma=2, theta0=None, data_max=None, data_min=None, model_mode='both', num_experts=2):
        """
        Initializes the Net_augmented_gate class with neural networks, pathloss model, and gating network.

        Parameters:
        - input_dim (int): The input dimension.
        - layer_wid (list): List of integers defining the width of each layer in the neural network.
        - nonlinearity (str): The nonlinearity function ('relu', 'sigmoid', etc.).
        - gamma (float): Path loss exponent (default is 2).
        - theta0 (list or None): Initial theta value for the pathloss model.
        - data_max (float or None): Maximum value for normalization.
        - data_min (float or None): Minimum value for normalization.
        - model_mode (str): Mode for combining models ('NN', 'PL', 'both').
        - num_experts (int): Number of expert models (default is 2).

        Outputs:
        - Initializes the augmented model with specified parameters.
        """
        super().__init__()
        self.model_PL = Polynomial3(gamma, theta0, data_max, data_min)
        self.model_NN = Net(input_dim, layer_wid, nonlinearity)
        self.experts = nn.ModuleList([Net(input_dim, layer_wid, nonlinearity) for _ in range(num_experts)])
        self.experts.append(self.model_PL)  # Add the pathloss model as the last expert
        self.gating_network = GatingNetwork(input_dim, num_experts + 1)
        self.model_mode = model_mode
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        """
        Forward pass for the augmented gating model.

        Parameters:
        - x (Tensor): Input tensor.

        Outputs:
        - (Tensor): Combined output from the experts using the gating network's weights.
        """
        if self.model_mode == 'NN':
            y = self.model_NN(x)
        elif self.model_mode == 'PL':
            y = self.model_PL(x)
        else:
            # Get the weights from the gating network
            gating_weights = self.gating_network(x)
            # Compute outputs from all experts
            expert_outputs = torch.stack([expert(x) for expert in self.experts[:-1]], dim=2)
            expert_outputs = torch.cat([expert_outputs, self.model_PL(x).unsqueeze(2)], dim=2)
            # Combine outputs using the gating weights
            y = torch.sum(gating_weights.unsqueeze(1) * expert_outputs, dim=2)
        return y

    def get_NN_param(self):
        """
        Retrieves the parameters of the neural network model.

        Outputs:
        - (Tensor): Flattened tensor of the neural network parameters.
        """
        return self.model_NN.get_param()

    def get_theta(self):
        """
        Retrieves the learned theta from the pathloss model.

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.model_PL.get_theta()
    
    def get_P0(self):
        """
        Retrieves the learned P0 from the pathloss model.

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.model_PL.get_P0()
    
    def get_gamma(self):
        """
        Retrieves the learned gamma from the pathloss model.

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.model_PL.get_gamma()