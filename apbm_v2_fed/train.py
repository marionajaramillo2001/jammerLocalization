import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import copy
from functools import partial
import torch.optim as optim
from apbm.model import Net_augmented


class Train(object):
    def __init__(self, model_nn, model_pl, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, batch_size, max_epochs_nn, max_epochs_pl):
        """
        Initializes the CrossVal class for cross-validation with early stopping.

        Parameters:
        - config (dict): Dictionary containing the configuration parameters.
        - batch_size (int): Batch size for data loaders.
        - optimizer_nn (function): Optimizer for the neural network parameters.
        - optimizer_theta (function): Optimizer for the theta parameter.
        - optimizer_P0 (function): Optimizer for the P0 parameter.
        - optimizer_gamma (function): Optimizer for the gamma parameter.
        - max_epochs (int): Maximum number of epochs for training.
        
        Output:
        Initializes the class attributes and settings.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.server_model_nn = model_nn
        self.server_model_pl = model_pl
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.optimizer_P0 = optimizer_P0
        self.optimizer_gamma = optimizer_gamma
        self.local_epochs_nn = 50
        self.local_epochs_pl = 50
        self.max_epochs_nn = max_epochs_nn  # Number of epochs for tuning the NN alone 
        self.max_epochs_pl = max_epochs_pl
        self.batch_size = batch_size
        self.num_rounds_nn = 10
        self.num_rounds_pl = 10
        self.weights_nn_type = 'max'
        self.weights_pl_type = 'max'
        
    def compute_loss_nn(self, output, target, weights):
        lambda_mse_loss_total = 1.0
        error = torch.norm(output - target, dim=1)
        mse_loss = torch.mean(error*weights)

        # # Add L2 regularization for NN parameters
        # l2_lambda = 0.00  # Regularization strength
        # l2_loss = 0
        # for param in model.parameters():
        #     l2_loss += torch.norm(param, 2)  # Compute L2 norm for all NN parameters
            
        loss = lambda_mse_loss_total*mse_loss # + l2_lambda * l2_loss
        
        return loss
        
    def compute_loss_pl(self, output, target, weights):
        lambda_mse_loss_total = 1.0
        error = torch.norm(output - target, dim=1)
        mse_loss = torch.mean(error*weights)

        loss = lambda_mse_loss_total*mse_loss
        
        return loss
    
    def model_update(self, dataloader, round, training_phase):
        dataset = dataloader.dataset
        
        indices = list(range(len(dataset)))  # Get all indices from the train dataset
        data, target = dataset[indices]  # Retrieve full dataset and target values
        
        if training_phase == 'NN':
            model_nn = copy.deepcopy(self.server_model_nn)
            model_nn.to(self.device)
            # Initialize optimizers
            optimizer_nn = self.optimizer_nn(list(model_nn.parameters()))
            
            if self.weights_nn_type == 'max':
                weights_nn = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
                weights_nn = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
            elif self.weights_nn_type == 'max_quadratic':
                weights_nn = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
                weights_nn = weights_nn**2
                weights_nn = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
            elif self.weights_nn_type == 'min_max':
                weights_nn = (target - torch.mean(target)).abs() + 1e-6
                
            model_nn.train()
            loss = 0
            for epoch in range(self.local_epochs_nn):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Select corresponding weights for the current batch
                    batch_weights = weights_nn[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].to(self.device)

                    # Zero the gradients
                    optimizer_nn.zero_grad()
                    output = model_nn(data)
                    loss = self.compute_loss_nn(output, target, batch_weights)
                    # # set if fedprox
                    # if self.algorithm == 'Fedprox':
                    #     for w, w_t in zip(model.parameters(), self.server_model.parameters()):
                    #         loss += self.mu/2*(w - w_t).norm(2)**2
                    # elif self.algorithm == 'Fedprox-PL':
                    #     l2_reg = 0
                    #     for w in model.model_NN.parameters():
                    #         l2_reg += self.betas[round]*torch.linalg.norm(w)**2
                    #     loss += self.mu*l2_reg

                    # Backward pass and gradient clipping
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)
                    optimizer_nn.step()
                    
                    loss += loss.item()
                    
            return model_nn, loss / self.local_epochs_nn
        
        elif training_phase == 'PL':
            model_pl = copy.deepcopy(self.server_model_pl)
            model_pl.to(self.device)
            # Initialize optimizers
            optimizer_theta = self.optimizer_theta([model_pl.theta])
            optimizer_P0 = self.optimizer_P0([model_pl.P0])
            optimizer_gamma = self.optimizer_gamma([model_pl.gamma])
            
            # scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=5, gamma=0.5)

            
            if self.weights_pl_type == 'max':
                weights_pl = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
                weights_pl = weights_pl / weights_pl.sum()  # Normalize weights to sum to 1
            elif self.weights_pl_type == 'max_quadratic':
                weights_pl = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
                weights_pl = weights_pl**2
                weights_pl = weights_pl / weights_pl.sum()  # Normalize weights to sum to 1
            elif self.weights_pl_type == 'min_max':
                weights_pl = (target - torch.mean(target)).abs() + 1e-6
            
            model_pl.train()
            loss = 0
            for epoch in range(self.local_epochs_pl):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Select corresponding weights for the current batch
                    batch_weights = weights_pl[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].to(self.device)

                    # Zero the gradients
                    optimizer_theta.zero_grad()
                    optimizer_P0.zero_grad()
                    optimizer_gamma.zero_grad()
                    
                    output = model_pl(data)
                    loss = self.compute_loss_pl(output, target, batch_weights)
                    # # set if fedprox
                    # if self.algorithm == 'Fedprox':
                    #     for w, w_t in zip(model.parameters(), self.server_model.parameters()):
                    #         loss += self.mu/2*(w - w_t).norm(2)**2
                    # elif self.algorithm == 'Fedprox-PL':
                    #     l2_reg = 0
                    #     for w in model.model_NN.parameters():
                    #         l2_reg += self.betas[round]*torch.linalg.norm(w)**2
                    #     loss += self.mu*l2_reg

                    # Backward pass and gradient clipping
                    loss.backward()
                    # Gradient clipping
                    max_norm_theta = max(1.0, 5.0 * (1 - epoch / self.max_epochs_pl))  # Dynamic for theta
                    nn.utils.clip_grad_norm_([model_pl.theta], max_norm=max_norm_theta)

                    optimizer_theta.step()
                    optimizer_P0.step()
                    optimizer_gamma.step()
                    
                    loss += loss.item()
                    
            return model_pl, loss / self.local_epochs_pl
        
    def server_update(self, model_list, uploaded_weights, training_phase):
        """
        Update the server model using the Federated Averaging (FedAvg) algorithm.
        Args:
            model_list (list): A list of client models to be averaged.
            uploaded_weights (list): A list of weights corresponding to each client model.
        Returns:
            The updated server model.
        Notes:
            - The server model's parameters are first set to zero.
            - Each client's model parameters are weighted by the corresponding value in `uploaded_weights` and added to the server model's parameters.
            - The server model's neural network (model_NN) and probabilistic layer (model_PL) are updated accordingly.
        """
        if training_phase == 'NN':
            with torch.no_grad():
                for name, param in self.server_model_nn.named_parameters():
                    param.data.zero_()
            for w, client_model_nn in zip(uploaded_weights, model_list):
                for server_nn_param, client_nn_param in zip(self.server_model_nn.parameters(), client_model_nn.parameters()):
                    server_nn_param.data += client_nn_param.data * w
            return self.server_model_nn
        
        elif training_phase == 'PL':
            with torch.no_grad():
                self.server_model_pl.get_theta().data.zero_()
            for w, client_model_pl in zip(uploaded_weights, model_list):
                self.server_model_pl.theta.data += client_model_pl.theta.data * w
                self.server_model_pl.P0.data += client_model_pl.P0.data * w
                self.server_model_pl.gamma.data += client_model_pl.gamma.data * w
            return self.server_model_pl
                
    
    def get_theta_init(self, model_nn, train_loader_splitted):
        positions, measurements = [], []

        for train_loader in train_loader_splitted:
            for batch_idx, (data, target) in enumerate(train_loader):
                positions.append(data.numpy())
                measurements.append(target.numpy())
            
        # Initialize theta0 to the position with the highest output from the neural network
        positions = np.concatenate(positions, axis=0)
        measurements = np.concatenate(measurements, axis=0)

        # Define grid range
        min_x, max_x = int(np.min(positions[:, 0])), int(np.max(positions[:, 0]))
        min_y, max_y = int(np.min(positions[:, 1])), int(np.max(positions[:, 1]))

        model_nn.eval()

        # Generate grid ranges for x and y
        grid_range_x = np.linspace(min_x - 1, max_x + 1, max_x - min_x + 3)
        grid_range_y = np.linspace(min_y - 1, max_y + 1, max_y - min_y + 3)

        # Create a 2D grid combining all points
        grid_x, grid_y = torch.meshgrid(torch.tensor(grid_range_x), torch.tensor(grid_range_y), indexing="ij")
        grid_tensor = torch.stack([grid_x, grid_y], dim=-1)  # Shape (N, N, 2)

        # Flatten the grid for batch evaluation
        grid_points = grid_tensor.view(-1, 2)  # Shape (N^2, 2)

        # Compute Z values for all grid points
        with torch.no_grad():
            Z = model_nn(grid_points.float()).view(grid_x.shape)  # Reshape back to (N, N)

        # Convert grid_x and grid_y to NumPy arrays
        X = grid_x.numpy()
        Y = grid_y.numpy()
                    
        # Create a 3D plot using Plotly
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_layout(
            title='3D Surface Plot',
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],  
            y=positions[:, 1],  
            z=measurements.flatten(), 
            mode='markers',
            marker=dict(size=4),
            name='Training Points'
        ))
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.01
        ))
        
        fig.show()

        # Find local maxima
        Z_np = Z.numpy()  # Convert to NumPy for easier processing
        local_maxima = []
        margin = 2  # Exclude a 5-point margin from the borders

        for i in range(margin, Z_np.shape[0] - margin):  # Exclude margin points
            for j in range(margin, Z_np.shape[1] - margin):  # Exclude margin points
                local_maxima.append((i, j, Z_np[i, j]))

        # Select the global maximum among local maxima
        if local_maxima:
            local_maxima = sorted(local_maxima, key=lambda x: x[2], reverse=True)  # Sort by Z value
            max_i, max_j, _ = local_maxima[0]  # Top local maximum
        else:
            raise ValueError("No local maxima found. Check the input field.")

        # Retrieve the corresponding x and y coordinates
        max_position = torch.tensor([grid_x[max_i, max_j], grid_y[max_i, max_j]])
        
        return nn.Parameter(max_position)
                
    
    def upload_weight(self, train_samples):
        """
        Update the weight of each client based on the number of training samples.

        Args:
            train_samples (list): A list containing the number of samples for each client.

        Returns:
            torch.Tensor: A tensor containing the weight of each client.
        """
        ws = []
        for i in range(len(train_samples)):
            ws.append(train_samples[i] / sum(train_samples))
        ws = torch.tensor(ws)
        return ws
    
    def train_test(self, train_loader_splitted, test_loader, real_loc, train_y_mean_splited):  
        train_samples = torch.zeros(len(train_loader_splitted))
        for i in range(len(train_loader_splitted)):
            train_samples[i]=len(train_loader_splitted[i].dataset)

        uploaded_weights = self.upload_weight(train_samples)
        
        # Start training the NN to get a good initialization for theta
        training_phase = 'NN' # Variable to indicate the training phase
        print("Training the NN model...")
    
        train_losses_nn_per_round = []
        for round in range(self.num_rounds_nn):
            model_nn_list = []
            weighted_average_loss_clients = 0
            for i in range(len(train_loader_splitted)):
                model_nn, loss_nn = self.model_update(train_loader_splitted[i], round, training_phase)
                print('Client %d: train_nn_loss = %f' % (i, loss_nn))
                model_nn_list.append(model_nn)
                weighted_average_loss_clients += loss_nn * uploaded_weights[i]
            train_losses_nn_per_round.append(weighted_average_loss_clients.detach().numpy() )

            self.server_model_nn = self.server_update(model_nn_list, uploaded_weights, training_phase)

            global_loss_nn = self.test_reg(self.server_model_nn, test_loader)
            print('Round %d: test_loss = %f' % (round, global_loss_nn))
                
        # Get theta initialization
        theta_init = self.get_theta_init(self.server_model_nn, train_loader_splitted)
        self.server_model_pl.theta = theta_init

        print(f"Initial theta: {theta_init}")
        
        # Train using the PL model
        training_phase = 'PL'
        print("Training the PL model...")
        
        train_losses_pl_per_round = []
        for round in range(self.num_rounds_pl):
            model_pl_list = []
            weighted_average_loss_clients = 0
            for i in range(len(train_loader_splitted)):
                model_pl, loss_pl = self.model_update(train_loader_splitted[i], round, training_phase)
                print('Client %d: train_pl_loss = %f' % (i, loss_pl))
                model_pl_list.append(model_pl)
                weighted_average_loss_clients += loss_pl * uploaded_weights[i]
            train_losses_pl_per_round.append(weighted_average_loss_clients.detach().numpy() )

            self.server_model_pl = self.server_update(model_pl_list, uploaded_weights, training_phase)
            global_loss_pl = self.test_reg(self.server_model_pl, test_loader)
            loc_error = self.test_loc(real_loc)
            print('Round %d: test_loss = %f' % (round, global_loss_pl))
            print('Round %d:', (round, loc_error))
                
        # Find the point in the train_dataset that is closest to the real_loc position
        closest_point = None
        min_distance = float('inf')

        for train_loader in train_loader_splitted:
            for data, _ in train_loader:
                distances = torch.norm(data - torch.tensor(real_loc, device=self.device), dim=1)
                min_idx = torch.argmin(distances)
                if distances[min_idx] < min_distance:
                    min_distance = distances[min_idx]
                    closest_point = data[min_idx].cpu().numpy()

        print(f"Closest point in the train dataset to the real location: {closest_point}")
        print(f"Minimum distance to the real location: {min_distance}")
        
        # Evaluate on the global test set
        global_test_nn_loss = self.test_reg(self.server_model_nn, test_loader)
        global_test_pl_loss = self.test_reg(self.server_model_pl, test_loader)
        jam_loc_error, predicted_jam_loc = self.test_loc(real_loc)
        learnt_P0, learnt_gamma = self.get_learnt_parameters(self.server_model_pl)
        
        
        return train_losses_nn_per_round, train_losses_pl_per_round, global_test_nn_loss, global_test_pl_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, model_nn, model_pl


    def test_reg(self, model, test_loader):
        """
        Evaluates the model on a test set.

        Parameters:
        - model (nn.Module): The trained model.
        - test_loader (DataLoader): DataLoader for the test set.

        Output:
        - avg_test_loss (float): The average test loss.
        """
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0  # Initialize total test loss
              
        test_target = []
        for _, target in test_loader:
            test_target.append(target)

        test_target = torch.cat(test_target, dim=0) 
        
        if self.weights_pl_type == 'max':
            test_weights = (test_target - torch.min(test_target)) / (torch.max(test_target) - torch.min(test_target) + 1e-6)
            test_weights = test_weights / test_weights.sum()  # Normalize weights to sum to 1
        elif self.weights_pl_type == 'max_quadratic':
            test_weights = (test_target - torch.min(test_target)) / (torch.max(test_target) - torch.min(test_target) + 1e-6)
            test_weights = test_weights**2
            test_weights = test_weights / test_weights.sum()
        elif self.weights_pl_type == 'min_max':
            test_weights = (test_target - torch.mean(test_target)).abs() + 1e-6
            
        with torch.no_grad():  # Disable gradient computation for testing
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_test_loss += self.compute_loss_pl(output, target, test_weights)

        # Calculate average test loss
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        
        return avg_test_loss
    
    def test_loc(self, real_loc):
        """
        Computes the localization error based on the model's learned parameters.

        Parameters:
        - real_loc (ndarray): The true location for comparison.

        Output:
        - result (float): The computed localization error.
        """
        predicted_jam_loc = self.server_model_pl.get_theta().detach().numpy()
        jam_loc_error = np.sqrt(np.mean((real_loc - predicted_jam_loc)**2))
        
        return jam_loc_error, predicted_jam_loc
    

    def get_learnt_parameters(self, model):
        """
        Get the learned parameters P0 and gamma from the model.

        Parameters:
        - model (nn.Module): The trained model.

        Output:
        - P0 (float): The learned P0 parameter.
        - gamma (float): The learned gamma parameter.
        """
        P0 = model.P0.item()
        gamma = model.gamma.item()
        
        return P0, gamma
