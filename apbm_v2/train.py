import torch
import torch.nn as nn
import numpy as np
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
        self.initial_model_nn = model_nn
        self.initial_model_pl = model_pl
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.optimizer_P0 = optimizer_P0
        self.optimizer_gamma = optimizer_gamma
        self.max_epochs_nn = max_epochs_nn  # Number of epochs for tuning the NN alone 
        self.max_epochs_pl = max_epochs_pl
        self.batch_size = batch_size
        self.num_rounds = 1  # For non-FL cases, this is 1 (kept for compatibility)
        self.weights_nn_type = 'max'
        self.weights_pl_type = 'max'
        
    def compute_loss_nn(self, model, data, target, weights):
        lambda_mse_loss_total = 1.0
        output = model(data)
        error = torch.norm(output - target, dim=1)
        mse_loss = torch.mean(error*weights)

        # Add L2 regularization for NN parameters
        l2_lambda = 0.00  # Regularization strength
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, 2)  # Compute L2 norm for all NN parameters
            
        loss = lambda_mse_loss_total*mse_loss + l2_lambda * l2_loss
        
        return loss
        
    def compute_loss_pl(self, model, data, target, weights):
        lambda_mse_loss_total = 1.0
        output = model(data)
        error = torch.norm(output - target, dim=1)
        mse_loss = torch.mean(error*weights)

        loss = lambda_mse_loss_total*mse_loss
        
        return loss

    def train_one_epoch_nn(self, model_nn, train_loader, optimizer_nn, epoch, full_weights):
        """
        Trains the model for one epoch with hybrid tuning (theta first, then theta + NN).
        
        Parameters:
        - model (nn.Module): The model to be trained.
        - train_loader (DataLoader): DataLoader for the training set.
        - optimizer_nn (torch.optim.Optimizer): Optimizer for neural network parameters.
        - epoch (int): Current epoch number.
        
        Output:
        - avg_train_loss (float): The average training loss for the epoch.
        """
        model_nn.train()  # Set the model to training mode
        total_train_loss = 0  # Initialize total training loss
                
        positions, measurements = [], []

        for batch_idx, (data, target) in enumerate(train_loader):
            positions.append(data.numpy())
            measurements.append(target.numpy())
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Select corresponding weights for the current batch
            batch_weights = full_weights[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].to(self.device)

            # Zero the gradients
            optimizer_nn.zero_grad()
            loss = self.compute_loss_nn(model_nn, data, target, batch_weights)

            # Backward pass and gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)

            optimizer_nn.step()

            total_train_loss += loss.item() * len(data)

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / sum(len(data) for data, _ in train_loader)
        
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
        
        # print(f"Max position: {max_position}")

        # Return the initialized theta with some added random noise
        return avg_train_loss, nn.Parameter(max_position)
    
    def get_theta_init(self, model_nn, train_loader):
        positions, measurements = [], []

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
                
    def train_one_epoch_pl(self, model_pl, train_loader, optimizer_theta, optimizer_P0, optimizer_gamma, epoch, full_weights):
        """
        Trains the model for one epoch with hybrid tuning (theta first, then theta + NN).
        
        Parameters:
        - model (nn.Module): The model to be trained.
        - train_loader (DataLoader): DataLoader for the training set.
        - optimizer_nn (torch.optim.Optimizer): Optimizer for neural network parameters.
        - optimizer_theta (torch.optim.Optimizer): Optimizer for theta parameters.
        - optimizer_P0 (torch.optim.Optimizer): Optimizer for P0 parameters.
        - optimizer_gamma (torch.optim.Optimizer): Optimizer for gamma parameters.
        - epoch (int): Current epoch number.
        
        Output:
        - avg_train_loss (float): The average training loss for the epoch.
        """
        model_pl.train()  # Set the model to training mode
        total_train_loss = 0  # Initialize total training loss

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Select corresponding weights for the current batch
            batch_weights = full_weights[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].to(self.device)

            # Zero the gradients
            optimizer_theta.zero_grad()
            optimizer_P0.zero_grad()
            optimizer_gamma.zero_grad()
            
            loss = self.compute_loss_pl(model_pl, data, target, batch_weights)

            # Backward pass and gradient clipping
            loss.backward()
            # Gradient clipping
            max_norm_theta = max(1.0, 5.0 * (1 - epoch / self.max_epochs_pl))  # Dynamic for theta
            nn.utils.clip_grad_norm_([model_pl.theta], max_norm=max_norm_theta)

            optimizer_theta.step()
            optimizer_P0.step()
            optimizer_gamma.step()

            total_train_loss += loss.item() * len(data)

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / sum(len(data) for data, _ in train_loader)
        return avg_train_loss
    
    
    def train_test(self, train_dataset, test_loader, real_loc):        
        model_nn = copy.deepcopy(self.initial_model_nn)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                            
        # Move the model to the selected device (GPU/CPU)
        model_nn.to(self.device)

        # Initialize optimizers
        optimizer_nn = self.optimizer_nn(list(model_nn.parameters()))
        
        # Precompute weights for the entire training dataset
        indices = range(len(train_dataset))  # Get all indices from the train dataset
        data, target = train_dataset[indices]  # Retrieve full dataset and target values
        
        if self.weights_nn_type == 'max':
            weights_nn = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
            weights_nn = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
        elif self.weights_nn_type == 'max_quadratic':
            weights_nn = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
            weights_nn = weights_nn**2
            weights_nn = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
        elif self.weights_nn_type == 'min_max':
            weights_nn = (target - torch.mean(target)).abs() + 1e-6

        train_losses_nn_per_epoch = []

        for epoch_idx in tqdm(range(self.max_epochs_nn)):
            avg_train_loss_nn, _ = self.train_one_epoch_nn(model_nn, train_loader, optimizer_nn, epoch_idx, weights_nn)
            train_losses_nn_per_epoch.append(avg_train_loss_nn)
            
        for param in model_nn.parameters():
            print(param.grad.norm())
            
        # Get theta initialization
        theta_init = self.get_theta_init(model_nn, train_loader)
        
        print(f"Initial theta: {theta_init}")
        
        model_pl = copy.deepcopy(self.initial_model_pl)

        model_pl.theta = theta_init
        
        # Initialize optimizers
        optimizer_theta = self.optimizer_theta([model_pl.theta])
        optimizer_P0 = self.optimizer_P0([model_pl.P0]) 
        optimizer_gamma = self.optimizer_gamma([model_pl.gamma])
        
        # scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=5, gamma=0.5)
        
        if self.weights_pl_type == 'max':
            weights_pl = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
            weights_pl = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
        elif self.weights_pl_type == 'max_quadratic':
            weights_pl = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
            weights_pl = weights_nn**2
            weights_pl = weights_nn / weights_nn.sum()  # Normalize weights to sum to 1
        elif self.weights_pl_type == 'min_max':
            weights_pl = (target - torch.mean(target)).abs() + 1e-6
        
        train_losses_pl_per_epoch = []

        for epoch_idx in tqdm(range(self.max_epochs_pl)):
            avg_train_loss_pl = self.train_one_epoch_pl(model_pl, train_loader, optimizer_theta, optimizer_P0, optimizer_gamma, epoch_idx, weights_pl)
            train_losses_pl_per_epoch.append(avg_train_loss_pl)
            
             # Step the scheduler for theta
            # scheduler_theta.step()
        
        # Find the point in the train_dataset that is closest to the real_loc position
        closest_point = None
        min_distance = float('inf')

        for data, _ in train_loader:
            distances = torch.norm(data - torch.tensor(real_loc, device=self.device), dim=1)
            min_idx = torch.argmin(distances)
            if distances[min_idx] < min_distance:
                min_distance = distances[min_idx]
                closest_point = data[min_idx].cpu().numpy()

        print(f"Closest point in the train dataset to the real location: {closest_point}")
        print(f"Minimum distance to the real location: {min_distance}")
        
        # Evaluate on the global test set
        global_test_nn_loss = self.test_reg(model_nn, test_loader)
        global_test_pl_loss = self.test_reg(model_pl, test_loader)
        jam_loc_error, predicted_jam_loc = self.test_loc(model_pl, real_loc)
        learnt_P0, learnt_gamma = self.get_learnt_parameters(model_pl)
        
        
        return train_losses_nn_per_epoch, train_losses_pl_per_epoch, global_test_nn_loss, global_test_pl_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, model_nn, model_pl


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
                total_test_loss += self.compute_loss_pl(model, data, target, test_weights) * len(data)

        # Calculate average test loss
        avg_test_loss = total_test_loss / sum(len(data) for data, _ in test_loader)
        return avg_test_loss
    
    def test_loc(self, model, real_loc):
        """
        Computes the localization error based on the model's learned parameters.

        Parameters:
        - real_loc (ndarray): The true location for comparison.

        Output:
        - result (float): The computed localization error.
        """
        predicted_jam_loc = model.get_theta().detach().numpy()
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
