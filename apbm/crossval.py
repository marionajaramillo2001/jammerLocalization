import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import copy
from functools import partial
import torch.optim as optim
from apbm.model import Net_augmented


class CrossVal(object):
    def __init__(self, model, theta_init, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, batch_size, mu, max_epochs, patience, early_stopping, betas=None):
        """
        Initializes the CrossVal class for cross-validation with early stopping.

        Parameters:
        - config (dict): Dictionary containing the configuration parameters.
        - batch_size (int): Batch size for data loaders.
        - optimizer_nn (function): Optimizer for the neural network parameters.
        - optimizer_theta (function): Optimizer for the theta parameter.
        - optimizer_P0 (function): Optimizer for the P0 parameter.
        - optimizer_gamma (function): Optimizer for the gamma parameter.
        - mu (float): Hyperparameter for regularization or optimization purposes.
        - max_epochs (int): Maximum number of epochs for training.
        - patience (int): Number of epochs to wait before early stopping if no improvement.
        - early_stopping (bool): Flag to enable or disable early stopping.
        - betas (list or None): List of beta values for weighted training, optional.
        
        Output:
        Initializes the class attributes and settings.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initial_model = model
        self.theta_init = theta_init
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.optimizer_P0 = optimizer_P0
        self.optimizer_gamma = optimizer_gamma
        self.theta_tuning_epochs = 0  # Number of epochs for tuning theta only
        self.theta_nn_separate = False  # Flag to separate theta and NN updates
        self.mu = mu
        self.max_epochs = max_epochs
        self.patience = patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.num_rounds = 1  # For non-FL cases, this is 1 (kept for compatibility)
        self.betas = [10 * (1 - (i + 1) / self.num_rounds) for i in range(self.num_rounds)] if betas else [0 for _ in range(self.num_rounds)]
        self.theta_init_fix = [200.0, 200.0]
        self.num_groups = 10  # Number of groups for mean_max_n_random initialization
        
    def get_theta_init(self, dataset):
        data_loader_theta_init = DataLoader(dataset, batch_size=len(dataset))
        # Identify the position with the highest y (received signal strength) for 'max_loc' initialization
        if self.theta_init == 'max_loc':
            for data, target in data_loader_theta_init:
                max_index = torch.argmax(target)  # Find the index of the max signal strength
                max_position = data[max_index]
                break
            return nn.Parameter(max_position + torch.randn(2))
        elif self.theta_init == 'random':
            # Random initialization of theta0 within a range
            return nn.Parameter(99 * torch.rand(2))
        elif self.theta_init == 'fix':
            # Fixed theta0 initialization at a specific point
            return nn.Parameter(torch.tensor(self.theta_init_fix))
        elif self.theta_init == 'mean_n_random':
            # Mean initialization of theta0 from n random positions
            random_positions = []
            for data, target in data_loader_theta_init:
                indices = torch.randperm(len(data))[:10]  # Get 10 random indices
                random_positions = data[indices] 
                break 
            mean_random_position = random_positions.mean(dim=0)
            return nn.Parameter(mean_random_position + torch.randn(2))
        elif self.theta_init == 'mean_max_n_random':
            num_groups = self.num_groups
            group_size = len(dataset) // num_groups
            random_indices = torch.randperm(len(dataset))
            max_positions = []
            for i in range(num_groups):
                if i == num_groups - 1:
                    group_indices = random_indices[i * group_size:]
                else:
                    group_indices = random_indices[i * group_size: (i + 1) * group_size]

                group_data_loader = DataLoader(dataset, batch_size=group_size, sampler=torch.utils.data.SubsetRandomSampler(group_indices))
                
                for data, target in group_data_loader:
                    max_index = torch.argmax(target)  # Find the index of the max signal strength in the group
                    max_position = data[max_index]
                    max_positions.append(max_position)
                    break
                            
            # Average the positions with the highest target values from each group
            mean_max_position = torch.stack(max_positions).mean(dim=0)
            return nn.Parameter(mean_max_position + torch.randn(2))
        
    def compute_loss(self, model, data, target):
            # Forward pass
            output = model(data)
            output_pl = model.model_PL(data)
            
            error = torch.norm(output - target, dim=1)
            error_pl = torch.norm(output_pl - target, dim=1)
            
            lambda_mse_loss_total = 1.0
            lambda_mse_loss_pl = 2.0
            
            # Use RSS (target values) to compute weights and normalize target values to the range [0, 1]
            normalized_target = (target - torch.min(target)) / (torch.max(target) - torch.min(target) + 1e-6)
            weights = normalized_target # Assign higher weight to higher target values (higher RSS)
            weights = weights / weights.sum() # Normalize weights to sum to 1
            
            # inverse_weights = 1/(normalized_target**2 + 1e-6)  # Inverse weights
            # inverse_weights = inverse_weights / inverse_weights.sum()  # Normalize inverse weights to sum to 1
            
            # distance_weights = torch.norm(data - model.model_PL.theta, dim=1)
            # distance_weights = distance_weights / distance_weights.sum()  # Normalize distance weights to sum to 1
            
            # distance_and_inverse_weights = distance_weights * inverse_weights
            # distance_and_inverse_weights = distance_and_inverse_weights / distance_and_inverse_weights.sum()  # Normalize distance and inverse weights to sum to 1
            
            mse_loss = torch.mean(error)
            mse_loss_pl = torch.mean(error_pl * weights)
            
            # Add L2 regularization for NN parameters
            l2_lambda = 0.01  # Regularization strength
            l2_loss = 0
            for param in model.model_NN.parameters():
                l2_loss += torch.norm(param, 2)  # Compute L2 norm for all NN parameters

            loss = lambda_mse_loss_total*mse_loss + lambda_mse_loss_pl * mse_loss_pl + l2_lambda * l2_loss
            
            return loss

    def train_one_epoch(self, model, train_loader, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, epoch):
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
        model.train()  # Set the model to training mode
        total_train_loss = 0  # Initialize total training loss

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the gradients
            
            optimizer_theta.zero_grad()
            optimizer_nn.zero_grad()
            optimizer_P0.zero_grad()
            optimizer_gamma.zero_grad()
            
            loss = self.compute_loss(model, data, target)
        

            # Backward pass and gradient clipping
            loss.backward()
            # Gradient clipping
            max_norm_theta = max(1.0, 5.0 * (1 - epoch / self.max_epochs))  # Dynamic for theta
            nn.utils.clip_grad_norm_([model.model_PL.theta], max_norm=max_norm_theta)

            remaining_params = [p for p in model.parameters() if p is not model.model_PL.theta]
            nn.utils.clip_grad_norm_(remaining_params, max_norm=1.0)

            # Update parameters
            if (epoch <= self.theta_tuning_epochs) or (epoch > self.theta_tuning_epochs and not self.theta_nn_separate):
                optimizer_theta.step()  # Always update theta
            if epoch > self.theta_tuning_epochs:  # Update NN only after theta tuning phase
                optimizer_nn.step()
            optimizer_P0.step()
            optimizer_gamma.step()

            total_train_loss += loss.item() * len(data)

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / sum(len(data) for data, _ in train_loader)
        return avg_train_loss
    
    def validate_one_epoch(self, model, val_loader):
            """
            Evaluates the model on the validation set for one epoch.

            Parameters:
            - val_loader (DataLoader): DataLoader for the validation set.
            
            Output:
            - avg_val_loss (float): The average validation loss for the epoch.
            """
            model.eval()  # Set the model to evaluation mode
            total_val_loss = 0  # Initialize total validation loss

            with torch.no_grad():  # Disable gradient computation for validation
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    loss = self.compute_loss(model, data, target)
                    total_val_loss += loss.item() * len(data)

            # Calculate average validation loss for the epoch
            avg_val_loss = total_val_loss / sum(len(data) for data, _ in val_loader)
            return avg_val_loss

    def train_crossval(self, indices_folds, crossval_dataset):
        """
        Trains and validates the model across multiple cross-validation folds.

        Parameters:
        - indices_folds (list): List of dictionaries containing train and validation indices for each fold.
        - crossval_dataset (Dataset): The dataset for cross-validation.
        - test_loader (DataLoader): DataLoader for the test set.
        - real_loc (ndarray): The actual location data for evaluation.

        Output:
        - all_train_losses_per_fold (list): Training losses for each fold.
        - all_val_losses_per_fold (list): Validation losses for each fold.
        - global_test_loss (float): Average test loss after all folds.
        - jam_loc_error (float): Localization error of the model.
        """
        all_train_losses_per_fold = []  # Store training losses per fold
        all_val_losses_per_fold = []  # Store validation losses per fold
        best_epochs_per_fold = []  # Store the best epoch for each fold
        
        
        # Iterate over each fold for cross-validation
        for fold_idx, fold in enumerate(indices_folds):
            print(f"Fold {fold_idx}")
            print("-------")
            
            model = copy.deepcopy(self.initial_model)
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = self.max_epochs
            
            model.model_PL.theta = self.get_theta_init(crossval_dataset)
            print(f"Initial theta for cross-validation fold {fold_idx}: {model.model_PL.theta}")
            
            # Create data loaders for the current fold
            train_loader = DataLoader(
                dataset=crossval_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(fold["train_index"]),
            )
            val_loader = DataLoader(
                dataset=crossval_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(fold["val_index"]),
            )

            # Move the model to the selected device (GPU/CPU)
            model.to(self.device)

            # Initialize optimizers for the current fold
            optimizer_nn = self.optimizer_nn(list(model.model_NN.parameters()) + [model.w])
            optimizer_theta = self.optimizer_theta([model.model_PL.theta])        
            optimizer_P0 = self.optimizer_P0([model.model_PL.P0]) 
            optimizer_gamma = self.optimizer_gamma([model.model_PL.gamma])
            
            # Define scheduler for theta
            scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=25, gamma=0.5)  # Halve LR every 10 epochs
            # scheduler_theta = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_theta, mode='min', factor=0.5, patience=5)
            
            # Lists to track training and validation losses for each epoch
            train_losses_per_epoch = []
            val_losses_per_epoch = []

            # Early stopping variables
            best_val_loss = float('inf')  # Track the best validation loss
            epochs_no_improve = 0  # Track epochs with no improvement

            # Training loop for each epoch
            for epoch_idx in tqdm(range(self.max_epochs)):
                avg_train_loss = self.train_one_epoch(model, train_loader, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, epoch_idx)
                train_losses_per_epoch.append(avg_train_loss)

                avg_val_loss = self.validate_one_epoch(model, val_loader)
                val_losses_per_epoch.append(avg_val_loss)

                # Step the scheduler for theta
                scheduler_theta.step()

                # Check if early stopping should be triggered
                if self.early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss  # Update best validation loss
                        epochs_no_improve = 0  # Reset counter
                        best_epoch = epoch_idx + 1  # Record the best epoch
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        epochs_no_improve += 1  # Increment counter if no improvement
                        if epochs_no_improve >= self.patience:
                            print(f"Early stopping at epoch {epoch_idx + 1}")
                            break

            # Load the best model if early stopping was triggered
            if self.early_stopping and (epochs_no_improve >= self.patience or epoch_idx == (self.max_epochs - 1)):
                model.load_state_dict(best_model_state)
                best_epochs_per_fold.append(best_epoch)
                print(f"Best model was found at epoch {best_epoch}")
            elif not self.early_stopping and epoch_idx == (self.max_epochs - 1):
                model.load_state_dict(best_model_state)

            # Append losses for this fold
            all_train_losses_per_fold.append(train_losses_per_epoch)
            all_val_losses_per_fold.append(val_losses_per_epoch)
            
        # Calculate the mean of the best epochs across all folds
        if self.early_stopping:
            mean_best_epoch = int(np.mean(best_epochs_per_fold))
        else:
            mean_best_epoch = self.max_epochs


        # Calculate the mean of the last validation losses across all folds
        last_val_losses = [val_losses[-1] for val_losses in all_val_losses_per_fold]
        last_val_loss_mean_across_folds = np.mean(last_val_losses)

        return all_train_losses_per_fold, all_val_losses_per_fold, last_val_loss_mean_across_folds, mean_best_epoch
    
    def train_test(self, train_dataset, test_loader, real_loc):        
        model = copy.deepcopy(self.initial_model)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        model.model_PL.theta = self.get_theta_init(train_dataset)
        print(f"Initial theta for complete training dataset: {model.model_PL.theta}")
                    
        # Move the model to the selected device (GPU/CPU)
        model.to(self.device)

        # Initialize optimizers for the current fold
        optimizer_nn = self.optimizer_nn(list(model.model_NN.parameters()) + [model.w])
        optimizer_theta = self.optimizer_theta([model.model_PL.theta])        
        optimizer_P0 = self.optimizer_P0([model.model_PL.P0]) 
        optimizer_gamma = self.optimizer_gamma([model.model_PL.gamma])
        
        # Define scheduler for theta
        scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=25, gamma=0.5)  # Halve LR every 10 epochs

        train_losses_per_epoch = []

        # Training loop for each epoch; in this case, max_epochs usually is the mean_best_epoch found during crossvalidation
        for epoch_idx in tqdm(range(self.max_epochs)):
            avg_train_loss = self.train_one_epoch(model, train_loader, optimizer_nn, optimizer_theta, optimizer_P0, optimizer_gamma, epoch_idx)
            train_losses_per_epoch.append(avg_train_loss)
            
            scheduler_theta.step()
        
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
        global_test_loss = self.test_reg(model, test_loader)
        jam_loc_error, predicted_jam_loc = self.test_loc(model, real_loc)
        learnt_P0, learnt_gamma = self.get_learnt_parameters(model)
        
        w_PL, w_NN = model.get_w_weights()
        print(f"w_PL: {w_PL}, w_NN: {w_NN}")
        
        return train_losses_per_epoch, global_test_loss, jam_loc_error, predicted_jam_loc, learnt_P0, learnt_gamma, model


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

        with torch.no_grad():  # Disable gradient computation for testing
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                total_test_loss += self.compute_loss(model, data, target) * len(data)

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
        P0 = model.model_PL.P0.item()
        gamma = model.model_PL.gamma.item()
        
        return P0, gamma
