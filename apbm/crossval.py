import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import copy

class CrossVal(object):
    def __init__(self, cv_model, batch_size, optimizer_nn, optimizer_theta, mu, max_epochs, patience, early_stopping, betas=None):
        """
        Initializes the CrossVal class for cross-validation with early stopping.

        Parameters:
        - cv_model (nn.Module): The model to be trained and evaluated.
        - batch_size (int): Batch size for data loaders.
        - optimizer_nn (function): Optimizer for the neural network parameters.
        - optimizer_theta (function): Optimizer for the theta parameters.
        - mu (float): Hyperparameter for regularization or optimization purposes.
        - max_epochs (int): Maximum number of epochs for training.
        - patience (int): Number of epochs to wait before early stopping if no improvement.
        - early_stopping (bool): Flag to enable or disable early stopping.
        - betas (list or None): List of beta values for weighted training, optional.
        
        Output:
        Initializes the class attributes and settings.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crossval_model = cv_model
        self.optimizer_nn = optimizer_nn
        self.optimizer_theta = optimizer_theta
        self.criterion = nn.MSELoss(reduction="mean")  # Loss function for training
        self.mu = mu
        self.max_epochs = max_epochs
        self.patience = patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.num_rounds = 1  # For non-FL cases, this is 1 (kept for compatibility)
        self.betas = [10 * (1 - (i + 1) / self.num_rounds) for i in range(self.num_rounds)] if betas else [0 for _ in range(self.num_rounds)]

    def train_one_epoch(self, train_loader, optimizer_nn, optimizer_theta):
        """
        Trains the model for one epoch.

        Parameters:
        - train_loader (DataLoader): DataLoader for the training set.
        - optimizer_nn (torch.optim.Optimizer): Optimizer for neural network parameters.
        - optimizer_theta (torch.optim.Optimizer): Optimizer for theta parameters.
        
        Output:
        - avg_train_loss (float): The average training loss for the epoch.
        """
        self.crossval_model.train()  # Set the model to training mode
        total_train_loss = 0  # Initialize total training loss

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the gradients
            optimizer_nn.zero_grad()
            optimizer_theta.zero_grad()

            # Forward pass
            output = self.crossval_model(data)
            loss = self.criterion(output, target)

            # Backward pass and gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(self.crossval_model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer_nn.step()
            optimizer_theta.step()

            total_train_loss += loss.item() * len(data)

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / sum(len(data) for data, _ in train_loader)
        return avg_train_loss

    def validate_one_epoch(self, val_loader):
        """
        Evaluates the model on the validation set for one epoch.

        Parameters:
        - val_loader (DataLoader): DataLoader for the validation set.
        
        Output:
        - avg_val_loss (float): The average validation loss for the epoch.
        """
        self.crossval_model.eval()  # Set the model to evaluation mode
        total_val_loss = 0  # Initialize total validation loss

        with torch.no_grad():  # Disable gradient computation for validation
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.crossval_model(data)
                loss = self.criterion(output, target)
                total_val_loss += loss.item() * len(data)

        # Calculate average validation loss for the epoch
        avg_val_loss = total_val_loss / sum(len(data) for data, _ in val_loader)
        return avg_val_loss

    def train(self, indices_folds, crossval_dataset, test_loader, real_loc):
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

        # Iterate over each fold for cross-validation
        for fold_idx, fold in enumerate(indices_folds):
            print(f"Fold {fold_idx}")
            print("-------")

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
            self.crossval_model.to(self.device)

            # Initialize optimizers for the current fold
            optimizer_nn = self.optimizer_nn(self.crossval_model.model_NN.parameters())
            optimizer_theta = self.optimizer_theta(self.crossval_model.model_PL.parameters())

            # Lists to track training and validation losses for each epoch
            train_losses_per_epoch = []
            val_losses_per_epoch = []

            # Early stopping variables
            best_val_loss = float('inf')  # Track the best validation loss
            epochs_no_improve = 0  # Track epochs with no improvement

            # Training loop for each epoch
            for epoch_idx in tqdm(range(self.max_epochs)):
                avg_train_loss = self.train_one_epoch(train_loader, optimizer_nn, optimizer_theta)
                train_losses_per_epoch.append(avg_train_loss)

                if val_loader is not None:
                    avg_val_loss = self.validate_one_epoch(val_loader)
                    val_losses_per_epoch.append(avg_val_loss)

                    # Check if early stopping should be triggered
                    if self.early_stopping:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss  # Update best validation loss
                            epochs_no_improve = 0  # Reset counter
                            best_epoch = epoch_idx + 1  # Record the best epoch
                            best_model_state = copy.deepcopy(self.crossval_model.state_dict())
                        else:
                            epochs_no_improve += 1  # Increment counter if no improvement
                            if epochs_no_improve >= self.patience:
                                print(f"Early stopping at epoch {epoch_idx + 1}")
                                break

            # Load the best model if early stopping was triggered
            if self.early_stopping and epochs_no_improve >= self.patience:
                self.crossval_model.load_state_dict(best_model_state)
                print(f"Best model was found at epoch {best_epoch}")

            # Append losses for this fold
            all_train_losses_per_fold.append(train_losses_per_epoch)
            all_val_losses_per_fold.append(val_losses_per_epoch)

        # Evaluate on the global test set
        global_test_loss = self.test_reg(self.crossval_model, test_loader)
        jam_loc_error = self.test_loc(real_loc)

        return all_train_losses_per_fold, all_val_losses_per_fold, global_test_loss, jam_loc_error

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
                output = model(data)
                total_test_loss += self.criterion(output, target).item() * len(data)

        # Calculate average test loss
        avg_test_loss = total_test_loss / sum(len(data) for data, _ in test_loader)
        return avg_test_loss
    
    def test_loc(self, real_loc):
        """
        Computes the localization error based on the model's learned parameters.

        Parameters:
        - real_loc (ndarray): The true location for comparison.

        Output:
        - result (float): The computed localization error.
        """
        result = np.sqrt(np.mean((real_loc - self.crossval_model.get_theta().detach().numpy())**2))
        return result

    def test_field(self, model, t):
        """
        Generates and saves a 3D field visualization of the model's output.

        Parameters:
        - model (nn.Module): The trained model to be visualized.
        - t (int or str): Identifier for the output file (e.g., an index or timestamp).

        Output:
        - Saves an HTML file containing a 3D plot of the model's output surface.
        """
        model.eval()  # Set the model to evaluation mode
        x = np.linspace(0, 100, 100)  # Define a grid range for x-axis
        y = np.linspace(0, 100, 100)  # Define a grid range for y-axis
        X, Y = np.meshgrid(x, y)  # Create a mesh grid for plotting
        Z = np.zeros(X.shape)  # Initialize Z (output values) with zeros

        # Disable gradient computation for visualization
        with torch.no_grad():
            for i in range(len(X)):
                for j in range(len(Y)):
                    # Predict the model's output for each (x, y) point
                    Z[i, j] = model(torch.tensor([[X[i, j], Y[i, j]]]).float()).item()

        # Create a 3D plot using Plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_layout(
            title='3D Surface Plot',
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        # Save the 3D plot as an HTML file
        fig.write_html(f'figs/field_{t}.html')