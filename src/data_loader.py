import os
import numpy as np
import scipy as sp
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class data_process:
    """
    A class to handle data loading, preprocessing, and splitting for training and validation.
    """

    def __init__(self, path, num_obs, test_ratio, data_preprocessing, noise, meas_noise_var, batch_size, split_method='random', time_t=0):
        """
        Parameters:
        - path (str): Path to the dataset files.
        - num_obs (int): Number of observations to consider in the dataset.
        - test_ratio (float): Proportion of the dataset allocated for testing.
        - data_preprocessing (int): Preprocessing method to apply:
            - 0: No preprocessing.
            - 1: Replace `-inf` values in Y with the minimum finite value in Y minus 10.
            - 2: Remove samples where `Y < -150.0`.
        - noise (bool): Whether to add noise to the dataset.
        - meas_noise_var (float): Variance of the measurement noise if noise is applied.
        - batch_size (int): Number of samples per batch for data loading.
        - split_method (str): Strategy for splitting the dataset ('random' by default).
        - time_t (int): Time index for time-series data (default is 0).

        Outputs:
        Initializes the class with train/test data tensors, preprocessing, and splitting parameters.
        """
        self.path = path
        self.split_method = split_method
        self.batch_size = batch_size
        self.num_obs = num_obs

        # Ensure correct path construction
        fname_Jloc = os.path.join(path, 'trueJamLoc.mat')
        fname_x = os.path.join(path, 'X.mat')
        fname_y = os.path.join(path, 'Y.mat')

        # Check if files exist
        for fname in [fname_Jloc, fname_x, fname_y]:
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Error: The file {fname} does not exist. Verify the dataset path.")

        # Load true jammer location
        data = io.loadmat(fname_Jloc)
        self.trueJloc = data['Jloc']

        # Load X and Y data
        x_data = io.loadmat(fname_x)
        y_data = io.loadmat(fname_y)

        X_all_obs = x_data['XX']
        Y_all_obs = y_data['YY']

        # Ensure enough observations
        if len(X_all_obs) >= self.num_obs and len(Y_all_obs) >= self.num_obs:
            indices = np.random.choice(len(X_all_obs), self.num_obs, replace=False)
            X = X_all_obs[indices].copy()
            Y = Y_all_obs[indices].copy()
        else:
            raise ValueError(f"Insufficient observations: Expected {num_obs}, but only {len(X_all_obs)} available.")

        # Time-series processing
        if len(X.shape) > 2:
            X = X[:, :, time_t]
        if len(Y.shape) > 1:
            Y = Y[:, time_t]
            self.trueJloc = self.trueJloc[time_t, :]

        # Apply preprocessing
        if data_preprocessing == 1:
            y_copy = Y.copy()
            Y[y_copy == -np.inf] = min(y_copy[y_copy != -np.inf]) - 10
        elif data_preprocessing == 2:
            valid_indices = ~(Y < -150.0)
            X = X[valid_indices, :]
            Y = Y[valid_indices]

        if noise:
            Y += np.random.normal(0, np.sqrt(meas_noise_var), Y.shape)

        # Convert Y to 2D
        Y = np.expand_dims(Y, axis=1)

        # Split into train/test
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_ratio, random_state=42, shuffle=True)

        # Convert to PyTorch tensors
        self.train_x_original = torch.from_numpy(Xtrain).float()
        self.train_y_original = torch.from_numpy(Ytrain).float()
        self.test_x_original = torch.from_numpy(Xtest).float()
        self.test_y_original = torch.from_numpy(Ytest).float()

        print('Train X shape: ', self.train_x_original.shape)
        print('Test X shape: ', self.test_x_original.shape)

    def split_dataset(self, num_nodes):
        """
        Splits the dataset into multiple training subsets and creates a test DataLoader.

        Parameters:
        - num_nodes (int): Number of partitions to split the training dataset into.

        Returns:
        - train_loader_splited (list of DataLoader): List of DataLoaders containing training data for each partition.
        - test_loader (DataLoader): DataLoader for the test set.
        """

        # Create a DataLoader for the test set
        test_dataset = CustomDataset(self.test_x_original, self.test_y_original)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        if self.split_method == 'random':
            # Perform random splitting of the training dataset into `num_nodes` partitions
            full_dataset = CustomDataset(self.train_x_original, self.train_y_original)
            dataset_size = len(full_dataset)
            # Determine the sizes of each partition
            part_sizes = [dataset_size // num_nodes for _ in range(num_nodes)]
            for i in range(dataset_size % num_nodes):  # Distribute remaining samples if dataset size is not evenly divisible
                part_sizes[i] += 1
            partitions = random_split(full_dataset, part_sizes)

            # Create a DataLoader for each partition
            train_loader_splited = [DataLoader(partition, batch_size=self.batch_size, shuffle=True, drop_last=True) for partition in partitions]

        return train_loader_splited, test_loader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initializes the CustomDataset class.

        Parameters:
        - features (Tensor): The input features of the dataset.
        - labels (Tensor): The labels for the dataset.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Output:
        - (int): The length of the dataset.
        """
        return len(self.features)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Parameters:
        - index (int): The index of the sample to retrieve.

        Output:
        - (tuple): A tuple containing the feature and label at the specified index.
        """
        return self.features[index], self.labels[index]