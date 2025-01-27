import numpy as np
import scipy as sp
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class data_process:
    """
    A class to handle data loading, preprocessing, and splitting for training and validation.

    Attributes:
    - path (str): The path to the data files.
    - time_t (int): Time index for time series data (default is 1).
    - test_ratio (float): Ratio of the data to be used as the test set (default is 0.3).
    - data_preprocessing (int): Flag to apply preprocessing to Y data (default is 1).
    - split_method (str): Method to split the data (default is 'random').
    - batch_size (int): Batch size for data loaders (default is 16).
    """

    def __init__(self, path, time_t, num_obs, test_ratio, data_preprocessing, noise, meas_noise_var, batch_size, split_method='random'):
        """
        Initializes the data_process class by loading and preprocessing data.

        Parameters:
        - path (str): Path to the data files.
        - time_t (int): Time index for time series data (default is 1).
        - test_ratio (float): Ratio for the train-test split (default is 0.3).
        - data_preprocessing (int): Flag to determine if preprocessing is applied (default is 1).
        - split_method (str): Method for splitting the dataset (default is 'random').
        - batch_size (int): Batch size for data loaders (default is 16).

        Outputs:
        Initializes the class with train/test data tensors, preprocessing, and splitting parameters.
        """
        self.path = path
        self.split_method = split_method
        self.batch_size = batch_size
        self.num_obs = num_obs

        # Load the true jammer location
        fname_Jloc = path + 'trueJamLoc.mat'
        data = io.loadmat(fname_Jloc)
        self.trueJloc = data['Jloc']

        # Load X data (input features)
        fname_x = path + 'X.mat'
        x_data = io.loadmat(fname_x)

        X_all_obs = x_data['XX']

        # Load Y data (labels)
        fname_y = path + 'Y.mat'
        y_data = io.loadmat(fname_y)
        Y_all_obs = y_data['YY']

        # Reduce the dataset to num_obs observations
        if len(X_all_obs) >= self.num_obs and len(Y_all_obs) >= self.num_obs:
            indices = np.random.choice(len(X_all_obs), self.num_obs, replace=False)
            X = X_all_obs[indices].copy()
            Y = Y_all_obs[indices].copy()
        else:
            X = X_all_obs.copy()
            Y = Y_all_obs.copy()
            print(f'Number of observations ({len(X)}) is less than the specified number of observations ({num_obs}). Using the full dataset.')
            
        # If data is time series, select the specified time index
        if len(X.shape) > 2:
            X = X[:, :, time_t]
        if len(Y.shape) > 1:
            Y = Y[:, time_t]
            self.trueJloc = self.trueJloc[time_t, :]

        # This block of code is responsible for applying preprocessing to the Y data if the
        # `data_preprocessing` flag is set to 1. Here's a breakdown of what it does:
        # Apply preprocessing if needed
        if data_preprocessing == 1:
            y_copy = Y.copy()
            Y[y_copy == -np.inf] = min(y_copy[y_copy != -np.inf]) - 10  # Replace -inf with minimum value minus 10
            # Y[y_copy < -100] = -100
        elif data_preprocessing == 2:
            # Find indices where Y contains -inf
            # valid_indices = ~(Y == -np.inf)
            valid_indices = ~(Y < -150.0)
            X = X[valid_indices,:]
            Y = Y[valid_indices]
            print('Number of valid samples: ', len(Y))
            
        if noise == 1:
            Y = Y + np.random.normal(0, np.sqrt(meas_noise_var), Y.shape)
            
        # Expand Y to 2D for compatibility
        Y = np.expand_dims(Y, axis=1)

        # Split data into training and testing sets
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_ratio, random_state=42, shuffle=True)

        # Convert data to PyTorch tensors
        self.train_x_original = torch.from_numpy(np.array(Xtrain)).float()
        self.train_y_original = torch.from_numpy(np.array(Ytrain)).float()
        self.test_x_original = torch.from_numpy(np.array(Xtest)).float()
        self.test_y_original = torch.from_numpy(np.array(Ytest)).float()
        print('Train X shape: ', self.train_x_original.shape)
        print('Test X shape: ', self.test_x_original.shape)

        # Find the maximum value in the training labels and its corresponding input
        self.train_y_max = torch.max(self.train_y_original)
        train_y_max_index = torch.argmax(self.train_y_original)
        self.train_x_max = self.train_x_original[train_y_max_index]

    def split_dataset(self, num_nodes):
        """
        Splits the dataset into training and validation sets using KFold cross-validation.

        Outputs:
        - indices_folds (list): List of dictionaries containing train and validation indices for each fold.
        - crossval_dataset (Dataset): PyTorch dataset for cross-validation.
        - test_loader (DataLoader): DataLoader for the test set.
        """
        train_y_max_splited = []  # To store max values of y for local nodes (if needed)
        train_splited = []  # To store split train data indices

        # Create a DataLoader for the test set
        test_dataset = CustomDataset(self.test_x_original, self.test_y_original)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        indices_folds = []

        # If using random splitting, apply KFold cross-validation
        if self.split_method == 'random':
            full_dataset = CustomDataset(self.train_x_original, self.train_y_original)
            dataset_size = len(full_dataset)
            part_sizes = [dataset_size // num_nodes for _ in range(num_nodes)]
            for i in range(dataset_size % num_nodes):  # Add 1 to some parts if the dataset_size is not divisible by 10
                part_sizes[i] += 1
            partitions = random_split(full_dataset, part_sizes)

            # Create DataLoader instances
            train_loader_splited = [DataLoader(partition, batch_size=self.batch_size, shuffle=True, drop_last=True) for partition in partitions]

            # Step 3: Get the max y value of each partition, and its corresponding x value  
            max_values_each_partition = []
            train_y_mean_splited = []          
            for i, partition in enumerate(partitions):
                # Since random_split returns Subset, we need to get the original indices
                max_index_in_partition = torch.argmax(partition.dataset.labels[partition.indices])
                original_index = partition.indices[max_index_in_partition]
                max_y = partition.dataset.labels[original_index]
                max_x = partition.dataset.features[original_index]
                max_values_each_partition.append(max_x)
                train_y_mean_splited.append(torch.mean(partition.dataset.labels))

        return train_loader_splited, test_loader, max_values_each_partition, train_y_mean_splited, self.train_x_max, self.train_y_max

# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle features and labels.

    Attributes:
    - features (Tensor): Input features for the dataset.
    - labels (Tensor): Corresponding labels for the dataset.
    """
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