import numpy as np
import scipy as sp
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import Dataset, DataLoader
import random

class data_process:
    """
    A class to handle data loading, preprocessing, and splitting for training and validation.

    Attributes:
    - path (str): The path to the data files.
    - time_t (int): Time index for time series data (default is 1).
    - test_ratio (float): Ratio of the data to be used as the test set (default is 0.3).
    - data_preprocesing (int): Flag to apply preprocessing to Y data (default is 1).
    - split_method (str): Method to split the data (default is 'random').
    - batch_size (int): Batch size for data loaders (default is 16).
    - bins_num (int): Number of bins for data binning (default is 10).
    """

    def __init__(self, path, time_t=1, test_ratio=0.3, data_preprocesing=1, split_method='random', batch_size=16, bins_num=10):
        """
        Initializes the data_process class by loading and preprocessing data.

        Parameters:
        - path (str): Path to the data files.
        - time_t (int): Time index for time series data (default is 1).
        - test_ratio (float): Ratio for the train-test split (default is 0.3).
        - data_preprocesing (int): Flag to determine if preprocessing is applied (default is 1).
        - split_method (str): Method for splitting the dataset (default is 'random').
        - batch_size (int): Batch size for data loaders (default is 16).
        - bins_num (int): Number of bins for data binning (default is 10).

        Outputs:
        Initializes the class with train/test data tensors, preprocessing, and splitting parameters.
        """
        self.path = path
        self.split_method = split_method
        self.batch_size = batch_size

        # Load the true jammer location
        fname_Jloc = path + 'trueJamLoc.mat'
        data = io.loadmat(fname_Jloc)
        self.trueJloc = data['Jloc']

        # Load X data (input features)
        fname_x = path + 'X.mat'
        x_data = io.loadmat(fname_x)
        X = x_data['XX']

        # Load Y data (labels)
        fname_y = path + 'Y.mat'
        y_data = io.loadmat(fname_y)
        Y = y_data['YY']

        # If data is time series, select the specified time index
        if len(X.shape) > 2:
            X = X[:, :, time_t]
            Y = Y[:, time_t]
            self.trueJloc = self.trueJloc[time_t, :]

        # Apply preprocessing if needed
        if data_preprocesing == 1:
            y_copy = Y.copy()
            Y[y_copy == -np.inf] = min(y_copy[y_copy != -np.inf]) - 10  # Replace -inf with minimum value minus 10

        # Expand Y to 2D for compatibility
        Y = np.expand_dims(Y, axis=1)

        # Split data into training and testing sets
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_ratio, random_state=42, shuffle=True)

        # Convert data to PyTorch tensors
        self.train_x_original = torch.from_numpy(np.array(Xtrain)).float()
        self.train_y_original = torch.from_numpy(np.array(Ytrain)).float()
        self.test_x_original = torch.from_numpy(np.array(Xtest)).float()
        self.test_y_original = torch.from_numpy(np.array(Ytest)).float()

        # Find the maximum value in the training labels and its corresponding input
        self.train_y_max = torch.max(self.train_y_original)
        train_y_max_index = torch.argmax(self.train_y_original)
        self.train_x_max = self.train_x_original[train_y_max_index]

        # Create bins for the training data
        self.bins = np.linspace(np.min(Ytrain), np.max(Ytrain), bins_num)

    def split_dataset(self):
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
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        indices_folds = []

        # If using random splitting, apply KFold cross-validation
        if self.split_method == 'random':
            crossval_dataset = CustomDataset(self.train_x_original, self.train_y_original)
            kf_train_val = KFold(n_splits=5, shuffle=True, random_state=22)

            # Create train and validation indices for each fold
            for j, (train_index, val_index) in enumerate(kf_train_val.split(self.train_x_original, self.train_y_original)):
                split_data = {'train_index': train_index, 'val_index': val_index}
                indices_folds.append(split_data)

        return indices_folds, crossval_dataset, test_loader

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