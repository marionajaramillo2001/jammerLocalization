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
    def __init__(self, path, time_t = 1, test_ratio=0.3, data_preprocesing = 1, split_method='random', batch_size=16, bins_num=10):
        self.path = path
        self.split_method = split_method
        self.batch_size = batch_size
        # load data
        # load true position
        fname_Jloc = path+'trueJamLoc.mat'
        data = io.loadmat(fname_Jloc)
        self.trueJloc = data['Jloc']
        # load matlab data
        fname_x = path+'X.mat'
        x_data = io.loadmat(fname_x)
        X = x_data['XX']
        fname_y = path+'Y.mat'
        y_data = io.loadmat(fname_y)
        Y = y_data['YY']
            
        # if data is time series
        if len(X.shape)>2:
            X = X[:,:,time_t]
            Y = Y[:,time_t]
            # extend Y to 2D
            # Y = np.expand_dims(Y, axis=1)
            self.trueJloc=self.trueJloc[time_t,:]
            
        if data_preprocesing==1:
            # preprocess the y data, change the y to minumum value -10 of min(y) if y is -inf
            y_copy = Y.copy()
            Y[y_copy == -np.inf] = min(y_copy[y_copy != -np.inf]) - 10

        # make Y to 2D
        Y = np.expand_dims(Y, axis=1)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_ratio, random_state=42, shuffle=True)
        
        self.train_x_original = torch.from_numpy(np.array(Xtrain)).float()
        self.train_y_original = torch.from_numpy(np.array(Ytrain)).float()
        self.test_x_original = torch.from_numpy(np.array(Xtest)).float()
        self.test_y_original = torch.from_numpy(np.array(Ytest)).float()

        # get the max value of train_y_original and its location
        self.train_y_max = torch.max(self.train_y_original)
        train_y_max_index = torch.argmax(self.train_y_original)
        self.train_x_max = self.train_x_original[train_y_max_index]
        self.bins = np.linspace(np.min(Ytrain), np.max(Ytrain), bins_num)
        # self.train_data = torch.utils.data.TensorDataset(self.train_x_original, self.train_y_original)
        # self.test_data = torch.utils.data.TensorDataset(self.test_x_original, self.test_y_original)

    def split_dataset(self):
        # train_len = len(self.train_data)
        # test_len = len(self.test_data)
        # the local nodes max value of y
        train_y_max_splited = []
        train_splited = []
        # test_splited = []
        # make a global test loader
        test_dataset = CustomDataset(self.test_x_original, self.test_y_original)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        indices_folds = []
        if self.split_method == 'random':
            crossval_dataset = CustomDataset(self.train_x_original, self.train_y_original)
            
            kf_train_val = KFold(n_splits=10, shuffle=True, random_state=22)
            # Iterate over the splits
            for j, (train_index, val_index) in enumerate(kf_train_val.split(self.train_x_original, self.train_y_original)):
                # Create a dictionary to store train_index and val_index
                split_data = {'train_index': train_index, 'val_index': val_index}
                # Append the dictionary to the list
                indices_folds.append(split_data)
                
        return indices_folds, crossval_dataset, test_loader

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    