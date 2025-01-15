import numpy as np
import scipy as sp
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

# parameters control the data
## data itself: path, noise, point_section

## data split: test_ratio, client_num, division_method(iid, non-iid)


class data_process:
    def __init__(self, path, time_t, test_ratio=0.3, data_preprocesing = 1, noise=True, noise_std = 3, data_selection=False, point_section = 200, split_method='random', local_bs=16, bins_num=10):
        self.path = path
        self.noise = noise
        self.point_section = point_section
        self.split_method = split_method
        self.local_bs = local_bs
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
            

        # data selection
        if data_selection:
            # select the high power points
            point_section_index = np.argsort(Y, axis=0)[-point_section:]
            X = X[point_section_index]
            Y = Y[point_section_index]

        # Add noise to power measurments
        if noise:
            Y = Y + np.random.normal(0, noise_std, Y.shape)

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

    def split_dataset(self, num_nodes):
        # train_len = len(self.train_data)
        # test_len = len(self.test_data)
        # the local nodes max value of y
        train_y_max_splited = []
        train_splited = []
        # test_splited = []
        # make a global test loader
        test_loader_global = DataLoader(CustomDataset(self.test_x_original, self.test_y_original), batch_size=self.local_bs, shuffle=True)
        max_values_each_partition = []
        train_y_mean_splited = []
        if self.split_method == 'random':
            full_dataset = CustomDataset(self.train_x_original, self.train_y_original)
            # Divide the dataset into 10 approximately equal parts
            dataset_size = len(full_dataset)
            part_sizes = [dataset_size // num_nodes for _ in range(num_nodes)]
            for i in range(dataset_size % num_nodes):  # Add 1 to some parts if the dataset_size is not divisible by 10
                part_sizes[i] += 1
            partitions = random_split(full_dataset, part_sizes)

            # Create DataLoader instances
            train_splited = [DataLoader(partition, batch_size=self.local_bs, shuffle=True) for partition in partitions]

            # partitions = divide_into_parts(self.train_x_original, self.train_y_original)
            # # Create DataLoader instances
            # train_splited = [DataLoader(CustomDataset(part_X, part_Y), batch_size=32, shuffle=True) for part_X, part_Y in partitions]

            # Step 3: Get the max y value of each partition, and its corresponding x value
            # Step 3: Get the max y value of each partition, and its corresponding x value
            
            for i, partition in enumerate(partitions):
                # Since random_split returns Subset, we need to get the original indices
                max_index_in_partition = torch.argmax(partition.dataset.labels[partition.indices])
                original_index = partition.indices[max_index_in_partition]
                max_y = partition.dataset.labels[original_index]
                max_x = partition.dataset.features[original_index]
                max_values_each_partition.append(max_x)
                train_y_mean_splited.append(torch.mean(partition.dataset.labels))

        elif self.split_method == 'area_grid':
            # Assume `partitioned_data` is the output from the previous `partition_space` function
            num_partitions_x1, num_partitions_x2 = num_nodes[0], num_nodes[1]
            partitioned_data = partition_space(self.train_x_original, self.train_y_original, num_partitions_x1, num_partitions_x2)

            # Parameters for DataLoader
            batch_size = self.local_bs
            shuffle = True

            for partition_idx, data in partitioned_data.items():
                # Create the Dataset from the partitioned data
                dataset = CustomDataset(data['x'], data['y'])
                
                # Create the DataLoader for the current partition
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
                
                # Store the DataLoader in the dictionary with the partition's index as the key
                train_splited.append(dataloader)
                # get the max values of each partition
                train_y_max_splited.append(torch.max(data['y']))
                max_index_in_partition = torch.argmax(data['y'])
                max_x = data['x'][max_index_in_partition]
                max_values_each_partition.append(max_x)
                # get the mean value of each partition
                train_y_mean_splited.append(torch.mean(data['y']))

        return train_splited, test_loader_global, max_values_each_partition, train_y_mean_splited,  self.bins, self.train_x_max, self.train_y_max


# Function to divide the dataset into 10 parts
# Function to divide the dataset into 10 parts
def divide_into_parts(X, Y, num_parts=10, shuffle=True):
    if shuffle:
        # Shuffle the indices
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    part_size = len(X) // num_parts
    remainder = len(X) % num_parts
    partitions = []
    for i in range(num_parts):
        part_indices = indices[i * part_size: (i + 1) * part_size] if i < num_parts - 1 else indices[i * part_size:]
        partitions.append((X[part_indices], Y[part_indices]))
    return partitions

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
import torch

def partition_space(x, y, num_partitions_x1, num_partitions_x2):
    # Determine the bounds of the area
    min_x1, min_x2 = torch.min(x, dim=0).values
    max_x1, max_x2 = torch.max(x, dim=0).values

    # Calculate the size of each partition
    partition_size_x1 = (max_x1 - min_x1) / num_partitions_x1
    partition_size_x2 = (max_x2 - min_x2) / num_partitions_x2

    # Function to determine the partition index for a given coordinate
    def get_partition_index(coord, min_coord, partition_size, num_partitions):
        # Calculate index and use clamp to keep it within valid range
        index = ((coord - min_coord) / partition_size).long()
        return index.clamp(0, num_partitions - 1)

    # Assign each point to a partition
    partition_indices = torch.zeros(x.size(0), 2, dtype=torch.long)
    for i, (x1, x2) in enumerate(x):
        partition_indices[i, 0] = get_partition_index(x1, min_x1, partition_size_x1, num_partitions_x1)
        partition_indices[i, 1] = get_partition_index(x2, min_x2, partition_size_x2, num_partitions_x2)

    # Create a dictionary to hold the partitioned data
    partitioned_data = {}
    for i, indices in enumerate(partition_indices):
        # Tuple index for the partition
        partition_idx = tuple(indices.tolist())
        if partition_idx not in partitioned_data:
            partitioned_data[partition_idx] = {'x': [], 'y': []}
        partitioned_data[partition_idx]['x'].append(x[i])
        partitioned_data[partition_idx]['y'].append(y[i])

    # Convert lists to tensors
    for partition_idx in partitioned_data:
        partitioned_data[partition_idx]['x'] = torch.stack(partitioned_data[partition_idx]['x'])
        partitioned_data[partition_idx]['y'] = torch.tensor(partitioned_data[partition_idx]['y'])

    return partitioned_data
