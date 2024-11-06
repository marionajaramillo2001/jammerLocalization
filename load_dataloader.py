# load_dataloader
import torch
from torch.utils.data.sampler import SubsetRandomSampler

us = 0.1
def Load_Dataloader(args):
    
    train_loader = args.train_data
    test_loader = args.test_data
    num_clients = args.num_clients
    local_bs = args.local_bs
    # turn the dataset into Tensordataset
    global_test_loader = torch.utils.data.DataLoader(
        args.test_data,
        batch_size=args.local_bs,
        shuffle=True
    )
    # open the Tensordataset
    train_len = len(train_loader)

    # split the train dataset into several clients
    train_loaders = []
    test_loaders = []
    # method 1: split the dataset randomly
    if args.split == 'random':
        for i in range(num_clients):
            train_loaders.append(
                torch.utils.data.DataLoader(
                    train_loader,
                    batch_size=local_bs,
                    sampler=SubsetRandomSampler(
                        range(i * train_len // num_clients,
                            (i + 1) * train_len // num_clients)
                    )
                )
            )
            test_loaders.append(
                torch.utils.data.DataLoader(
                    test_loader,
                    batch_size=local_bs,
                    sampler=SubsetRandomSampler(
                        range(i * len(test_loader) // num_clients,
                                (i + 1) * len(test_loader) // num_clients)
                    )
                )
            )
        y_power_ratios = torch.ones(num_clients)
    # method 2: split the dataset by label
    elif args.split == 'area_grid':
        # get the x,y from traiin_loader
        x = train_loader[:][0]
        y = train_loader[:][1]
        # max value of each column in x
        x_max = torch.max(x, 0)[0]
        # min value of each column in x
        x_min = torch.min(x, 0)[0]
        # get the range of each column in x
        x_range = x_max - x_min
        x_grid_0= x_range[0]/args.grid_size[0]
        x_grid_1= x_range[1]/args.grid_size[1]
        # divide the dataset into several clients by x_grid
        data_points_in_clients = []
        for i in range(args.grid_size[0]):
            for j in range(args.grid_size[1]):
                data_points_in_client = []
                for k in range(len(x)):
                    if x[k][0] >= x_min[0] + i * x_grid_0 and x[k][0] <= x_min[0] + (i + 1) * x_grid_0 and x[k][1] >= x_min[1] + j * x_grid_1 and x[k][1] <= x_min[1] + (j + 1) * x_grid_1:
                        data_points_in_client.append(k)
                data_points_in_clients.append(data_points_in_client)
        #  check if data_points_in_clients is empty
        for i in range(len(data_points_in_clients)):
            if len(data_points_in_clients[i]) == 0:
                print('data_points_in_clients is empty')
                # extract 3 points from other clients
                for j in range(len(data_points_in_clients)):
                    if len(data_points_in_clients[j]) > 4:
                        data_points_in_clients[i].extend(data_points_in_clients[j][:2])
                        # remove the 3 points from the client
                        data_points_in_clients[j] = data_points_in_clients[j][2:]
                        break
        # turn the index to dataloader
        y_power_ratios = torch.zeros(num_clients)
        for client_i in range(num_clients):
            data_points_in_client = data_points_in_clients[client_i]
            x_in_client = x[data_points_in_client]
            y_in_client = y[data_points_in_client]
            train_loaders.append(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x_in_client, y_in_client),
                    batch_size=local_bs,
                    shuffle=True
                )
            )
            # get the average value of y_in_client
            y_power_ratios[client_i] = torch.mean(y_in_client)
        # normalize the y_power_ratios
        # y_power_ratios = y_power_ratios / torch.sum(y_power_ratios)
        if args.power_ratio == 'exp':
            y_power_ratios = torch.exp(y_power_ratios)/torch.sum(torch.exp(y_power_ratios))
        else:
            y_power_ratios = (1/torch.abs(y_power_ratios)) / torch.sum(1/torch.abs(y_power_ratios)) 
        test_loaders = [global_test_loader for i in range(num_clients)]

    # method 3: split the dataset by every datapoint
    elif args.split == 'point':
        for i, data_point in enumerate(train_loader):
            train_loaders.append(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.unsqueeze(data_point[0], dim=0),torch.unsqueeze(data_point[1],dim=1)),
                    batch_size=local_bs,
                    shuffle=True
                )
            )

        test_loaders = [global_test_loader for i in range(num_clients)]
        y = train_loader[:][1]
        y_power_ratios = torch.exp(y)/torch.sum(torch.exp(y))
    elif args.split == '2points':
        # get the x,y from traiin_loader
        x = train_loader[:][0]
        y = train_loader[:][1]
        # randomly select 2 points from the dataset
        data_points_in_clients = []
        for i in range(num_clients):
            # randomly split the dataset into multiple clients
            chunk_size = len(x) // num_clients
            start = i * chunk_size
            end = (i + 1) * chunk_size
            data_points_in_client = list(range(start, end))
            data_points_in_clients.append(data_points_in_client)
        # turn the index to dataloader
        y_power_ratios = torch.zeros(num_clients)
        for client_i in range(num_clients):
            data_points_in_client = data_points_in_clients[client_i]
            x_in_client = x[data_points_in_client]
            y_in_client = y[data_points_in_client]
            train_loaders.append(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x_in_client, y_in_client),
                    batch_size=local_bs,
                    shuffle=True
                )
            )
            # get the average value of y_in_client
            y_power_ratios[client_i] = torch.mean(y_in_client)
        # normalize the y_power_ratios
        if args.power_ratio == 'exp':
            y_power_ratios = torch.exp(y_power_ratios)/torch.sum(torch.exp(y_power_ratios))
        else:
            y_power_ratios = (1/torch.abs(y_power_ratios)) / torch.sum(1/torch.abs(y_power_ratios)) 
        test_loaders = [global_test_loader for i in range(num_clients)]
    return train_loaders, test_loaders, global_test_loader, y_power_ratios
