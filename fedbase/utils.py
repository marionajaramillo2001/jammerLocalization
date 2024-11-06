import torch
from pathlib import Path
import json
import numpy as np
import datetime as d

def save_checkpoint(model_structure, model_parameter, optimizer_parameter, path):
    checkpoint = {'model_structure': model_structure,
                  'model_parameter': model_parameter, 'optimizer_parameter': optimizer_parameter}
    torch.save(checkpoint, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model_structure']
    model.load_state_dict(checkpoint['model_parameter'])
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    # optimizer = checkpoint['optimizer']
    # model.eval()
    return model, checkpoint['optimizer_parameter']


# KL divergence variant log p/q,abs(log p-log q)
def blackbox_mc(model, testset):
    outputs = model(testset)
    return outputs.sum(axis=1)/len(testset)

# model similarity
def similarity(model_1,model_2):
    return abs(log(blackbox_mc(model_1))-log(blackbox_mc(model_2)))

def log(file_name, nodes, server, H=None,assign_method=None, bayes=None):
    local_file = './log/' + file_name + "_" + d.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(np.random.choice(10**3)) + ".json"
    log = {}
    log['node'] = {}
    for i in range(len(nodes)):
        log['node'][str(i)] = list(nodes[i].test_metrics)
        # log['node_confusion_matrix'] = list(nodes[i].con_mats)
    try:
        log['server'] = list(server.test_metrics)
        log['best_assignment'] = list(server.test_metrics_best)
        log['clustering'] = str(server.clustering)
        log['assign_method'] = str(assign_method)
        log['bayes'] = str(bayes)
        log['H'] = str(H)
        # log['confusion_matrix'] = list(server.con_mats)
        # con_mats is a numpy array, can not be saved in json,how to save it?
        # transfer con_mats to a format can be saved in json
        log['confusion_matrix'] = []
        for i in range(len(server.con_mats)):
            log['confusion_matrix'].append(server.con_mats[i].tolist())

    except:
        print('No server')
    # pd.to_pickle(log, local_file)
    # print(log)
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    with  open(local_file, 'w') as handle:
        json.dump(log, handle, indent=4)
    # read
    # if os.path.exists(local_file):
    #     with open(local_file, 'r') as f:
    #         log = json.load(f)
    #         # print(log)
        
def add_(input_str):
    return f'_{str(input_str)}' if input_str is not None else ''