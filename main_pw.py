# main.py
from gnss_fl_learning_pw import *
from joblib import Parallel, delayed

grid_sizes = [(5,10)]#(2,2),(5,5),(5,10),'2points','point',(2,2),(2,3),(2,4),(3,4),(4,4),(4,5),(4,6),(4,7)] #(2,2),(2,4), (4,4),(4,5),(2,2),(2,4),(1,2),(2,2),(2,3),(1,2),(2,2),,(2,4)
weight_ratios = ['sample'] #'sample','equal'
data_names = ['R13'] #,'PL2','PL11','R2'
fl_pw = 1
algorithms = ['PowerFedAvg','Fedavg'] #'Fedavg','Fedprox-PL, PowerFedAvg
mus=[0] #,0.1,1,10,100
local_epochs = 2
epochs_init = 100
# model_mode = 'both'
model_modes = ['both']#,'PL', 'both','NN'
data_preprocesing = 1
lr = 0.01
test_mode = 1
split = 'random' #'point'

if fl_pw == 0:
    Parallel(n_jobs=1)(delayed(main)(fl_pw, grid_sizes, weight_ratios, data_name) \
        for data_name in data_names)
else:
    # for weight_ratio in weight_ratios:
    #     for grid_size in grid_sizes:
    #         main(fl_pw, grid_size, weight_ratio)
    Parallel(n_jobs=4)(delayed(main)(fl_pw, grid_size, weight_ratio, data_name,
         algorithm,mu,local_epochs,epochs_init,model_mode,\
            data_preprocesing=data_preprocesing,lr=lr, test_mode=test_mode, split=split) \
        for grid_size in grid_sizes \
            for weight_ratio in weight_ratios \
                for data_name in data_names \
                    for algorithm in algorithms \
                        for mu in mus \
                            for model_mode in model_modes)
