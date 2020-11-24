# ----------- Imports -----------

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw

# ------- Static variables ------

RANDOM_SEED = 42

# ----- Function declaration ----

# Lambda net specific

# Create neural network structure
def create_lambdanet_model(lambda_network_layers, input_neurons, dropout, optimizer, loss_function, metrics):
    
    model = Sequential()

    model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=input_neurons)) #1024
    
    if dropout > 0:
        model.add(Dropout(dropout))

    for neurons in lambda_network_layers[1:]:
        model.add(Dense(neurons, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))   
            
    model.add(Dense(1)) # activation=linear ist default
    
    # only keras functions allowed
    model.compile(optimizer=optimizer,
                  loss=loss_function, #huber_loss(val_min, val_max), #'mape',#'mean_absolute_error',#root_mean_squared_error,
                  metrics=metrics)
    return model

# Calculate metrics after a lambda net is fully trained
def calc_metrics(data_dict, advanced_size):
       
    metrics_dics = []
    
    #-----------------------------------------------------------------
    # Relative data metrics
    
    data_names = [('valid', 'VALID'), ('test', 'TEST')]   
    
    for data, DATA in data_names:
        
        tmp_dic = dict()
        
        # Standard Metrics ---
        
        functions = [mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error_keras, r2_score,
                     relative_absolute_average_error, relative_maximum_average_error]
        function_names = ['MAE', 'RMSE', 'MAPE', 'R2' ,'RAAE', 'RMAE']
        round_ = 4

        for func, name in zip(functions, function_names):
            tmp_dic[name+' FV '+DATA+' PRED'] = np.round(func(data_dict['y_'+data], data_dict['y_pred_'+data]), round_)
            tmp_dic[name+' FV '+DATA+' POLY'] = np.round(func(data_dict['y_'+data], data_dict['y_'+data+'_polynomial']), round_)
            tmp_dic[name+' FV '+DATA+' POLY PRED'] = np.round(func(data_dict['y_'+data+'_polynomial'], data_dict['y_pred_'+data]),
                                                              round_)
            tmp_dic[name+' FV '+DATA+' LSTSQ'] = np.round(func(data_dict['y_'+data], data_dict['y_'+data+'_lstsq']), round_)

        # Advancded Metrics ---

        x_data_advanced = data_dict['x_'+data][:advanced_size]
        y_data_advanced = data_dict['y_'+data][:advanced_size]

        # Frechet Distance
        tmp_dic['FD FV '+DATA+' PRED'] = np.round(frechet_dist(np.column_stack((x_data_advanced,
                                                                                y_data_advanced)),
                                                               np.column_stack((x_data_advanced,
                                                                                data_dict['y_pred_'+data][:advanced_size]))),
                                                  round_)
        tmp_dic['FD FV '+DATA+' POLY'] = np.round(frechet_dist(np.column_stack((x_data_advanced,
                                                                                y_data_advanced)),
                                                               np.column_stack((x_data_advanced,
                                                                                data_dict['y_'+data+'_polynomial'][:advanced_size]))),
                                                  round_)
        tmp_dic['FD FV '+DATA+' POLY PRED'] = np.round(frechet_dist(np.column_stack((x_data_advanced,
                                                                                 data_dict['y_'+data+'_polynomial'][:advanced_size])),
                                                                    np.column_stack((x_data_advanced,
                                                                                     data_dict['y_pred_'+data][:advanced_size]))),
                                                       round_)
        tmp_dic['FD FV '+DATA+' LSTSQ'] = np.round(frechet_dist(np.column_stack((x_data_advanced,
                                                                                 y_data_advanced)), 
                                                                np.column_stack((x_data_advanced,
                                                                                 data_dict['y_'+data+'_lstsq'][:advanced_size]))),
                                                   round_)
        # Dynamic Time Warping
        tmp_dic['DTW FV '+DATA+' PRED'] = np.round(dtw(np.column_stack((x_data_advanced,
                                                                        y_data_advanced)),
                                                       np.column_stack((x_data_advanced,
                                                                        data_dict['y_pred_'+data][:advanced_size])))[0], 
                                                   round_) 
        tmp_dic['DTW FV '+DATA+' POLY'] = np.round(dtw(np.column_stack((x_data_advanced, 
                                                                        y_data_advanced)),
                                                       np.column_stack((x_data_advanced,
                                                                        data_dict['y_'+data+'_polynomial'][:advanced_size])))[0],
                                                   round_)
        tmp_dic['DTW FV '+DATA+' POLY PRED'] = np.round(dtw(np.column_stack((x_data_advanced,
                                                                             data_dict['y_'+data+'_polynomial'][:advanced_size])),
                                                            np.column_stack((x_data_advanced,
                                                                             data_dict['y_pred_'+data][:advanced_size])))[0],
                                                        round_)   
        tmp_dic['DTW FV '+DATA+' LSTSQ'] = np.round(dtw(np.column_stack((x_data_advanced,
                                                                         y_data_advanced)), 
                                                        np.column_stack((x_data_advanced, 
                                                                         data_dict['y_'+data+'_lstsq'][:advanced_size])))[0], 
                                                    round_)
        
        metrics_dics.append(tmp_dic)
    
    #----------------------------------------------------------------- 
    # Static data metrics

    functions = [np.std, np.mean]
    function_names = ['STD', 'MEAN']
    
    for func, name in zip(functions, function_names):
        
        tmp_dic = dict()
        
        tmp_dic[(name+' FV TRAIN REAL')] = func(data_dict['y_train'])
        tmp_dic[(name+' FV VALID REAL')] = func(data_dict['y_valid'])
        tmp_dic[(name+' FV TEST REAL')] = func(data_dict['y_test'])

        tmp_dic[(name+' FV VALID PRED')] = func(data_dict['y_pred_valid'])
        tmp_dic[(name+' FV TEST PRED')] = func(data_dict['y_pred_test'])

        tmp_dic[(name+' FV VALID POLY')] = func(data_dict['y_valid_polynomial'])
        tmp_dic[(name+' FV TEST POLY')] = func(data_dict['y_test_polynomial'])

        tmp_dic[(name+' FV VALID LSTSQ')] = func(data_dict['y_valid_lstsq'])
        tmp_dic[(name+' FV TEST LSTSQ')] = func(data_dict['y_test_lstsq'])
        
        metrics_dics.append(tmp_dic)
    
    #-----------------------------------------------------------------

    return metrics_dics # order: [valid_metrics_dic, test_metrics_dic, std_metrics_dic, mean_metrics_dic] 


# - deprecated