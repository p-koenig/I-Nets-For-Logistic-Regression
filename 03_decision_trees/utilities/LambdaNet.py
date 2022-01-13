#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
#from tqdm import tqdm_notebook as tqdm
#import pickle
import numpy as np
import pandas as pd
import scipy as sp

from functools import reduce
from more_itertools import random_product 

#import math

from joblib import Parallel, delayed
from collections.abc import Iterable
#from scipy.integrate import quad

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score 


import tensorflow as tf
import random 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#import tensorflow_addons as tfa

#udf import
#from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *
from utilities.DecisionTree_BASIC import *

import copy

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
                                
#######################################################################################################################################################
##################################################################Lambda Net Wrapper###################################################################
#######################################################################################################################################################
class LambdaNetDataset():
    
    lambda_net_list = None
    
    network_parameters_array = None
    network_list = None
    
    seed_list = None
    index_list = None
    
    target_function_parameters_array = None
    target_function_list = None
    
    X_test_lambda_array = None
    #y_test_lambda_array = None
    
    def __init__(self, lambda_net_list):
        
        self.lambda_net_list = lambda_net_list
        
        self.network_parameters_array = np.array([lambda_net.network_parameters for lambda_net in lambda_net_list])
        self.network_list = [lambda_net.network for lambda_net in lambda_net_list]
        
        self.seed_list = np.array([lambda_net.seed for lambda_net in lambda_net_list])
        self.index_list = np.array([lambda_net.index for lambda_net in lambda_net_list])
        
        self.target_function_parameters_array = np.array([lambda_net.target_function_parameters for lambda_net in lambda_net_list])
        self.target_function_list = [lambda_net.target_function for lambda_net in lambda_net_list]
      
        self.X_test_lambda_array = np.array([lambda_net.X_test_lambda for lambda_net in lambda_net_list])
        #self.y_test_lambda_array = np.array([lambda_net.y_test_lambda for lambda_net in lambda_net_list])
    
        self.shape = (self.network_parameters_array.shape[0], 1 + 1 + self.network_parameters_array.shape[1] + self.target_function_parameters_array.shape[1])
    
    def __repr__(self):
        return str(self.as_pandas(config).head())
    def __str__(self):
        return str(self.as_pandas(config).head())
    
    def __len__(self):
        return len(self.lambda_net_list)
        
    def predict(self, X_data, n_jobs=1):  
        y_pred_list = []
        
        if n_jobs == 1:
            for network in self.network_list:
                y_pred = network.predict(X_data)
                y_pred_list.append(y_pred)
        else:
            parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky')
            
            def predict_from_network(network, X_data):
                return network.predict(X_data)
            
            y_pred_list = parallel(delayed(predict_from_network)(network, X_data) for network in self.network_list)   
            
        
        return np.array(y_pred_list)
    
    def predict_test_data(self, n_jobs=1):  
        y_pred_list = []
        
        if n_jobs == 1:
            for network in self.network_list:
                y_pred = network.predict_test_data()
                y_pred_list.append(y_pred)
        else:
            parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky')
            
            def predict_from_network(network):
                return network.predict_test_data()
            
            y_pred_list = parallel(delayed(predict_from_network)(network) for network in self.network_list)   
            
        
        return np.array(y_pred_list)

    def predict_target_function(self, X_data, n_jobs=1):  
        y_pred_list = []
        
        if n_jobs == 1:
            for network in self.network_list:
                y_pred = network.predict_target_function(X_data)
                y_pred_list.append(y_pred)
        else:
            parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky')
            
            def predict_from_function(network, X_data):
                return network.predict_target_function(X_data)
            
            y_pred_list = parallel(delayed(predict_from_function)(target_function, X_data) for target_function in self.target_function_list)   
            
        
        return np.array(y_pred_list)
    
    def predict_test_data_target_function(self, n_jobs=1):  
        y_pred_list = []
        
        if n_jobs == 1:
            for network in self.network_list:
                y_pred = network.predict_test_data_target_function()
                y_pred_list.append(y_pred)
        else:
            parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky')
            
            def predict_from_function(network):
                return network.predict_test_data_target_function()
            
            y_pred_list = parallel(delayed(predict_from_function)(target_function) for target_function in self.target_function_list)   
            
        
        return np.array(y_pred_list)    
    
    
    def as_pandas(self, config):  
                
        lambda_dataframe = pd.DataFrame(data=[lambda_net.as_array() for lambda_net in self.lambda_net_list], 
                                columns=self.lambda_net_list[0].return_column_names(config), 
                                index=[lambda_net.index for lambda_net in self.lambda_net_list])
        lambda_dataframe['seed'] = lambda_dataframe['seed'].astype(int)
        
        return lambda_dataframe

    
    def get_lambda_nets_by_seed(self, seed_list):
        lambda_nets_by_seed = []
        for lambda_net in self.lambda_net_list:
            if lambda_net.seed in seed_list:
                lambda_nets_by_seed.append(lambda_net)
    
        return LambdaNetDataset(lambda_nets_by_seed)
    
    def get_lambda_nets_by_lambda_index(self, lambda_index_list):
        lambda_nets_by_lambda_index = []
        for lambda_net in self.lambda_net_list:
            if lambda_net.index in lambda_index_list:
                lambda_nets_by_lambda_index.append(lambda_net)
    
        return LambdaNetDataset(lambda_nets_by_lambda_index) 
    
    def get_lambda_net_by_lambda_index(self, lambda_index):
        for lambda_net in self.lambda_net_list:
            if lambda_net.index in lambda_index:
                return lambda_net
    
        return None
    
    def sample(self, size, seed=42):
        
        assert isinstance(size, int) or isinstance(size, float), 'Wrong sample size specified'
        
        random.seed(seed)
        
        sample_lambda_net_list = None
        
        if isinstance(size, int):
            sample_lambda_net_list = random.sample(self.lambda_net_list, size)
        elif isinstance(size, float):
            size = int(np.round(len(self.lambda_net_list)*size))
            sample_lambda_net_list = random.sample(self.lambda_net_list, size)
            
        return LambdaNetDataset(sample_lambda_net_list)

    

    

class LambdaNet():
    network_parameters = None
    network = None
    
    seed = None
    index = None
    
    target_function_parameters = None
    target_function = None
    
    X_test_lambda = None
    #y_test_lambda = None
    
    config = None
    
    #def __init__(self, line_weights, line_X_data, line_y_data, config):
    def __init__(self, line_weights, config):
        from utilities.utility_functions import network_parameters_to_network, generate_decision_tree_from_array
        
        assert isinstance(line_weights, np.ndarray), 'line is no array: ' + str(line_weights) 
        
        self.number_of_variables = config['data']['number_of_variables']  
        self.number_of_classes = config['data']['num_classes']  
        
        self.function_parameter_size = get_number_of_function_parameters(config['function_family']['dt_type'], config['function_family']['maximum_depth'], self.number_of_variables, self.number_of_classes)#(2 ** config['function_family']['maximum_depth'] - 1) * (self.number_of_variables + 1) + (2 ** config['function_family']['maximum_depth']) * self.number_of_classes
        
        self.network_parameter_size = get_number_of_lambda_net_parameters(config['lambda_net']['lambda_network_layers'], self.number_of_variables, self.number_of_classes)
        
        self.index = int(line_weights[0])
        self.seed = int(line_weights[1])
        
        try:
            condition = config['data']['dt_type_train'] != None or config['data']['maximum_depth_train'] != None or config['data']['decision_sparsity_train'] != None
        except:
            condition = False
                        
        if config['data']['function_generation_type'] == 'make_classification' or condition:
            function_parameter_size_actual = self.function_parameter_size
            self.function_parameter_size = len(line_weights) - self.network_parameter_size - 2 #2 for seed and index
            self.target_function_parameters = np.array([0 for i in range(function_parameter_size_actual)])
        else:
            self.target_function_parameters = line_weights[range(2, self.function_parameter_size+2)].astype(float)
        
        self.network_parameters = line_weights[self.function_parameter_size+2:].astype(float)
        
        if config['i_net']['normalize_lambda_nets']:
            self.network_parameters = shaped_network_parameters_to_array(normal_neural_net(shape_flat_network_parameters(copy.deepcopy(self.network_parameters), generate_base_model(config).get_weights()), config), config)        
        
        assert self.network_parameters.shape[0] == self.network_parameter_size, 'weights have incorrect shape ' + str(self.network_parameters.shape[0]) + ' but should be ' + str(self.network_parameter_size)
        
        #assert self.index == line_X_data[0] == line_y_data[0], 'indices do not match: ' + str(self.index) + ', ' + str(line_X_data[0]) + ', ' + str(line_y_data[0])
        
        #line_X_data = line_X_data[1:]
        #line_y_data = line_y_data[1:]
        #self.X_test_lambda = np.transpose(np.array([line_X_data[i::self.number_of_variables] for i in range(self.number_of_variables)]))
        #self.y_test_lambda = line_y_data.reshape(-1,1)
                
        data_generation_seed = self.seed + self.index
        self.X_test_lambda = generate_random_data_points_custom(low=config['data']['x_min'], 
                                                            high=config['data']['x_max'], 
                                                            size=int(np.round(config['data']['lambda_dataset_size']*0.25)), 
                                                            variables=config['data']['number_of_variables'], 
                                                            distrib=config['evaluation']['random_evaluation_dataset_distribution'],
                                                            categorical_indices=config['data']['categorical_indices'],
                                                            seed=data_generation_seed)          
        
        
        #self.network = network_parameters_to_network(self.network_parameters, config)           
        #self.target_function = generate_decision_tree_from_array(self.target_function_parameters, config)
                        
    def __repr__(self):
        return str(self.network_parameters)
    def __str__(self):
        return str(self.network_parameters)
        
    def initialize_network(self, config, base_model=None):
        self.network = network_parameters_to_network(self.network_parameters, config, base_model)           
        
    def initialize_target_function(self, config):
        self.target_function = generate_decision_tree_from_array(self.target_function_parameters, config)

    def predict(self, X_data):
        if self.network is not None:
            y_pred = self.network.predict(X_data)
        else:
            y_pred = network_parameters_to_pred(self.network_parameters, X_data)
            
        return y_pred
        
    def predict_test_data(self):
                
        if self.network is not None:
            y_pred = self.network.predict(self.X_test_lambda)
        else:
            y_pred = network_parameters_to_pred(self.network_parameters, self.X_test_lambda)
            
        return y_pred        
    
    def predict_target_function(self, X_data):
        if self.target_function is not None:
            y_pred = self.target_function.predict(X_data)
        else:
            pass
            #TBD
            #y_pred = (self.target_function_parameters, X_data)
            
        return y_pred    
    
    def predict_test_data_target_function(self):
        if self.target_function is not None:
            y_pred = self.target_function.predict(self.X_test_lambda)
        else:
            
            pass
            #TBD
            #y_pred = (self.target_function_parameters, self.X_test_lambda)
            
        return y_pred    
 

    
    def as_pandas(self, config): 
        columns = return_column_names(self, config)
        data = as_array(self)
        
        df = pd.DataFrame(data=data, columns=columns, index=[self.index])
        df['seed'] = df['seed'].astype(int)
        
        return df
    
    def as_array(self):
        data = np.hstack([self.index, self.seed, self.target_function_parameters, self.network_parameters])
        return data
    
    def return_column_names(self, config):  
        
        from utilities.utility_functions import flatten_list, generate_decision_tree_identifier
        
        target_function_identifiers = generate_decision_tree_identifier(config)
        network_parameter_identifiers = ['wb_' + str(i) for i in range(self.network_parameters.shape[0])]
        
        columns = list(flatten_list(['index', 'seed', target_function_identifiers, network_parameter_identifiers]))
                
        return columns 

    

#######################################################################################################################################################
#################################################################Lambda Net TRAINING###################################################################
#######################################################################################################################################################

def generate_lambda_net_from_config(config, seed=None):

    if seed is None:
        seed = config['computation']['RANDOM_SEED']

    output_neurons = 1 if config['data']['num_classes']==2 else config['data']['num_classes']
    output_activation = 'sigmoid' if config['data']['num_classes']==2 else 'softmax'

    model = Sequential()

    #kerase defaults: kernel_initializer='glorot_uniform', bias_initializer='zeros'               
    model.add(Dense(config['lambda_net']['lambda_network_layers'][0], activation='relu', input_dim=config['data']['number_of_variables'], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), bias_initializer='zeros'))

    if config['lambda_net']['dropout_lambda'] > 0:
        model.add(Dropout(config['lambda_net']['dropout_lambda']))

    for neurons in config['lambda_net']['lambda_network_layers'][1:]:
        model.add(Dense(neurons, 
                        activation='relu', 
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), 
                        bias_initializer='zeros'))

        if config['lambda_net']['dropout_lambda'] > 0:
            model.add(Dropout(config['lambda_net']['dropout_lambda']))   

    model.add(Dense(output_neurons, 
                    activation=output_activation, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), 
                    bias_initializer='zeros'))

    
    optimizer = tf.keras.optimizers.get(config['lambda_net']['optimizer_lambda'])
    
    try:
        optimizer.learning_rate = learning_rate=config['lambda_net']['learning_rate_lambda']
    except:
        pass
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',#tf.keras.losses.get(config['lambda_net']['loss_lambda']),
                  metrics=[tf.keras.metrics.get("binary_accuracy")]
                 )
    
    return model

def train_lambda_net(config,
                     lambda_index,
                     X_data_with_function, 
                     y_data_with_function, 
                     callbacks=None, 
                     return_history=False, 
                     printing=False, 
                     return_model=False):
    
    
    
    from utilities.utility_functions import generate_paths, pairwise
    
    assert (X_data_with_function[0].values == y_data_with_function[0].values).all()
    
    function = X_data_with_function[0]
    
    X_data = X_data_with_function[1]
    y_data = y_data_with_function[1]
    
    if isinstance(X_data, pd.DataFrame) or isinstance(X_data_lambda, pd.Series):
        X_data = X_data.values
    if isinstance(y_data, pd.DataFrame) or isinstance(y_data, pd.Series):
        y_data = y_data.values
    if isinstance(function, pd.DataFrame) or isinstance(function, pd.Series):
        function = function.values            
    
    paths_dict = generate_paths(config, path_type = 'lambda_net') 
    
    current_seed = config['computation']['RANDOM_SEED']
    if config['lambda_net']['number_initializations_lambda'] == -1:
        current_seed = current_seed + lambda_index
    if config['lambda_net']['number_initializations_lambda'] > 1:
        current_seed = (current_seed + lambda_index) % config['lambda_net']['number_initializations_lambda']            
    

                
    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=config['computation']['RANDOM_SEED'])           
    X_train, X_valid, y_train, y_valid= train_test_split(X_train_with_valid, y_train_with_valid, test_size=0.1, random_state=config['computation']['RANDOM_SEED'])           

    model = generate_lambda_net_from_config(config, current_seed)

    if config['lambda_net']['early_stopping_lambda']:
        if callbacks == None:
            callbacks = []
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=config['lambda_net']['early_stopping_min_delta_lambda'], verbose=0, mode='min', restore_best_weights=True)
        callbacks.append(early_stopping)

    model_history = model.fit(X_train,
                  y_train, 
                  epochs=config['lambda_net']['epochs_lambda'], 
                  batch_size=config['lambda_net']['batch_lambda'], 
                  callbacks=callbacks,
                  validation_data=(X_valid, y_valid),
                  verbose=0,
                  workers=0)

    history = model_history.history

    y_train_pred = model.predict(X_train) 
    y_valid_pred = model.predict(X_valid)                
    y_test_pred = model.predict(X_test)


    pred_list = {
                  'lambda_index': lambda_index,
                  'X_train': X_train, 
                  'y_train': y_train, 
                  'y_train_pred': y_train_pred, 
                  'X_valid': X_valid, 
                  'y_valid': y_valid,
                  'y_valid_pred': y_valid_pred,
                  'X_test': X_test, 
                  'y_test': y_test,
                  'y_test_pred': y_test_pred,
                }

    scores_train = None#evaluate_lambda_net(y_train, y_train_pred, 'TRAIN')
    scores_valid = None#evaluate_lambda_net(y_valid, y_valid_pred, 'VALID')

    scores_list = [lambda_index,
                 scores_train,
                 scores_valid]            

        
    if printing:        

        directory = './data/weights/weights_' + paths_dict['path_identifier_lambda_net_data']
        
        path_weights = directory  + '/' + 'weights' + '.txt' 
        #path_X_data = directory + '/' + 'X_test_lambda' + '.txt'
        #path_y_data = directory + '/' + 'y_test_lambda' + '.txt'
        
        with open(path_weights, 'a') as text_file: 
            text_file.write(str(lambda_index))
            text_file.write(', ' + str(current_seed))
            for value in function: 
                text_file.write(', ' + str(value))  
            for value in shaped_network_parameters_to_array(model.get_weights(), config): 
                text_file.write(', ' + str(value))                 
            #for layer_weights, biases in pairwise(model.get_weights()):   
            #    for neuron in layer_weights:
            #        for weight in neuron:
            #            text_file.write(', ' + str(weight))
            #    for bias in biases:
            #        text_file.write(', ' + str(bias))
            text_file.write('\n')

            text_file.close()          

        #with open(path_X_data, 'a') as text_file: 
        #    text_file.write(str(lambda_index))
        #    for row in X_test:
        #        for value in row:
        #            text_file.write(', ' + str(value))
        #    text_file.write('\n')
        #    text_file.close()                

        #with open(path_y_data, 'a') as text_file: 
        #    text_file.write(str(lambda_index))          
        #    for value in y_test.flatten():
        #        text_file.write(', ' + str(value))
        #    text_file.write('\n')
        #    text_file.close()                    


            
    if return_model:
        return {'index': lambda_index, 
                'seed': current_seed, 
                'function': function, 
                'scores': scores_list, 
                'preds': pred_list, 
                'history': history, 
                'model': model}
    elif return_history:
        return {'index': lambda_index, 
                'seed': current_seed, 
                'function': function, 
                'scores': scores_list, 
                'preds': pred_list, 
                'history': history}
    else:
        return {'index': lambda_index, 
                'seed': current_seed, 
                'function': function, 
                'scores': scores_list, 
                'preds': pred_list}
    
    
#######################################################################################################################################################
#######################################################LAMBDA-NET EVALUATION FUNCTION##################################################################
#######################################################################################################################################################

def evaluate_lambda_net(y_data_real_lambda, 
                        y_data_pred_lambda,
                        identifier=''):
        
    y_data_real_lambda = np.nan_to_num(y_data_real_lambda)
    
    y_data_pred_lambda = np.nan_to_num(y_data_pred_lambda)
    y_data_pred_lambda_int = np.round(y_data_pred_lambda).astype(int)
    
    accuracy_score_real_VS_predLambda = np.round(accuracy_score(y_data_real_lambda, y_data_pred_lambda_int), 4)  
    auc_real_VS_predLambda = np.round(roc_auc_score(y_data_real_lambda, y_data_pred_lambda), 4)    
    f1_score_real_VS_predLambda = np.round(f1_score(y_data_real_lambda, y_data_pred_lambda_int), 4)    
    log_loss_real_VS_predLambda = np.round(log_loss(y_data_real_lambda, y_data_pred_lambda), 4)
    
    
    
    
    return {
             'ACC FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': accuracy_score_real_VS_predLambda,
             'ROCAUC FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': auc_real_VS_predLambda,
             'F1 FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': f1_score_real_VS_predLambda,
             'LOGLOSS FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': log_loss_real_VS_predLambda,
            }
    
        

def get_number_of_lambda_net_parameters(lambda_network_layer_list, number_of_variables, num_classes):
    layers_with_input_output = flatten_list([number_of_variables, lambda_network_layer_list, [1 if num_classes == 2 else num_classes]])
    number_of_lambda_parameters = 0
    for i in range(len(layers_with_input_output)-1):
        number_of_lambda_parameters += (layers_with_input_output[i]+1)*layers_with_input_output[i+1]  
        
    return number_of_lambda_parameters