#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

import itertools 
from tqdm.notebook import tqdm
#import pickle
#import cloudpickle
import dill 

import traceback

import numpy as np
import pandas as pd
import scipy as sp
import time

from functools import reduce
from more_itertools import random_product 

#import math
from joblib import Parallel, delayed


from collections.abc import Iterable
#from scipy.integrate import quad

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
#from similaritymeasures import frechet_dist, area_between_two_curves, dtw

from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import tensorflow as tf
import tensorflow_probability as tfp

import autokeras as ak
from autokeras import adapters, analysers
from keras_tuner.engine import hyperparameters
from tensorflow.python.util import nest

import random 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from matplotlib import pyplot as plt
import seaborn as sns

from sympy import Symbol, sympify, lambdify, abc, SympifyError

#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *
from utilities.DecisionTree_BASIC import *

from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

            
#######################################################################################################################################################
######################################################################AUTOKERAS BLOCKS#################################################################
#######################################################################################################################################################

class CombinedOutputInet(ak.Head):

    def __init__(self, loss = None, metrics = None, output_dim=None, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, hp, inputs=None):    
        #inputs = nest.flatten(inputs)
        #if len(inputs) == 1:
        #    return inputs
        output_node = concatenate(inputs)           
        return output_node

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self._add_one_dimension = len(analyser.shape) == 1

    def get_adapter(self):
        return adapters.RegressionAdapter(name=self.name)

    def get_analyser(self):
        return analysers.RegressionAnalyser(
            name=self.name, output_dim=self.output_dim
        )

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )
        return hyper_preprocessors            

class OutputInet(ak.Head):

    def __init__(self, loss = None, metrics = None, output_dim=None, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, hp, inputs=None):    
        #inputs = nest.flatten(inputs)
        #if len(inputs) == 1:
        #    return inputs
        output_node = inputs#concatenate(inputs)           
        return output_node

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self._add_one_dimension = len(analyser.shape) == 1

    def get_adapter(self):
        return adapters.RegressionAdapter(name=self.name)

    def get_analyser(self):
        return analysers.RegressionAnalyser(
            name=self.name, output_dim=self.output_dim
        )

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )
        return hyper_preprocessors            

    
class CustomDenseInet(ak.Block):
    
    neurons=None
    activation=None
    
    def __init__(self, neurons=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.activation = activation
        
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(units=self.neurons, activation=self.activation)
        #layer = Dense(1, activation='linear')
        output_node = layer(input_node)
        return output_node    

class SingleDenseLayerBlock(ak.Block):
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.Dense(
            hp.Int("num_units", min_value=16, max_value=512, step=16, default=32)
        )
        output_node = layer(input_node)
        return output_node
    
class DeepDenseLayerBlock(ak.Block):
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        
        num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1, default=2)
        activation = hp.Choice("activation", values=['relu', 'sigmoid', 'tanh'], default='relu')
        
        num_units_list = []
        dropout_list = []
        for i in range(5):
            num_units = hp.Int("num_units_" + str(i), min_value=64, max_value=2048, step=64, default=512)
            dropout = hp.Choice("dropout_" + str(i), [0.0, 0.1, 0.3, 0.5], default=0.0)
            num_units_list.append(num_units)
            dropout_list.append(dropout)
        
        for i in range(num_layers):
            if i == 0:
                hidden_node = tf.keras.layers.Dense(
                    units = num_units_list[i],
                    activation = activation
                )(input_node)
                hidden_node = tf.keras.layers.Dropout(
                    rate = dropout_list[i]
                )(hidden_node)   
            else:
                hidden_node = tf.keras.layers.Dense(
                    units = num_units_list[i],
                    activation = activation
                )(hidden_node)
                hidden_node = tf.keras.layers.Dropout(
                    rate = dropout_list[i]
                )(hidden_node)               
        return hidden_node  

#######################################################################################################################################################
#################################################################I-NET RESULT CALCULATION##############################################################
#######################################################################################################################################################
    
def interpretation_net_training(lambda_net_train_dataset, 
                                lambda_net_valid_dataset, 
                                lambda_net_test_dataset,
                                config,
                                callback_names=[]):

    
    print('----------------------------------------------- TRAINING INTERPRETATION NET -----------------------------------------------')
    start = time.time() 
    
    (history, 
     (X_valid, y_valid), 
     (X_test, y_test), 
     loss_function, 
     metrics,
     encoder_model) = train_inet(lambda_net_train_dataset,
                        lambda_net_valid_dataset,
                        lambda_net_test_dataset,
                        config,
                        callback_names)
    
    end = time.time()     
    inet_train_time = (end - start) 
    minutes, seconds = divmod(int(inet_train_time), 60)
    hours, minutes = divmod(minutes, 60)        
    print('Training Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')    
           
    if config['computation']['load_model']:
        paths_dict = generate_paths(config, path_type = 'interpretation_net')

        path = './data/results/' + paths_dict['path_identifier_interpretation_net'] + '/history' + '.pkl'
        with open(path, 'rb') as f:
            history = pickle.load(f)  
                

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------ LOADING MODELS -----------------------------------------------------')

    start = time.time() 

    model = load_inet(loss_function=loss_function, metrics=metrics, config=config)

    end = time.time()     
    inet_load_time = (end - start) 
    minutes, seconds = divmod(int(inet_load_time), 60)
    hours, minutes = divmod(minutes, 60)        
    print('Loading Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
       
    if not config['i_net']['nas']:
        generate_history_plots(history, config)
        save_results(history, config)    
    

            
    return ((X_valid, y_valid), 
            (X_test, y_test),
            
            history,
            loss_function,
            metrics,
            
            model,
            encoder_model)
    
    
#######################################################################################################################################################
######################################################################I-NET TRAINING###################################################################
#######################################################################################################################################################

def load_inet(loss_function, metrics, config):
    
    from utilities.utility_functions import generate_paths
    
    dt_string =  ('_depth' + str(config['function_family']['maximum_depth']) +
              '_beta' + str(config['function_family']['beta']) +
              '_decisionSpars' +  str(config['function_family']['decision_sparsity']) + 
              '_' + str(config['function_family']['dt_type']))
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    if config['i_net']['nas']:
        path = './data/saved_models/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + dt_string         
        #path = './data/saved_models/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + '_' + paths_dict['path_identifier_lambda_net_data'] + dt_string         
    else:
        path = './data/saved_models/' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version'])
        
        #path = './data/saved_models/' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version'])


    model = []
    from tensorflow.keras.utils import CustomObjectScope
    loss_function = dill.loads(loss_function)
    metrics = dill.loads(metrics)       

    #with CustomObjectScope({'custom_loss': loss_function}):
    custom_object_dict = {}
    custom_object_dict[loss_function.__name__] = loss_function
    for metric in  metrics:
        custom_object_dict[metric.__name__] = metric        
        
    model = tf.keras.models.load_model(path, custom_objects=custom_object_dict) # #, compile=False
        
    return model

 
    
        
def generate_inet_train_data(lambda_net_dataset, config, encoder_model=None):
    #X_data = None
    X_data_flat = None
    y_data = None
    normalization_parameter_dict = None
    
    X_data = lambda_net_dataset.network_parameters_array
    
        
    if not config['i_net']['optimize_decision_function']: #target polynomial as inet target
        y_data = lambda_net_dataset.target_function_parameters_array
    else:
        y_data = np.zeros_like(lambda_net_dataset.target_function_parameters_array)
        
    if config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2:
        print('RESTRUCTURING DATA')
        X_data, X_data_flat = restructure_data_cnn_lstm(X_data, config, subsequences=None)
    elif config['i_net']['data_reshape_version'] == 3: #autoencoder
        if encoder_model is None:
            X_data, X_data_flat, autoencoder = autoencode_data(X_data, config)
            return X_data, X_data_flat, y_data, autoencoder
        else:
            X_data, X_data_flat, _ = autoencode_data(X_data, config, encoder_model)
            #return X_data, X_data_flat, y_data, None
        
    return X_data, X_data_flat, y_data, None


def train_inet(lambda_net_train_dataset,
              lambda_net_valid_dataset,
              lambda_net_test_dataset, 
              config,
              callback_names):
    
    from utilities.utility_functions import unstack_array_to_list
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    dt_string =  ('_depth' + str(config['function_family']['maximum_depth']) +
              '_beta' + str(config['function_family']['beta']) +
              '_decisionSpars' +  str(config['function_family']['decision_sparsity']) + 
              '_' + str(config['function_family']['dt_type']))
    
    ############################## DATA PREPARATION ###############################
    
    random_model = generate_base_model(config)
    np.random.seed(config['computation']['RANDOM_SEED'])
        
    random_network_parameters = random_model.get_weights()
    network_parameters_structure = [network_parameter.shape for network_parameter in random_network_parameters]         

    (X_train, X_train_flat, y_train, encoder_model) = generate_inet_train_data(lambda_net_train_dataset, config)
    (X_valid, X_valid_flat, y_valid, _) = generate_inet_train_data(lambda_net_valid_dataset, config, encoder_model)
    if lambda_net_test_dataset is not None:
        (X_test, X_test_flat, y_test, _) = generate_inet_train_data(lambda_net_test_dataset, config, encoder_model)
    
    
    ############################## OBJECTIVE SPECIFICATION AND LOSS FUNCTION ADJUSTMENTS ###############################
    metrics = []
    loss_function = None
    
    distribution_dict_list = lambda_net_train_dataset.distribution_dict_list_list
    distribution_dict_list.extend(lambda_net_valid_dataset.distribution_dict_list_list)    
        
    use_distribution_list = False if config['data']['max_distributions_per_class'] is None else True
    
    if config['i_net']['function_value_loss']:
        if config['i_net']['function_representation_type'] == 1:
            pass
            #metrics.append(tf.keras.losses.get('mae'))
        if config['i_net']['optimize_decision_function']:
            loss_function = inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list)
            #metrics.append(inet_target_function_fv_loss_wrapper(config))
            for metric in config['i_net']['metrics']:
                metrics.append(inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, metric, use_distribution_list=use_distribution_list))  
                #metrics.append(inet_target_function_fv_metric_wrapper(config, metric))  
        else:
            loss_function = inet_target_function_fv_loss_wrapper(config)
            metrics.append(inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list))
            for metric in config['i_net']['metrics']:
                metrics.append(inet_target_function_fv_metric_wrapper(config, metric))  
                metrics.append(inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, metric, use_distribution_list=use_distribution_list))  
    else:
        metrics.append(inet_target_function_fv_loss_wrapper(config))
        metrics.append(inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list))
        if config['i_net']['optimize_decision_function']:
            raise SystemExit('Coefficient Loss not implemented for decision function optimization')            
        else:
            if config['i_net']['function_representation_type'] == 1:
                loss_function = tf.keras.losses.get('mae') #inet_coefficient_loss_wrapper(inet_loss)
            else:
                raise SystemExit('Coefficient Loss not implemented for selected function representation')
    
                
    distribution_dict_index_train = np.array([[i] for i in range(y_train.shape[0])])
    distribution_dict_index_valid = np.array([[len(distribution_dict_index_train) + i] for i in range(y_valid.shape[0])])
    
    print(distribution_dict_index_train.shape)
    print(X_train.shape)
    print(y_train.shape)
    
    #print('np.hstack((y_train, X_train_flat, distribution_dict_index_train))', np.hstack((y_train, X_train, distribution_dict_index_train)))
    #print('np.hstack((y_train, X_train_flat))', np.hstack((y_train, X_train)))
    if use_distribution_list:
        
        distribution_dict_row_array_train = lambda_net_train_dataset.distribution_dict_row_array
        distribution_dict_row_array_valid = lambda_net_valid_dataset.distribution_dict_row_array

        distribution_dict_row_array_train[distribution_dict_row_array_train == ' NaN'] = np.nan
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' normal'] = 0
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' uniform'] = 1
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' gamma'] = 2
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' exponential'] = 3
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' beta'] = 4
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' binomial'] = 5
        distribution_dict_row_array_train[distribution_dict_row_array_train == ' poisson'] = 6
        distribution_dict_row_array_train = distribution_dict_row_array_train.astype(np.float32)

        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' NaN'] = np.nan
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' normal'] = 0
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' uniform'] = 1
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' gamma'] = 2
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' exponential'] = 3
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' beta'] = 4
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' binomial'] = 5
        distribution_dict_row_array_valid[distribution_dict_row_array_valid == ' poisson'] = 6
        distribution_dict_row_array_valid = distribution_dict_row_array_valid.astype(np.float32)        
        if False:

            if config['i_net']['data_reshape_version'] is not None:
                y_train_model = np.hstack((y_train, X_train_flat, distribution_dict_row_array_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat, distribution_dict_row_array_valid)))   
            else:
                y_train_model = np.hstack((y_train, X_train, distribution_dict_row_array_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid, distribution_dict_row_array_valid))) 
        else:
            
            if False:
                #print(tf.executing_eagerly())
                #tf.config.run_functions_eagerly(False)
                #print(tf.executing_eagerly())
                random_evaluation_dataset_list_train_tf = tf.map_fn(generate_dataset_from_distributions_line_tf_wrapper(number_of_variables_function=config['data']['number_of_variables'], 
                                                       number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                       max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                       seed_function = np.random.randint(1_000_000), 
                                                       flip_percentage=0,
                                                       config=config), distribution_dict_row_array_train, parallel_iterations=1)
                
                random_evaluation_dataset_array_train = random_evaluation_dataset_list_train_tf.numpy()
                print(random_evaluation_dataset_array_train.shape)
                random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))
                print(random_evaluation_dataset_flat_array_train.shape)
                
                random_evaluation_dataset_list_valid_tf = tf.map_fn(generate_dataset_from_distributions_line_tf_wrapper(number_of_variables_function=config['data']['number_of_variables'], 
                                                       number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                       max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                       seed_function = np.random.randint(1_000_000), 
                                                       flip_percentage=0,
                                                       config=config), distribution_dict_row_array_valid, parallel_iterations=1)
                
                random_evaluation_dataset_array_valid = random_evaluation_dataset_list_valid_tf.numpy()
                print(random_evaluation_dataset_array_valid.shape)
                random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))
                print(random_evaluation_dataset_flat_array_valid.shape)                
                #tf.config.run_functions_eagerly(True)
            elif True:
                
                parallel_data_generation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
                random_evaluation_dataset_list_train_tf = parallel_data_generation(delayed(generate_dataset_from_distributions_line_tf)(number_of_variables_function=config['data']['number_of_variables'], 
                                                       number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                       max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                       seed_function = np.random.randint(1_000_000), 
                                                       flip_percentage=0,
                                                       config=config,
                                                       distribution_line_function=distribution_dict_row_array) for distribution_dict_row_array in distribution_dict_row_array_train)           
                                                       #distribution_line_function=distribution_dict_row_array) for distribution_dict_row_array in distribution_dict_row_array_train_unstacked))                 

                random_evaluation_dataset_array_train = np.array(random_evaluation_dataset_list_train_tf)
                print('random_evaluation_dataset_array_train.shape', random_evaluation_dataset_array_train.shape)
                random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))
                print('random_evaluation_dataset_flat_array_train.shape', random_evaluation_dataset_flat_array_train.shape)
                                                                     
                parallel_data_generation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
                random_evaluation_dataset_list_valid_tf = parallel_data_generation(delayed(generate_dataset_from_distributions_line_tf)(number_of_variables_function=config['data']['number_of_variables'], 
                                                       number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                       max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                       seed_function = np.random.randint(1_000_000), 
                                                       flip_percentage=0,
                                                       config=config,
                                                       distribution_line_function=distribution_dict_row_array) for distribution_dict_row_array in distribution_dict_row_array_valid)           
                                                       #distribution_line_function=distribution_dict_row_array) for distribution_dict_row_array in distribution_dict_row_array_train_unstacked))                 

                random_evaluation_dataset_array_valid =  np.array(random_evaluation_dataset_list_valid_tf)
                print('random_evaluation_dataset_array_valid.shape', random_evaluation_dataset_array_valid.shape)
                random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))
                print('random_evaluation_dataset_flat_array_valid.shape', random_evaluation_dataset_flat_array_valid.shape)                    
            else:
                print('distribution_dict_row_array_train.shape', distribution_dict_row_array_valid.shape)
                random_evaluation_dataset_flat_list_train = []
                for distribution_line in distribution_dict_row_array_train:
                    distribution_name_list, (distribution_parameters_0_param_1_list, 
                                              distribution_parameters_0_param_2_list, 
                                              distribution_parameters_1_param_1_list, 
                                              distribution_parameters_1_param_2_list) = line_to_distribution_structured_tf(distribution_line, config)

                    random_evaluation_dataset = generate_dataset_from_distributions_tf(distribution_name_list_function=distribution_name_list, 
                                                                                       distribution_parameters_0_param_1_list_function = distribution_parameters_0_param_1_list,
                                                                                       distribution_parameters_0_param_2_list_function = distribution_parameters_0_param_2_list,
                                                                                       distribution_parameters_1_param_1_list_function = distribution_parameters_1_param_1_list,
                                                                                       distribution_parameters_1_param_2_list_function = distribution_parameters_1_param_2_list,
                                                                                        number_of_variables_function=config['data']['number_of_variables'], 
                                                                                        number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                                                        max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                                                        seed_function = np.random.randint(1_000_000), 
                                                                                        flip_percentage=0, 
                                                                                        )

                    random_evaluation_dataset_flat = random_evaluation_dataset.numpy().ravel()
                    random_evaluation_dataset_flat_list_train.append(random_evaluation_dataset_flat)

                random_evaluation_dataset_flat_array_train = np.array(random_evaluation_dataset_flat_list_train)
                print('random_evaluation_dataset_flat_array_train.shape', random_evaluation_dataset_flat_array_train.shape)

            

                random_evaluation_dataset_flat_list_valid = []
                for distribution_line in distribution_dict_row_array_valid:

                    distribution_name_list, (distribution_parameters_0_param_1_list, 
                                              distribution_parameters_0_param_2_list, 
                                              distribution_parameters_1_param_1_list, 
                                              distribution_parameters_1_param_2_list) = line_to_distribution_structured_tf(distribution_line, config)

                    random_evaluation_dataset = generate_dataset_from_distributions_tf(distribution_name_list_function=distribution_name_list, 
                                                                                       distribution_parameters_0_param_1_list_function = distribution_parameters_0_param_1_list,
                                                                                       distribution_parameters_0_param_2_list_function = distribution_parameters_0_param_2_list,
                                                                                       distribution_parameters_1_param_1_list_function = distribution_parameters_1_param_1_list,
                                                                                       distribution_parameters_1_param_2_list_function = distribution_parameters_1_param_2_list,
                                                                                        number_of_variables_function=config['data']['number_of_variables'], 
                                                                                        number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                                                                        max_distributions_per_class_function = config['data']['max_distributions_per_class'], 
                                                                                        seed_function = np.random.randint(1_000_000), 
                                                                                        flip_percentage=0, 
                                                                                        )

                    random_evaluation_dataset_flat = random_evaluation_dataset.numpy().ravel()
                    random_evaluation_dataset_flat_list_valid.append(random_evaluation_dataset_flat)

                random_evaluation_dataset_flat_array_valid = np.array(random_evaluation_dataset_flat_list_valid)            
                print('random_evaluation_dataset_flat_array_valid.shape', random_evaluation_dataset_flat_array_valid.shape)
            
            if config['i_net']['data_reshape_version'] is not None:
                y_train_model = np.hstack((y_train, X_train_flat, random_evaluation_dataset_flat_array_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat, random_evaluation_dataset_flat_array_valid)))   
            else:
                y_train_model = np.hstack((y_train, X_train, random_evaluation_dataset_flat_array_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid, random_evaluation_dataset_flat_array_valid)))             
        
    else:
        if config['i_net']['data_reshape_version'] is not None:
            y_train_model = np.hstack((y_train, X_train_flat))   
            valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
        else:
            y_train_model = np.hstack((y_train, X_train))   
            valid_data = (X_valid, np.hstack((y_valid, X_valid)))                   
    #loss_function = inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config)
    #metrics = [inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, 'binary_crossentropy'), inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, 'mae'), inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, 'binary_accuracy')]
        
    ############################## BUILD MODEL ###############################
    if not config['computation']['load_model']:
        if config['i_net']['nas']:
            from tensorflow.keras.utils import CustomObjectScope
            
            #loss_function = inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config)
            #metrics = []
            
            custom_object_dict = {}
            loss_function_name = loss_function.__name__
            custom_object_dict[loss_function_name] = loss_function
            metric_names = []
            for metric in metrics:
                metric_name = metric.__name__
                metric_names.append(metric_name)
                custom_object_dict[metric_name] = metric  

            #print(custom_object_dict)    
            #print(metric_names)
            #print(loss_function_name)

            #CustomDenseInet(neurons, activation)
            #config['i_net']['function_representation_type']
            #config['function_family']['dt_type']          
                                                           
            
            with CustomObjectScope(custom_object_dict):
                if config['i_net']['nas_type'] == 'SEQUENTIAL':
                    input_node = ak.Input()
                    hidden_node = DeepDenseLayerBlock()(input_node)   
                    
                elif config['i_net']['nas_type'] == 'CNN': 
                    input_node = ak.Input()
                    hidden_node = ak.ConvBlock()(input_node)
                    hidden_node = ak.DenseBlock()(hidden_node)
                    
                elif config['i_net']['nas_type'] == 'LSTM':
                    input_node = ak.Input()
                    hidden_node = ak.RNNBlock()(input_node)
                    hidden_node = ak.DenseBlock()(hidden_node)

                elif config['i_net']['nas_type'] == 'CNN-LSTM': 
                    input_node = ak.Input()
                    hidden_node = ak.ConvBlock()(input_node)
                    hidden_node = ak.RNNBlock()(hidden_node)
                    hidden_node = ak.DenseBlock()(hidden_node)

                elif config['i_net']['nas_type'] == 'CNN-LSTM-parallel':                         
                    input_node = ak.Input()
                    hidden_node1 = ak.ConvBlock()(input_node)
                    hidden_node2 = ak.RNNBlock()(input_node)
                    hidden_node = ak.Merge()([hidden_node1, hidden_node2])
                    hidden_node = ak.DenseBlock()(hidden_node)

                internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
                leaf_node_num_ = 2 ** config['function_family']['maximum_depth']                        
                    
                if config['i_net']['function_representation_type'] == 1:
                    if config['function_family']['dt_type'] == 'SDT':     
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['data']['number_of_variables'])(hidden_node_outputs_coeff)
                        else:
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['data']['number_of_variables'])(hidden_node)        
                        outputs_list = [outputs_coeff]
                        
                        
                    elif config['function_family']['dt_type'] == 'vanilla':                                  
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                              activation='sigmoid')(hidden_node_outputs_coeff)
                        else:                        
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                              activation='sigmoid')(hidden_node)   
                        outputs_list = [outputs_coeff]
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_index = SingleDenseLayerBlock()(hidden_node)
                            outputs_index = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                                  activation='linear', 
                                                                  name='outputs_index_')(hidden_node_outputs_index)
                        else:                              
                            outputs_index = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                                  activation='linear', 
                                                                  name='outputs_index_')(hidden_node)      
                        outputs_list.append(outputs_index)
                        
                elif config['i_net']['function_representation_type'] == 2:
                    if config['function_family']['dt_type'] == 'SDT':       
                        number_output_coefficients = internal_node_num_ * config['function_family']['decision_sparsity']
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients)(hidden_node_outputs_coeff)
                        else:
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients)(hidden_node)

                        outputs_list = [outputs_coeff]

                        for outputs_index in range(internal_node_num_):
                            for var_index in range(config['function_family']['decision_sparsity']):
                                if config['i_net']['additional_hidden']:
                                    hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], activation='softmax')(hidden_node_outputs_identifer)
                                else:
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], activation='softmax')(hidden_node)
                                outputs_list.append(outputs_identifer)    

                    
                    elif config['function_family']['dt_type'] == 'vanilla':  

                        number_output_coefficients = internal_node_num_ * config['function_family']['decision_sparsity']
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, activation='sigmoid')(hidden_node_outputs_coeff)
                        else:                                 
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, activation='sigmoid')(hidden_node)
                            
                        outputs_list = [outputs_coeff]
                        for outputs_index in range(internal_node_num_):
                            for var_index in range(config['function_family']['decision_sparsity']):
                                output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                                if config['i_net']['additional_hidden']:
                                    hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], activation='softmax')(hidden_node_outputs_identifer)
                                else:                                  
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], activation='softmax')(hidden_node)
                                outputs_list.append(outputs_identifer)    



                        
                elif config['i_net']['function_representation_type'] == 3:
                    if config['function_family']['dt_type'] == 'SDT':                              
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(internal_node_num_*config['data']['number_of_variables'])(hidden_node_outputs_coeff)
                        else: 
                            outputs_coeff = CustomDenseInet(internal_node_num_*config['data']['number_of_variables'])(hidden_node)
                        
                        outputs_list = [outputs_coeff]


                        for outputs_index in range(internal_node_num_):
                            if config['i_net']['additional_hidden']:
                                hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                      activation='softmax')(hidden_node_outputs_identifer)
                            else:                             
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                      activation='softmax')(hidden_node)
                            outputs_list.append(outputs_identifer)    

               
                    
                    
                    elif config['function_family']['dt_type'] == 'vanilla': 
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=internal_node_num_*config['data']['number_of_variables'], activation='sigmoid')(hidden_node_outputs_coeff)
                        else:                              
                            outputs_coeff = CustomDenseInet(neurons=internal_node_num_*config['data']['number_of_variables'], activation='sigmoid')(hidden_node)
                        outputs_list = [outputs_coeff]


                        for outputs_index in range(internal_node_num_):
                            if config['i_net']['additional_hidden']:
                                hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], activation='softmax')(hidden_node_outputs_identifer)
                            else:                               
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], activation='softmax')(hidden_node)
                            outputs_list.append(outputs_identifer)    
                
                if config['function_family']['dt_type'] == 'SDT':
                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_bias = SingleDenseLayerBlock()(hidden_node)
                        outputs_bias = CustomDenseInet(internal_node_num_)(hidden_node_outputs_bias)
                    else:    
                        outputs_bias = CustomDenseInet(internal_node_num_)(hidden_node)
                    outputs_list.append(outputs_bias)    

                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_leaf_nodes = SingleDenseLayerBlock()(hidden_node)
                        outputs_leaf_nodes = CustomDenseInet(leaf_node_num_ * config['data']['num_classes'])(hidden_node_outputs_leaf_nodes)
                    else:                            
                        outputs_leaf_nodes = CustomDenseInet(leaf_node_num_ * config['data']['num_classes'])(hidden_node)
                    outputs_list.append(outputs_leaf_nodes)     

                    output_node = CombinedOutputInet()(outputs_list)                         
                elif config['function_family']['dt_type'] == 'vanilla':  
                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_leaf_nodes = SingleDenseLayerBlock()(hidden_node)
                        outputs_leaf_nodes = CustomDenseInet(neurons=leaf_node_num_, activation='sigmoid')(hidden_node_outputs_leaf_nodes)
                    else:                         
                        outputs_leaf_nodes = CustomDenseInet(neurons=leaf_node_num_, activation='sigmoid')(hidden_node)
                    outputs_list.append(outputs_leaf_nodes)    
                    
                    output_node = CombinedOutputInet()(outputs_list)
                        
                timestr = time.strftime("%Y%m%d-%H%M%S")
                directory = './data/autokeras/' + paths_dict['path_identifier_lambda_net_data'] + '/' + paths_dict['path_identifier_lambda_net_data'] + dt_string + '/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_reshape' + str(config['i_net']['data_reshape_version']) + '_' + timestr
                #directory = './data/autokeras/' + paths_dict['path_identifier_lambda_net_data'] + dt_string + '/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_reshape' + str(config['i_net']['data_reshape_version']) + '_' + timestr
                    
                    
                

                auto_model = ak.AutoModel(inputs=input_node, 
                                    outputs=output_node,
                                    loss=loss_function_name,
                                    metrics=metric_names,
                                    objective='val_loss',
                                    overwrite=True,
                                    tuner='greedy',#'hyperband',#"bayesian",'greedy', 'random'
                                    max_trials=config['i_net']['nas_trials'],
                                    directory=directory,
                                    seed=config['computation']['RANDOM_SEED'])

                ############################## PREDICTION ###############################
                print('TRAIN DATAS SHAPE: ', X_train.shape)

                auto_model.fit(
                    x=X_train,
                    y=y_train_model,
                    validation_data=valid_data,
                    epochs=config['i_net']['epochs'],
                    batch_size=config['i_net']['batch_size'],
                    callbacks=return_callbacks_from_string('early_stopping'),
                    verbose=2,
                    )         

                history = auto_model.tuner.oracle.get_best_trials(min(config['i_net']['nas_trials'], 5))
                model = auto_model.export_model()
                
                model.save('./data/saved_models/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + dt_string , save_format='tf')         
                
        else: 
            inputs = Input(shape=X_train.shape[1], 
                           name='input')

            hidden = tf.keras.layers.Dense(config['i_net']['dense_layers'][0], 
                                           name='hidden1_' + str(config['i_net']['dense_layers'][0]))(inputs)
            hidden = tf.keras.layers.Activation(activation='relu', 
                                                name='activation1_' + 'relu')(hidden)

            if config['i_net']['dropout'][0] > 0:
                hidden = tf.keras.layers.Dropout(config['i_net']['dropout'][0], 
                                                 name='dropout1_' + str(config['i_net']['dropout'][0]))(hidden)

            for layer_index, neurons in enumerate(config['i_net']['dense_layers'][1:]):
                hidden = tf.keras.layers.Dense(neurons, 
                                               name='hidden' + str(layer_index+2) + '_' + str(neurons))(hidden)
                hidden = tf.keras.layers.Activation(activation='relu', 
                                                    name='activation'  + str(layer_index+2) + '_relu')(hidden)
                
                if config['i_net']['dropout'][layer_index+1] > 0:
                    hidden = tf.keras.layers.Dropout(config['i_net']['dropout'][layer_index+1], 
                                                     name='dropout' + str(layer_index+2) + '_' + str(config['i_net']['dropout'][layer_index+1]))(hidden)

            internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
            leaf_node_num_ = 2 ** config['function_family']['maximum_depth']                    
                    
            if config['i_net']['function_representation_type'] == 1:
                if config['function_family']['dt_type'] == 'SDT':
                    outputs_coeff_neurons = internal_node_num_ * config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2)(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons)(hidden_outputs_coeff)
                    else:
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons)(hidden)        
                    outputs_list = [outputs_coeff]
                        
                    
                elif config['function_family']['dt_type'] == 'vanilla':   
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, name='hidden_outputs_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              name='outputs_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                           
                    else:                    
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              name='outputs_coeff_' + str(outputs_coeff_neurons))(hidden)   
                    
                    outputs_list = [outputs_coeff]
                    
                    outputs_index_neurons = internal_node_num_ * config['function_family']['decision_sparsity']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_index = tf.keras.layers.Dense(outputs_index_neurons*2, name='hidden_outputs_index_' + str(outputs_index_neurons))(hidden)
                        outputs_index = tf.keras.layers.Dense(outputs_index_neurons, 
                                                              activation='linear', 
                                                              name='outputs_index_' + str(outputs_index_neurons))(hidden_outputs_index)                                
                    else:                          
                        outputs_index = tf.keras.layers.Dense(outputs_index_neurons, 
                                                              activation='linear', 
                                                              name='outputs_index_' + str(outputs_index_neurons))(hidden)      

                    outputs_list.append(outputs_index)
                    
            elif config['i_net']['function_representation_type'] == 2:
                if config['function_family']['dt_type'] == 'SDT':                        
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity'] 
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                                
                    else:                               
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)

                    outputs_list = [outputs_coeff]

                    for outputs_index in range(internal_node_num_):
                        for var_index in range(config['function_family']['decision_sparsity']):
                            output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                            outputs_identifer_neurons = config['data']['number_of_variables']
                            if config['i_net']['additional_hidden']:
                                hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, name='hidden_' + output_name)(hidden)
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          name=output_name)(hidden_outputs_identifer)                               
                            else:                                  
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          name=output_name)(hidden)
                            outputs_list.append(outputs_identifer)        
                    
                elif config['function_family']['dt_type'] == 'vanilla':                    
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity'] 
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                        
                    else:                        
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(internal_node_num_):
                        for var_index in range(config['function_family']['decision_sparsity']):
                            output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                            outputs_identifer_neurons = config['data']['number_of_variables']
                            if config['i_net']['additional_hidden']:
                                hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, name='hidden_' + output_name)(hidden)
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      name=output_name)(hidden_outputs_identifer)                       
                            else:                                 
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          name=output_name)(hidden)
                            outputs_list.append(outputs_identifer)    

                
            elif config['i_net']['function_representation_type'] == 3:                
                if config['function_family']['dt_type'] == 'SDT':
                    
                    outputs_coeff_neurons = internal_node_num_*config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                      
                    else:                          
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                    outputs_list = [outputs_coeff]
                    
                    
                    for outputs_index in range(internal_node_num_):
                        output_name = 'output_identifier_' + str(outputs_index+1)
                        outputs_identifer_neurons = config['data']['number_of_variables']
                        if config['i_net']['additional_hidden']:
                            hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, name='hidden_' + output_name)(hidden)
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      name=output_name)(hidden_outputs_identifer)                     
                        else:                          
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      name=output_name)(hidden)
                        outputs_list.append(outputs_identifer)    
              
                    
                elif config['function_family']['dt_type'] == 'vanilla':   
                    outputs_coeff_neurons = internal_node_num_*config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, name='hidden' + 'output_coeff_' + str(outputs_coeff_neurons))(hidden)    
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                          activation='sigmoid', 
                                                          name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                        
                    else:
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                    outputs_list = [outputs_coeff]
                    
                    
                    for outputs_index in range(internal_node_num_):
                        output_name = 'output_identifier_' + str(outputs_index+1)
                        outputs_identifer_neurons = config['data']['number_of_variables']
                        if config['i_net']['additional_hidden']:
                            hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, name='hidden' + output_name)(hidden)                        
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                  activation='softmax', 
                                                                  name=output_name)(hidden_outputs_identifer)
                        else:
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                  activation='softmax', 
                                                                  name=output_name)(hidden)                            
                        outputs_list.append(outputs_identifer)    

       
                          
            if config['function_family']['dt_type'] == 'SDT':
                outputs_bias_neurons = internal_node_num_
                if config['i_net']['additional_hidden']:
                    hidden_outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons*2, name='hidden_' + 'output_bias_' + str(outputs_bias_neurons))(hidden)    
                    outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons, 
                                                         #activation='tanh', 
                                                         name='output_bias_' + str(outputs_bias_neurons))(hidden_outputs_bias)
                else:
                    outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons, 
                                                         #activation='tanh', 
                                                         name='output_bias_' + str(outputs_bias_neurons))(hidden)
                outputs_list.append(outputs_bias)     

                outputs_leaf_nodes_neurons = leaf_node_num_ * config['data']['num_classes']
                if config['i_net']['additional_hidden']:
                    hidden_outputs_bias = tf.keras.layers.Dense(outputs_leaf_nodes_neurons*2, name='hidden_' + 'output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)    
                    outputs_bias = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                               #activation='tanh', 
                                                               name='output_leaf_nodes_' + str(outputs_leaf_nodes_neurons))(hidden_outputs_bias)
                else:                
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                               #activation='tanh', 
                                                               name='output_leaf_nodes_' + str(outputs_leaf_nodes_neurons))(hidden)
                outputs_list.append(outputs_leaf_nodes)     

                outputs = concatenate(outputs_list, name='output_combined')            
            elif config['function_family']['dt_type'] == 'vanilla':
                outputs_leaf_nodes_neurons = leaf_node_num_
                if config['i_net']['additional_hidden']:
                    hidden_outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons*2, name='hidden_' + 'output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)    
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                           activation='sigmoid', 
                                                           name='output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden_outputs_leaf_nodes)                    
                else:
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                           activation='sigmoid', 
                                                           name='output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)
                outputs_list.append(outputs_leaf_nodes)    

                outputs = concatenate(outputs_list, name='output_combined')        
                    

            model = Model(inputs=inputs, outputs=outputs)
            
            if config['i_net']['early_stopping']:
                callback_names.append('early_stopping')
            
            callbacks = return_callbacks_from_string(callback_names, config)            

            optimizer = tf.keras.optimizers.get(config['i_net']['optimizer'])
            optimizer.learning_rate = config['i_net']['learning_rate']


            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=metrics
                         )

            verbosity = 2 #if n_jobs ==1 else 0

            ############################## PREDICTION ###############################
            history = model.fit(X_train,
                      y_train_model,
                      epochs=config['i_net']['epochs'], 
                      batch_size=config['i_net']['batch_size'], 
                      validation_data=valid_data,
                      callbacks=callbacks,
                      verbose=verbosity)

            history = history.history
            
            
            
            model.save('./data/saved_models/' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version']), save_format='tf')            
            #model.save('./data/saved_models/'  + '_' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version']), save_format='tf')
                
    else:
        history = None
        
    return history, (X_valid, y_valid), (X_test, y_test), dill.dumps(loss_function), dill.dumps(metrics), encoder_model





def normalize_lambda_net(flat_weights, random_evaluation_dataset, base_model=None, config=None): 
        
    if base_model is None:
        base_model = generate_base_model()
    else:
        base_model = dill.loads(base_model)
        
    from utilities.LambdaNet import weights_to_model
                
    model = weights_to_model(flat_weights, config=config, base_model=base_model)
            
    model_preds_random_data = model.predict(random_evaluation_dataset)
    
    min_preds = model_preds_random_data.min()
    max_preds = model_preds_random_data.max()

    
    model_preds_random_data_normalized = (model_preds_random_data-min_preds)/(max_preds-min_preds)

    shaped_weights = model.get_weights()

    normalization_factor = (max_preds-min_preds)#0.01
    #print(normalization_factor)

    normalization_factor_per_layer = normalization_factor ** (1/(len(shaped_weights)/2))
    #print(normalization_factor_per_layer)

    numer_of_layers = int(len(shaped_weights)/2)
    #print(numer_of_layers)

    shaped_weights_normalized = []
    current_bias_normalization_factor = normalization_factor_per_layer
    current_bias_normalization_factor_reverse = normalization_factor_per_layer ** (len(shaped_weights)/2)
    
    for index, (weights, biases) in enumerate(pairwise(shaped_weights)):
        #print('current_bias_normalization_factor', current_bias_normalization_factor)
        #print('current_bias_normalization_factor_reverse', current_bias_normalization_factor_reverse)
        #print('normalization_factor_per_layer', normalization_factor_per_layer)          
        if index == numer_of_layers-1:
            weights = weights/normalization_factor_per_layer#weights * normalization_factor_per_layer
            biases = biases/current_bias_normalization_factor - min_preds/normalization_factor #biases * current_bias_normalization_factor            
        else:
            weights = weights/normalization_factor_per_layer#weights * normalization_factor_per_layer
            biases = biases/current_bias_normalization_factor#biases * current_bias_normalization_factor            

        #weights = (weights-min_preds/current_bias_normalization_factor_reverse)/normalization_factor_per_layer#weights * normalization_factor_per_layer
        #biases = (biases-min_preds/current_bias_normalization_factor_reverse)/normalization_factor_per_layer#biases * current_bias_normalization_factor
        shaped_weights_normalized.append(weights)
        shaped_weights_normalized.append(biases)

        current_bias_normalization_factor = current_bias_normalization_factor * normalization_factor_per_layer
        current_bias_normalization_factor_reverse = current_bias_normalization_factor_reverse / normalization_factor_per_layer  
    flat_weights_normalized = flatten_list(shaped_weights_normalized)
    
    return flat_weights_normalized, (min_preds, max_preds)
    

    




def calculate_all_function_values(lambda_net_dataset, polynomial_dict):
          
    n_jobs_parallel_fv = n_jobs
    backend='threading'

    if n_jobs_parallel_fv <= 5:
        n_jobs_parallel_fv = 10

    #backend='threading' 
    #backend='sequential' 

    with tf.device('/CPU:0'):        
        function_value_dict = {
            'lambda_preds': np.nan_to_num(lambda_net_dataset.make_prediction_on_test_data()),
            'target_polynomials': np.nan_to_num(lambda_net_dataset.return_target_poly_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),          
            'lstsq_lambda_pred_polynomials': np.nan_to_num(lambda_net_dataset.return_lstsq_lambda_pred_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),         
            'lstsq_target_polynomials': np.nan_to_num(lambda_net_dataset.return_lstsq_target_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),   
            'inet_polynomials': np.nan_to_num(parallel_fv_calculation_from_polynomial(polynomial_dict['inet_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)),      
        }


        try:
            print('metamodel_poly')
            variable_names = ['X' + str(i) for i in range(n)]
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_poly'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['metamodel_poly'] =  function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    

        try:
            print('metamodel_functions')
            variable_names = ['X' + str(i) for i in range(n)]
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['metamodel_functions'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    

        try:
            print('metamodel_functions_no_GD')
            variable_names = ['X' + str(i) for i in range(n)]
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metametamodel_functions_no_GD'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['metamodel_functions_no_GD'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    

        try:
            print('symbolic_regression_functions')
            variable_names = ['X' + str(i) for i in range(n)]
            #variable_names[0] = 'x'        
            function_values = parallel_fv_calculation_from_sympy(polynomial_dict['symbolic_regression_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend, variable_names=variable_names)
            function_value_dict['symbolic_regression_functions'] = function_values#np.nan_to_num(function_values)

            #print(function_values)

            #for function_value in function_values:
            #    if np.isnan(function_value).any() or np.isinf(function_value).any():
            #        print(function_value)

            #print(function_values[-2])

            #for function_value in function_value_dict['symbolic_regression_functions']:
            #    if np.isnan(function_value).any() or np.isinf(function_value).any():
            #        print(function_value)        
            #print(function_value_dict['symbolic_regression_functions'][-2])

        except KeyError as ke:
            print('Exit', KeyError)    
        try:
            print('per_network_polynomials')
            function_values = parallel_fv_calculation_from_polynomial(polynomial_dict['per_network_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)        
            function_value_dict['per_network_polynomials'] = function_values#np.nan_to_num(function_values)
        except KeyError as ke:
            print('Exit', KeyError)    


    return function_value_dict
    
def evaluate_all_predictions(function_value_dict, polynomial_dict):
    
    ############################## EVALUATION ###############################
    evaluation_key_list = []
    evaluation_scores_list = []
    evaluation_distrib_list = []
    for combination in itertools.combinations(function_value_dict.keys(), r=2):
        key_1 = combination[0]
        key_2 = combination[1]
        
        try:
            polynomials_1 = polynomial_dict[key_1]
            if type(polynomials_1[0]) != np.ndarray and type(polynomials_1[0]) != list:
                polynomials_1 = None            
        except KeyError:
            polynomials_1 = None
            
        try:
            polynomials_2 = polynomial_dict[key_2]
            if type(polynomials_2[0]) != np.ndarray and type(polynomials_2[0]) != list:
                polynomials_2 = None
        except KeyError:
            polynomials_2 = None
            

            
        function_values_1 = function_value_dict[key_1]
        function_values_2 = function_value_dict[key_2]
        
        
        evaluation_key = key_1 + '_VS_' + key_2
        print(evaluation_key)
        evaluation_key_list.append(evaluation_key)
                
        evaluation_scores, evaluation_distrib = evaluate_interpretation_net(polynomials_1, 
                                                                            polynomials_2, 
                                                                            function_values_1, 
                                                                            function_values_2)        
        evaluation_scores_list.append(evaluation_scores)
        evaluation_distrib_list.append(evaluation_distrib)
        
        
    scores_dict = pd.DataFrame(data=evaluation_scores_list,
                               index=evaluation_key_list)        
        
    
    mae_distrib_dict = pd.DataFrame(data=[evaluation_distrib['MAE'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
    
        
    r2_distrib_dict = pd.DataFrame(data=[evaluation_distrib['R2'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
 
    
    distrib_dicts = {'MAE': mae_distrib_dict, 
                     'R2': r2_distrib_dict}        
    
    
    return scores_dict, distrib_dicts


def per_network_poly_generation(lambda_net_dataset, optimization_type='scipy', backend='loky'): 
        
    printing = True if n_jobs == 1 else False
    
    #if use_gpu and False:
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers if use_gpu else ''
        #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        #os.environ['XLA_FLAGS'] =  '--xla_gpu_cuda_data_dir=/usr/lib/cuda-10.1'     
        #backend = 'sequential'
        #printing = True
        
    #backend = 'sequential'
    #printing = True       
    #per_network_optimization_dataset_size = 5

    if optimization_type=='tf':
        
        per_network_hyperparams = {
            'optimizer': tf.keras.optimizers.RMSprop,
            'lr': 0.02,
            'max_steps': 500,
            'early_stopping': 10,
            'restarts': 3,
            'per_network_dataset_size': per_network_optimization_dataset_size,
        }


        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 #'list_of_monomial_identifiers': list_of_monomial_identifiers,
                 'x_min': x_min,
                 'x_max': x_max,
                 'sparse_poly_representation_version': sparse_poly_representation_version,
                }

        with tf.device('/CPU:0'):

            parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)

            per_network_optimization_polynomials = parallel_per_network(delayed(per_network_poly_optimization_tf)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                  lambda_network_weights, 
                                                                                                                  list_of_monomial_identifiers, 
                                                                                                                  config,
                                                                                                                  optimizer = per_network_hyperparams['optimizer'],
                                                                                                                  lr = per_network_hyperparams['lr'], 
                                                                                                                  max_steps = per_network_hyperparams['max_steps'], 
                                                                                                                  early_stopping = per_network_hyperparams['early_stopping'], 
                                                                                                                  restarts = per_network_hyperparams['restarts'],
                                                                                                                  printing = printing,
                                                                                                                  return_error = True) for lambda_network_weights in lambda_network_weights_list)      

            del parallel_per_network

    elif optimization_type=='scipy':    

        per_network_hyperparams = {
            'optimizer':  'Powell',
            'jac': 'fprime',
            'max_steps': 500,
            'restarts': 3,
            'per_network_dataset_size': per_network_optimization_dataset_size,
        }

        
        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 'x_min': x_min,
                 'x_max': x_max,
                 'sparse_poly_representation_version': sparse_poly_representation_version,     
                'max_optimization_minutes': max_optimization_minutes,
                 }
        with tf.device('/CPU:0'):
            if False:
                result = per_network_poly_optimization_scipy(per_network_hyperparams['per_network_dataset_size'], 
                                                  lambda_network_weights_list[0], 
                                                  list_of_monomial_identifiers, 
                                                  config,
                                                  optimizer = per_network_hyperparams['optimizer'],
                                                  jac = per_network_hyperparams['jac'],
                                                  max_steps = per_network_hyperparams['max_steps'], 
                                                  restarts = per_network_hyperparams['restarts'],
                                                  printing = True,
                                                  return_error = True)
                print(result)        

            parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)

            result_list_per_network = parallel_per_network(delayed(per_network_poly_optimization_scipy)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                      lambda_network_weights, 
                                                                                                                      list_of_monomial_identifiers, 
                                                                                                                      config,
                                                                                                                      optimizer = per_network_hyperparams['optimizer'],
                                                                                                                      jac = per_network_hyperparams['jac'],
                                                                                                                      max_steps = per_network_hyperparams['max_steps'], 
                                                                                                                      restarts = per_network_hyperparams['restarts'],
                                                                                                                      printing = printing,
                                                                                                                      return_error = True) for lambda_network_weights in lambda_network_weights_list)      
            per_network_optimization_errors = [result[0] for result in result_list_per_network]
            per_network_optimization_polynomials = [result[1] for result in result_list_per_network]          

            del parallel_per_network    
    
    #if use_gpu:
        #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers    
        
    return per_network_optimization_polynomials


def restructure_network_parameters(shaped_network_parameters, config):
    
    if config['i_net']['data_reshape_version'] == 0: #one sequence for biases and one sequence for weights per layer (padded to maximum size)
        
        max_size = 0
        for weights in shaped_network_parameters:
            max_size = max(max_size, max(weights.shape)) 
        
        padded_network_parameters_list = []
        for layer_weights, biases in pairwise(shaped_network_parameters):
            padded_weights_list = []
            for weights in layer_weights:
                padded_weights = np.pad(weights, (int(np.floor((max_size-weights.shape[0])/2)), int(np.ceil((max_size-weights.shape[0])/2))), 'constant')
                padded_weights_list.append(padded_weights)
            padded_biases = np.pad(biases, (int(np.floor((max_size-biases.shape[0])/2)), int(np.ceil((max_size-biases.shape[0])/2))), 'constant')
            padded_network_parameters_list.append(padded_biases)
            padded_network_parameters_list.extend(padded_weights_list)   

        return padded_network_parameters_list
    
    elif config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2: #each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer    
    
        lambda_net_structure = flatten_list([config['data']['number_of_variables'], config['lambda_net']['lambda_network_layers'], 1 if config['data']['num_classes'] == 2 else None])                 
        number_of_paths = reduce(lambda x, y: x * y, lambda_net_structure)

        network_parameters_sequence_list = np.array([]).reshape(number_of_paths, 0)    
        for layer_index, (weights, biases) in zip(range(1, len(lambda_net_structure)), pairwise(shaped_network_parameters)):

            layer_neurons = lambda_net_structure[layer_index]    
            previous_layer_neurons = lambda_net_structure[layer_index-1]

            assert biases.shape[0] == layer_neurons
            assert weights.shape[0]*weights.shape[1] == previous_layer_neurons*layer_neurons

            bias_multiplier = number_of_paths//layer_neurons
            weight_multiplier = number_of_paths//(previous_layer_neurons * layer_neurons)

            extended_bias_list = []
            for bias in biases:
                extended_bias = np.tile(bias, (bias_multiplier,1))
                extended_bias_list.extend(extended_bias)


            extended_weights_list = []
            for weight in weights.flatten():
                extended_weights = np.tile(weight, (weight_multiplier,1))
                extended_weights_list.extend(extended_weights)      

            network_parameters_sequence = np.concatenate([extended_weights_list, extended_bias_list], axis=1)
            network_parameters_sequence_list = np.hstack([network_parameters_sequence_list, network_parameters_sequence])


        number_of_paths = network_parameters_sequence_list.shape[0]
        number_of_unique_paths = np.unique(network_parameters_sequence_list, axis=0).shape[0]
        number_of_nonUnique_paths = number_of_paths-number_of_unique_paths

        if number_of_nonUnique_paths > 0:
            pass
            #print("Number of non-unique rows: " + str(number_of_nonUnique_paths))
            #print(network_parameters_sequence_list)     
            
        return network_parameters_sequence_list
    
    return None
    
    
def autoencode_data(X_data, config, encoder_model=None):
    
    X_data_flat = X_data
    
    class AutoEncoders(Model):

        def __init__(self, num_features, reduction_size):

            super().__init__()
            self.encoder = Sequential(
                [
                  Dense(num_features//2, activation="relu"),
                  Dense(reduction_size*2, activation="relu"),
                  Dense(reduction_size, activation="relu", name='sequential')
                ]
            )

            self.decoder = Sequential(
                [
                  Dense(reduction_size*2, activation="relu"),
                  Dense(num_features//2, activation="relu"),
                  Dense(num_features, activation="linear")
                ]
            )

        def call(self, inputs):

            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded
    
    if encoder_model is None:
        
        encoder_model = AutoEncoders(num_features=X_data.shape[1], reduction_size=5*config['data']['number_of_variables']*config['function_family']['maximum_depth'])

        encoder_model.compile(
            loss='mae',
            metrics=['mae'],
            optimizer='adam'
        )

        history = encoder_model.fit(
            X_data[100:], 
            X_data[100:], 
            epochs=250,
            batch_size=256, 
            validation_data=(X_data[:100], X_data[:100]),
            callbacks=return_callbacks_from_string('early_stopping'),
            verbose=2)
    
    encoder_layer = encoder_model.encoder#auto_encoder.get_layer('sequential')
    X_data = encoder_layer.predict(X_data)    
    
    return X_data, X_data_flat, encoder_model
    
def restructure_data_cnn_lstm(X_data, config, subsequences=None):
    import multiprocessing
    import psutil
    #version == 0: one sequence for biases and one sequence for weights per layer (padded to maximum size)
    #version == 1: each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer (no. columns == number of paths and no. rows = number of layers/length of path)
    #version == 2:each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer + transpose matrices  (no. columns == number of layers/length of path and no. rows = number of paths )
    
    base_model = generate_base_model(config)
       
    X_data_flat = X_data

    
    shaped_weights_list = []
    for data in tqdm(X_data):
        shaped_weights = shape_flat_network_parameters(data, base_model.get_weights())
        shaped_weights_list.append(shaped_weights)

    max_size = 0
    for weights in shaped_weights:
        max_size = max(max_size, max(weights.shape))      
        

    cores = multiprocessing.cpu_count()
        
    n_jobs = config['computation']['n_jobs']
    if n_jobs < 0:
        n_jobs = cores + n_jobs
    cpu_usage = psutil.cpu_percent() / 100
    n_jobs = max(int((1-cpu_usage) * n_jobs), 1)

    parallel_restructure_weights = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')
    
    X_data_list = parallel_restructure_weights(delayed(restructure_network_parameters)(shaped_weight, config=config) for shaped_weight in shaped_weights_list)      
    X_data = np.array(X_data_list)          
    del parallel_restructure_weights    
        
    if config['i_net']['data_reshape_version'] == 2: #transpose matrices (if false, no. columns == number of paths and no. rows = number of layers/length of path)
        X_data = np.transpose(X_data, (0, 2, 1))

    if config['i_net']['lstm_layers'] != None and config['i_net']['cnn_layers'] != None: #generate subsequences for cnn-lstm
        subsequences = 1 #for each bias+weights
        timesteps = X_train.shape[1]//subsequences

        X_data = X_data.reshape((X_data.shape[0], subsequences, timesteps, X_data.shape[2]))

    return X_data, X_data_flat

    


#######################################################################################################################################################
################################################################SAVING AND PLOTTING RESULTS############################################################
#######################################################################################################################################################    
    
    
def generate_history_plots(history, config):
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    plt.plot(history[list(history.keys())[1]])
    plt.plot(history[list(history.keys())[len(history.keys())//2+1]])
    plt.title('model ' + list(history.keys())[len(history.keys())//2+1])
    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net'] + '/' + list(history.keys())[len(history.keys())//2+1] + '.png')
    plt.clf()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net'] + '/loss_' + '.png')   
    
    plt.clf() 
            
            
def save_results(history, config):
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    path = './data/results/' + paths_dict['path_identifier_interpretation_net'] + '/history' + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(history, f, protocol=2)   


def plot_and_save_single_polynomial_prediction_evaluation(lambda_net_test_dataset_list, function_values_test_list, polynomial_dict_test_list, rand_index=1, plot_type=2):
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    x_vars = ['x' + str(i) for i in range(1, n+1)]

    columns = x_vars.copy()
    columns.append('FVs')

    columns_single = x_vars.copy()

    vars_plot = lambda_net_test_dataset_list[-1].X_test_data_list[rand_index]    
    
    custom_representation_keys_fixed = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials']
    custom_representation_keys_dynamic = ['inet_polynomials', 'per_network_polynomials']
    sympy_representation_keys = ['metamodel_poly', 'metamodel_functions', 'metamodel_functions_no_GD', 'symbolic_regression_functions']
    
    #keys = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials', 'inet_polynomials', 'per_network_polynomials', 'metamodel_functions']
    
    lambda_train_data = lambda_net_test_dataset_list[-1].y_test_data_list[rand_index].ravel()
    lambda_train_data_size = lambda_train_data.shape[0]
    lambda_train_data_str = np.array(['Lambda Train Data' for i in range(lambda_train_data_size)])  
    columns_single.append('Lambda Train Data')
    
    lambda_model_preds = function_values_test_list[-1]['lambda_preds'][rand_index].ravel()
    eval_size_plot = lambda_model_preds.shape[0]
    lambda_model_preds_str = np.array(['Lambda Model Preds' for i in range(eval_size_plot)])
    columns_single.append('Lambda Model Preds')
    
    identifier_list =[lambda_train_data_str, lambda_model_preds_str]
    plot_data_single_list = [vars_plot, lambda_train_data, lambda_model_preds]
    for key in custom_representation_keys_fixed:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue            
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, force_complete_poly_representation=True, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
    
    for key in custom_representation_keys_dynamic:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue        
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
        
    for key in sympy_representation_keys:
        try:
            function_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue
        function_by_key_string = str(function_by_key)
        function_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(function_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))        
    
    identifier = np.concatenate(identifier_list)
    plot_data_single = pd.DataFrame(data=np.column_stack(plot_data_single_list), columns=columns_single)
    vars_plot_all_preds = np.vstack([vars_plot for i in range(len(columns_single[n:]))])
    preds_plot_all = np.vstack(plot_data_single_list[1:]).ravel()         
        
    plot_data = pd.DataFrame(data=np.column_stack([vars_plot_all_preds, preds_plot_all]), columns=columns)
    plot_data['Identifier'] = identifier       
     
    
    location = './data/plotting/'
    folder = paths_dict['path_identifier_interpretation_net'] + '/'
        
    if plot_type == 1:
        
        
        pp = sns.pairplot(data=plot_data,
                      #kind='reg',
                      hue='Identifier',
                      y_vars=['FVs'],
                      x_vars=x_vars, 
                      height=7.5,
                      aspect=2)
        file = 'pp3in1_' + str(rand_index) + '.pdf'                 
        
    elif plot_type == 2:

        pp = sns.pairplot(data=plot_data,
                          #kind='reg',
                          hue='Identifier',
                          #y_vars=['FVs'],
                          #x_vars=x_vars, 
                          height=10//n)
             
        file = 'pp3in1_extended_' + str(rand_index) + '.pdf'  
        
    elif plot_type == 3:
        
        pp = sns.pairplot(data=plot_data_single,
                          #kind='reg',
                          y_vars=columns_single[n:],
                          x_vars=x_vars, 
                          height=3,
                          aspect=3)

        file = 'pp1_' + str(rand_index) + '.pdf'                   
        
    path = location + folder + file
    pp.savefig(path, format='pdf')
    plt.show()    
    
    if False:
        real_poly_VS_lstsq_target_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_target_poly)
        real_poly_VS_lstsq_target_poly_r2 = r2_score(real_poly_fvs, lstsq_target_poly)        

        real_poly_VS_inet_poly_mae = mean_absolute_error(real_poly_fvs, inet_poly_fvs)
        real_poly_VS_inet_poly_r2 = r2_score(real_poly_fvs, inet_poly_fvs)    

        real_poly_VS_perNet_poly_mae = mean_absolute_error(real_poly_fvs, per_network_opt_poly_fvs)
        real_poly_VS_perNet_poly_r2 = r2_score(real_poly_fvs, per_network_opt_poly_fvs)    

        real_poly_VS_lambda_model_preds_mae = mean_absolute_error(real_poly_fvs, lambda_model_preds)
        real_poly_VS_lambda_model_preds_r2 = r2_score(real_poly_fvs, lambda_model_preds)

        real_poly_VS_lstsq_lambda_preds_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_lambda_preds_poly)
        real_poly_VS_lstsq_lambda_preds_poly_r2 = r2_score(real_poly_fvs, lstsq_lambda_preds_poly)   

        from prettytable import PrettyTable

        tab = PrettyTable()

        tab.field_names = ["Comparison",  "MAE", "R2-Score", "Poly 1", "Poly 2"]
        tab._max_width = {"Poly 1" : 50, "Poly 2" : 50}

        tab.add_row(["Target Poly \n vs. \n LSTSQ Target Poly \n", real_poly_VS_lstsq_target_poly_mae, real_poly_VS_lstsq_target_poly_r2, polynomial_target_string, polynomial_lstsq_target_string])
        tab.add_row(["Target Poly \n vs. \n I-Net Poly \n", real_poly_VS_inet_poly_mae, real_poly_VS_inet_poly_r2, polynomial_target_string, polynomial_inet_string])
        tab.add_row(["Target Poly \n vs. \n Per Network Opt Poly \n", real_poly_VS_perNet_poly_mae, real_poly_VS_perNet_poly_r2, polynomial_target_string, polynomial_per_network_opt_string])
        tab.add_row(["Target Poly \n vs. \n Lambda Preds \n", real_poly_VS_lambda_model_preds_mae, real_poly_VS_lambda_model_preds_r2, polynomial_target_string, '-'])
        tab.add_row(["Target Poly \n vs. \n LSTSQ Lambda Preds Poly \n", real_poly_VS_lstsq_lambda_preds_poly_mae, real_poly_VS_lstsq_lambda_preds_poly_r2, polynomial_target_string, polynomial_lstsq_lambda_string])

        print(tab)

            

            