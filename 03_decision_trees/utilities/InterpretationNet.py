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
from tensorflow.keras.layers import Dense, Dropout, Lambda, Activation
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

from autokeras.keras_layers import CastToFloat32     

from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module
from typing import Optional
    
#######################################################################################################################################################
######################################################################AUTOKERAS BLOCKS#################################################################
#######################################################################################################################################################


class InputInet(node_module.Node, io_hypermodel.IOHyperModel):
    """Input node for tensor data.
    The data should be numpy.ndarray or tf.data.Dataset.
    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build_node(self, hp):
        return tf.keras.Input(shape=self.shape)#tf.keras.Input(shape=self.shape, dtype=self.dtype)

    def build(self, hp, inputs=None):
        #input_node = nest.flatten(inputs)[0]
        return inputs#keras_layers.CastToFloat32()(input_node)

    def get_adapter(self):
        return adapters.InputAdapter()

    def get_analyser(self):
        return analysers.InputAnalyser()

    def get_block(self):
        return blocks.GeneralBlock()

    def get_hyper_preprocessors(self):
        return []

def cast_to_float32(tensor):
    if tensor.dtype == tf.float32:
        return tensor
    if tensor.dtype == tf.string:
        return tf.strings.to_number(tensor, tf.float32)
    return tf.cast(tensor, tf.float32)

class CombinedOutputInet(ak.Head):

    def __init__(self, loss = None, metrics = None, output_dim=None, seed=42, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.seed = seed
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

    def __init__(self, loss = None, metrics = None, output_dim=None, seed=42, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.seed = seed
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
    
    def __init__(self, neurons=None, activation=None, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.neurons = neurons
        self.activation = activation
        
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(units=self.neurons, 
                      activation=self.activation, 
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        #layer = Dense(1, activation='linear')
        output_node = layer(input_node)
        #output_node = tf.keras.layers.Activation(activation=self.activation)(output_node)
        return output_node    

class SingleDenseLayerBlock(ak.Block):
    
    def __init__(self, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.Dense(
            hp.Int("num_units", min_value=16, max_value=512, step=16, default=32), 
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
        )
        output_node = layer(input_node)
        return output_node
    
class DeepDenseLayerBlock(ak.Block):
    
    def __init__(self, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        
        tf.random.set_seed(self.seed)

        input_node = tf.nest.flatten(inputs)[0]
        
        num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1, default=3)
        #activation = hp.Choice("activation", values=['relu', 'sigmoid', 'tanh'], default='sigmoid')
        num_units_list = []
        dropout_list = []
        activation_list = []
        
        for i in range(5):
            num_units = hp.Int("num_units_" + str(i), min_value=64, max_value=2048, default=512) #, step=64
            dropout = hp.Choice("dropout_" + str(i), [0.0, 0.1, 0.3, 0.5], default=0.0)
            activation = hp.Choice("activation_" + str(i), values=['relu', 'sigmoid', 'tanh'], default='relu')
            num_units_list.append(num_units)
            dropout_list.append(dropout)
            activation_list.append(activation)
        
        for i in range(num_layers):
            if i == 0:
                hidden_node = tf.keras.layers.Dense(
                    units = num_units_list[i],
                    #activation = activation_list[i], 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
                )(input_node)
                hidden_node = tf.keras.layers.Activation(activation=activation_list[i])(hidden_node)
                if dropout_list[i] > 0:
                    hidden_node = tf.keras.layers.Dropout(
                        rate = dropout_list[i],
                        seed = self.seed
                    )(hidden_node)   
            else:
                hidden_node = tf.keras.layers.Dense(
                    units = num_units_list[i],
                    #activation =  activation_list[i], 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
                )(hidden_node)
                hidden_node = tf.keras.layers.Activation(activation=activation_list[i])(hidden_node)
                if dropout_list[i] > 0:
                    hidden_node = tf.keras.layers.Dropout(
                        rate = dropout_list[i],
                        seed = self.seed
                    )(hidden_node)               
        return hidden_node  


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)
  
    
    
class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', restore_best_weights=True, start_epoch = 50): 
        super(CustomStopper, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=0, mode=mode,  restore_best_weights=True)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
            
            
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
    if config['i_net']['data_reshape_version'] == 3: #autoencoder
        encoder_model = load_encoder_model(config)
    
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

def load_encoder_model(config): 
    
    from utilities.utility_functions import generate_paths
    
    dt_string =  ('_depth' + str(config['function_family']['maximum_depth']) +
              '_beta' + str(config['function_family']['beta']) +
              '_decisionSpars' +  str(config['function_family']['decision_sparsity']) + 
              '_' + str(config['function_family']['dt_type']))
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    if config['i_net']['nas']:
        path = './data/saved_models/' + 'encoder_model_' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + dt_string         
        #path = './data/saved_models/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + '_' + paths_dict['path_identifier_lambda_net_data'] + dt_string         
    else:
        path = './data/saved_models/' + 'encoder_model_' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version'])

        
        #path = './data/saved_models/' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version'])        
        
    encoder_model = tf.keras.models.load_model(path) # #, compile=False
    
    return encoder_model
    
def generate_inet_labels_learned(lambda_net, config, config_adjusted):
    
    from utilities.utility_functions import network_parameters_to_network, get_shaped_parameters_for_decision_tree, get_parameters_from_sklearn_decision_tree
    
    network = network_parameters_to_network(lambda_net.network_parameters, config)       
    X_train_lambda = lambda_net.X_train_lambda
    y_train_lambda = lambda_net.y_train_lambda

    y_train_lambda_pred_network = np.round(network.predict(X_train_lambda)).astype(np.int64)

    tree = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
    tree.fit(X_train_lambda, y_train_lambda_pred_network)

    splits, leaf_probabilities_single = get_shaped_parameters_for_decision_tree(get_parameters_from_sklearn_decision_tree(tree, config), config_adjusted)
    splits = splits.numpy()
    leaf_probabilities_single = leaf_probabilities_single.numpy()
    #print(splits, leaf_probabilities)
    splits_features = tfa.seq2seq.hardmax(splits).numpy()#np.argmax(splits[0], axis=)
    #print(splits_features)
    splits_values = splits
    #print(splits)
    leaf_probabilities = np.vstack([leaf_probabilities_single, 1-leaf_probabilities_single]).T
    #print(leaf_probabilities)

    parameters_decision_tree = np.hstack([splits_values.ravel(), splits_features.ravel(), leaf_probabilities.ravel()]) # np.hstack([splits_values.ravel(), leaf_probabilities.ravel()])#
    #print(parameters_decision_tree)     
    
    return parameters_decision_tree
    
    
        
def generate_inet_train_data(lambda_net_dataset, config, encoder_model=None):
    #X_data = None
    X_data_flat = None
    y_data = None
    normalization_parameter_dict = None
    
    X_data = lambda_net_dataset.network_parameters_array
    

            
                    
    if not config['i_net']['optimize_decision_function']: #target polynomial as inet target
        y_data = lambda_net_dataset.target_function_parameters_array
    else:
        if not config['i_net']['function_value_loss']:

            
            if False:
                y_train_learned = []
                #for network_parameters in tqdm(X_train_flat):
                for lambda_net in tqdm(lambda_net_dataset.lambda_net_list):
                    network = network_parameters_to_network(lambda_net.network_parameters, config)       
                    X_train_lambda = lambda_net.X_train_lambda
                    y_train_lambda = lambda_net.y_train_lambda

                    y_train_lambda_pred_network = np.round(network.predict(X_train_lambda)).astype(np.int64)

                    tree = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
                    tree.fit(X_train_lambda, y_train_lambda_pred_network)

                    config_adjusted = deepcopy(config)
                    config_adjusted['i_net']['function_representation_type'] = 1
                    config_adjusted['function_family']['dt_type'] = 'vanilla'
                    config_adjusted['function_family']['decision_sparsity'] = 1                    
                    splits, leaf_probabilities_single = get_shaped_parameters_for_decision_tree(get_parameters_from_sklearn_decision_tree(tree, config), config_adjusted)
                    splits = splits.numpy()
                    leaf_probabilities_single = leaf_probabilities_single.numpy()
                    #print(splits, leaf_probabilities)
                    splits_features = tfa.seq2seq.hardmax(splits).numpy()#np.argmax(splits[0], axis=)
                    #print(splits_features)
                    splits_values = splits
                    #print(splits)
                    leaf_probabilities = np.vstack([leaf_probabilities_single, 1-leaf_probabilities_single]).T
                    #print(leaf_probabilities)

                    parameters_decision_tree = np.hstack([splits_values.ravel(), splits_features.ravel(), leaf_probabilities.ravel()]) # np.hstack([splits_values.ravel(), leaf_probabilities.ravel()])#
                    #print(parameters_decision_tree)                
                    y_train_learned.append(parameters_decision_tree)
            else:
                config_adjusted = deepcopy(config)
                config_adjusted['i_net']['function_representation_type'] = 1
                config_adjusted['function_family']['dt_type'] = 'vanilla'
                config_adjusted['function_family']['decision_sparsity'] = 1
                    
                parallel_data_generation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
                y_train_learned = parallel_data_generation(delayed(generate_inet_labels_learned)(lambda_net, config, config_adjusted) for lambda_net in lambda_net_dataset.lambda_net_list)                

            y_data = np.vstack(y_train_learned).astype(np.float64)
            print('DTs Trained')
            print('Example:', y_data[0])
        else:        
            y_data = np.zeros_like(lambda_net_dataset.target_function_parameters_array)
        
    if config['i_net']['data_reshape_version'] == 0 or config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2:
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
    from keras.utils.generic_utils import get_custom_objects
    from keras import backend as K
    
    def sigmoid_squeeze(x, factor=2):
        x = 1/(1+K.exp(-factor*x))
        return x                        

    def sigmoid_relaxed(x, factor=2):
        x = 1/(1+K.exp(-x/factor))
        return x      


    get_custom_objects().update({'sigmoid_squeeze': Activation(sigmoid_squeeze)})     
    get_custom_objects().update({'sigmoid_relaxed': Activation(sigmoid_relaxed)})     

    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    dt_string =  ('_depth' + str(config['function_family']['maximum_depth']) +
              '_beta' + str(config['function_family']['beta']) +
              '_decisionSpars' +  str(config['function_family']['decision_sparsity']) + 
              '_' + str(config['function_family']['dt_type']))
    
    ############################## DATA PREPARATION ###############################
    
    random_model = generate_base_model(config)#generate_base_model(config, disable_batchnorm=True)
    np.random.seed(config['computation']['RANDOM_SEED'])
        
    random_network_parameters = random_model.get_weights()
    network_parameters_structure = [network_parameter.shape for network_parameter in random_network_parameters]     
    
    print('network_parameters_structure', network_parameters_structure)

    (X_train, X_train_flat, y_train, encoder_model) = generate_inet_train_data(lambda_net_train_dataset, config)
    (X_valid, X_valid_flat, y_valid, _) = generate_inet_train_data(lambda_net_valid_dataset, config, encoder_model)
    if lambda_net_test_dataset is not None:
        (X_test, X_test_flat, y_test, _) = generate_inet_train_data(lambda_net_test_dataset, config, encoder_model)
    else:
        X_test = None
        X_test_flat = None
        y_test = None         
            
    if config['i_net']['separate_weight_bias']:
        print('separate_weight_bias')
        lambda_network_layers_complete = flatten_list([config['data']['number_of_variables'], config['lambda_net']['lambda_network_layers']])
        
        bias_list = []
        weight_list = []
        
        
        if lambda_net_test_dataset is not None:
            data_list = [X_train, X_train_flat, X_valid, X_valid_flat, X_test, X_test_flat]
        else:
            data_list = [X_train, X_train_flat, X_valid, X_valid_flat]
            
        bias_count = 0
        weight_count = 0
        print('X_train.shape', X_train.shape)
        for X_data in data_list:
            if X_data is None:
                continue
            start_index = 0
            print('X_data.shape', X_data.shape)
            for i in range(1, len(lambda_network_layers_complete)):
                weight_number = lambda_network_layers_complete[i-1]*lambda_network_layers_complete[i]
                print('weight_number', weight_number)
                weight_list.append(X_data[:,start_index:weight_number])
                print('start_index', i, start_index)
                start_index += weight_number
                weight_count += weight_number
                bias_number = lambda_network_layers_complete[i]
                bias_list.append(X_data[:,start_index:bias_number])
                print('bias_number', bias_number)
                start_index += bias_number
                bias_count += bias_number
                print('start_index', i, start_index)
            X_data = np.hstack(flatten_list([weight_list, bias_list]))
            print(X_data.shape)
            print(bias_count, weight_count)
        print('X_train.shape', X_train.shape)
    ############################## OBJECTIVE SPECIFICATION AND LOSS FUNCTION ADJUSTMENTS ###############################
    metrics = []
    loss_function = None
    
    distribution_dict_list = lambda_net_train_dataset.distribution_dict_list_list
    distribution_dict_list.extend(lambda_net_valid_dataset.distribution_dict_list_list)    
        
    try:
        use_distribution_list = config['data']['use_distribution_list'] if config['data']['max_distributions_per_class'] is not None else False
    except:
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
        if config['i_net']['function_representation_type'] >= 3:
            if config['i_net']['optimize_decision_function']:
                
                loss_function = inet_decision_function_fv_loss_wrapper_parameters(config)
                
                metrics.append(inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list))
                for metric in config['i_net']['metrics']:
                    metrics.append(inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, metric, use_distribution_list=use_distribution_list))    

                if False:
                    metrics.append(inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list))
                    #metrics.append(inet_target_function_fv_loss_wrapper(config))
                    for metric in config['i_net']['metrics']:
                        metrics.append(inet_decision_function_fv_metric_wrapper(random_model, network_parameters_structure, config, metric, use_distribution_list=use_distribution_list))  
                        #metrics.append(inet_target_function_fv_metric_wrapper(config, metric))                  
        else:
            raise SystemExit('Coefficient Loss not implemented for configuration')
        
        if False:
            metrics.append(inet_target_function_fv_loss_wrapper(config))
            metrics.append(inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list))
            if config['i_net']['optimize_decision_function']:
                raise SystemExit('Coefficient Loss not implemented for decision function optimization')            
            else:
                if config['i_net']['function_representation_type'] == 1:
                    loss_function = tf.keras.losses.get('mae') #inet_coefficient_loss_wrapper(inet_loss)
                else:
                    raise SystemExit('Coefficient Loss not implemented for selected function representation')
    
    #loss_function = inet_decision_function_fv_loss_wrapper_parameters(config)
    #loss_function = inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list)
    #metrics = [inet_decision_function_fv_loss_wrapper(random_model, network_parameters_structure, config, use_distribution_list=use_distribution_list)]
        
    distribution_dict_index_train = np.array([[i] for i in range(y_train.shape[0])])
    distribution_dict_index_valid = np.array([[len(distribution_dict_index_train) + i] for i in range(y_valid.shape[0])])
    
    #print('np.hstack((y_train, X_train_flat, distribution_dict_index_train))', np.hstack((y_train, X_train, distribution_dict_index_train)))
    #print('np.hstack((y_train, X_train_flat))', np.hstack((y_train, X_train)))
    
    
    if use_distribution_list and config['data']['max_distributions_per_class'] is not None:
        
        if True:
            idx_train = np.random.randint(lambda_net_train_dataset.X_train_lambda_array.shape[1], size=config['evaluation']['random_evaluation_dataset_size'])
            idx_valid = np.random.randint(lambda_net_valid_dataset.X_train_lambda_array.shape[1], size=config['evaluation']['random_evaluation_dataset_size'])

            random_evaluation_dataset_array_train = lambda_net_train_dataset.X_train_lambda_array[:,idx_train]
            random_evaluation_dataset_array_valid = lambda_net_valid_dataset.X_train_lambda_array[:,idx_valid]
                        
            random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, random_evaluation_dataset_array_train.shape[1]*config['data']['number_of_variables']))                        
            random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, random_evaluation_dataset_array_valid.shape[1]*config['data']['number_of_variables']))
        
        elif 'make_class' in config['data']['function_generation_type']:

            random_evaluation_dataset_array_train = lambda_net_train_dataset.X_test_lambda_array#[:, np.random.choice(lambda_net_train_dataset.shape[1], config['evaluation']['random_evaluation_dataset_size'], replace=False), :]#[:,:config['evaluation']['random_evaluation_dataset_size'],:]
            random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))  
            
            random_evaluation_dataset_array_valid = lambda_net_valid_dataset.X_test_lambda_array#[:, np.random.choice(lambda_net_valid_dataset.shape[1], config['evaluation']['random_evaluation_dataset_size'], replace=False), :]#[:,:config['evaluation']['random_evaluation_dataset_size'],:]         
            random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))   
            
                        
            random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, random_evaluation_dataset_array_train.shape[1]*config['data']['number_of_variables']))
            
            random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, random_evaluation_dataset_array_valid.shape[1]*config['data']['number_of_variables']))
            
            
        else:        

            max_distributions_per_class = deepcopy(config['data']['max_distributions_per_class'])
            if max_distributions_per_class == 0:
                max_distributions_per_class = 1
            #print(lambda_net_train_dataset.distribution_dict_row_array[0])
            parallel_data_generation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
            random_evaluation_dataset_list_train = parallel_data_generation(delayed(generate_dataset_from_distributions_line)(line_distribution_parameters=distribution_dict_row,
                                            number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                            max_distributions_per_class_function=max_distributions_per_class, 
                                            config=config,
                                            random_parameters_distribution=config['data']['random_parameters_distribution'],
                                            flip_percentage=config['data']['noise_injected_level'],
                                            data_noise=config['data']['data_noise'],
                                            seed_function=seed_function) for distribution_dict_row, seed_function in zip(lambda_net_train_dataset.distribution_dict_row_array, lambda_net_train_dataset.seed_list))



            random_evaluation_dataset_array_train = np.array(random_evaluation_dataset_list_train)

            random_evaluation_dataset_flat_array_train = random_evaluation_dataset_array_train.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))



            parallel_data_generation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
            random_evaluation_dataset_list_valid = parallel_data_generation(delayed(generate_dataset_from_distributions_line)(line_distribution_parameters=distribution_dict_row,
                                            number_of_samples_function=config['evaluation']['random_evaluation_dataset_size'], 
                                            max_distributions_per_class_function=max_distributions_per_class, 
                                            config=config,
                                            random_parameters_distribution=config['data']['random_parameters_distribution'],
                                            flip_percentage=config['data']['noise_injected_level'],
                                            data_noise=config['data']['data_noise'],
                                            seed_function=seed_function) for distribution_dict_row, seed_function in zip(lambda_net_valid_dataset.distribution_dict_row_array, lambda_net_valid_dataset.seed_list))



            random_evaluation_dataset_array_valid = np.array(random_evaluation_dataset_list_valid)

            random_evaluation_dataset_flat_array_valid = random_evaluation_dataset_array_valid.reshape((-1, config['evaluation']['random_evaluation_dataset_size']*config['data']['number_of_variables']))

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

            #config['i_net']['function_representation_type']
            #config['function_family']['dt_type']          
                                                           
            
            with CustomObjectScope(custom_object_dict):
                if config['i_net']['nas_type'] == 'SEQUENTIAL':
                    input_node = InputInet()#ak.Input()
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
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['data']['number_of_variables'], 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else:
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['data']['number_of_variables'], 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)        
                        outputs_list = [outputs_coeff]
                        
                        
                    elif config['function_family']['dt_type'] == 'vanilla':                                  
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock(seed=config['computation']['RANDOM_SEED'])(hidden_node)
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else:                        
                            outputs_coeff = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)   
                        outputs_list = [outputs_coeff]
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_index = SingleDenseLayerBlock()(hidden_node)
                            outputs_index = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                                  activation='linear', 
                                                                  seed=config['computation']['RANDOM_SEED'], 
                                                                  name='outputs_index_')(hidden_node_outputs_index)
                        else:                              
                            outputs_index = CustomDenseInet(internal_node_num_ * config['function_family']['decision_sparsity'], 
                                                                  activation='linear', 
                                                                  seed=config['computation']['RANDOM_SEED'], 
                                                                  name='outputs_index_')(hidden_node)      
                        outputs_list.append(outputs_index)
                        
                elif config['i_net']['function_representation_type'] == 2:
                    if config['function_family']['dt_type'] == 'SDT':       
                        number_output_coefficients = internal_node_num_ * config['function_family']['decision_sparsity']
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else:
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)

                        outputs_list = [outputs_coeff]

                        for outputs_index in range(internal_node_num_):
                            for var_index in range(config['function_family']['decision_sparsity']):
                                if config['i_net']['additional_hidden']:
                                    hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], 
                                                                        activation='softmax', 
                                                                        seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_identifer)
                                else:
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], 
                                                                        activation='softmax', 
                                                                        seed=config['computation']['RANDOM_SEED'])(hidden_node)
                                outputs_list.append(outputs_identifer)    

                    
                    elif config['function_family']['dt_type'] == 'vanilla':  

                        number_output_coefficients = internal_node_num_ * config['function_family']['decision_sparsity']
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else:                                 
                            outputs_coeff = CustomDenseInet(neurons=number_output_coefficients, 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)
                            
                        outputs_list = [outputs_coeff]
                        for outputs_index in range(internal_node_num_):
                            for var_index in range(config['function_family']['decision_sparsity']):
                                output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                                if config['i_net']['additional_hidden']:
                                    hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], 
                                                                        activation='softmax', 
                                                                        seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_identifer)
                                else:                                  
                                    outputs_identifer = CustomDenseInet(neurons=config['data']['number_of_variables'], 
                                                                        activation='softmax', 
                                                                        seed=config['computation']['RANDOM_SEED'])(hidden_node)
                                outputs_list.append(outputs_identifer)    



                        
                elif config['i_net']['function_representation_type'] == 3:
                    if config['function_family']['dt_type'] == 'SDT':                              
                        
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(internal_node_num_*config['data']['number_of_variables'], 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else: 
                            outputs_coeff = CustomDenseInet(internal_node_num_*config['data']['number_of_variables'], 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)
                        
                        outputs_list = [outputs_coeff]


                        for outputs_index in range(internal_node_num_):
                            if config['i_net']['additional_hidden']:
                                hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                    activation='softmax', 
                                                                    seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_identifer)
                            else:                             
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                    activation='softmax', 
                                                                    seed=config['computation']['RANDOM_SEED'])(hidden_node)
                            outputs_list.append(outputs_identifer)    

               
                    
                    
                    elif config['function_family']['dt_type'] == 'vanilla': 
                        if config['i_net']['additional_hidden']:
                            hidden_node_outputs_coeff = SingleDenseLayerBlock()(hidden_node)
                            outputs_coeff = CustomDenseInet(neurons=internal_node_num_*config['data']['number_of_variables'], 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_coeff)
                        else:                              
                            outputs_coeff = CustomDenseInet(neurons=internal_node_num_*config['data']['number_of_variables'], 
                                                            activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)
                        outputs_list = [outputs_coeff]


                        for outputs_index in range(internal_node_num_):
                            if config['i_net']['additional_hidden']:
                                hidden_node_outputs_identifer = SingleDenseLayerBlock()(hidden_node)
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                    activation='softmax', 
                                                                    seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_identifer)
                            else:                               
                                outputs_identifer = CustomDenseInet(config['data']['number_of_variables'], 
                                                                    activation='softmax', 
                                                                    seed=config['computation']['RANDOM_SEED'])(hidden_node)
                            outputs_list.append(outputs_identifer)    
                
                if config['function_family']['dt_type'] == 'SDT':
                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_bias = SingleDenseLayerBlock()(hidden_node)
                        outputs_bias = CustomDenseInet(internal_node_num_, 
                                                       seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_bias)
                    else:    
                        outputs_bias = CustomDenseInet(internal_node_num_, 
                                                       seed=config['computation']['RANDOM_SEED'])(hidden_node)
                    outputs_list.append(outputs_bias)    

                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_leaf_nodes = SingleDenseLayerBlock()(hidden_node)
                        outputs_leaf_nodes = CustomDenseInet(leaf_node_num_ * config['data']['num_classes'], 
                                                             seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_leaf_nodes)
                    else:                            
                        outputs_leaf_nodes = CustomDenseInet(leaf_node_num_ * config['data']['num_classes'], 
                                                             seed=config['computation']['RANDOM_SEED'])(hidden_node)
                    outputs_list.append(outputs_leaf_nodes)     

                    output_node = CombinedOutputInet()(outputs_list)                         
                elif config['function_family']['dt_type'] == 'vanilla':  
                    if config['i_net']['additional_hidden']:
                        hidden_node_outputs_leaf_nodes = SingleDenseLayerBlock()(hidden_node)
                        outputs_leaf_nodes = CustomDenseInet(neurons=leaf_node_num_, 
                                                             activation='sigmoid', 
                                                             seed=config['computation']['RANDOM_SEED'])(hidden_node_outputs_leaf_nodes)
                    else:                         
                        outputs_leaf_nodes = CustomDenseInet(neurons=leaf_node_num_, 
                                                             activation='sigmoid', 
                                                            seed=config['computation']['RANDOM_SEED'])(hidden_node)
                    outputs_list.append(outputs_leaf_nodes)    
                    
                    output_node = CombinedOutputInet()(outputs_list)
                        
                timestr = time.strftime("%Y%m%d-%H%M%S")
                directory = './data/autokeras/' + paths_dict['path_identifier_lambda_net_data'] + '/' + paths_dict['path_identifier_lambda_net_data'] + dt_string + '/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_reshape' + str(config['i_net']['data_reshape_version']) + '_' + timestr
                #directory = './data/autokeras/' + paths_dict['path_identifier_lambda_net_data'] + dt_string + '/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_reshape' + str(config['i_net']['data_reshape_version']) + '_' + timestr
                    
                    
                    
                def _compile_keras_model_adjusted(self, hp, model):
                    # Specify hyperparameters from compile(...)
                    optimizer_name = hp.Choice(
                        "optimizer",
                        ["adam", "rmsprop", "adadelta", "adagrad"],
                        default="adam",
                    )

                    learning_rate = hp.Choice(
                        "learning_rate", [1e-1, 1e-2, 1e-3, 1e-4, 2e-5, 1e-5], default=1e-3
                    )
                    
                    optimizer = tf.keras.optimizers.get(optimizer_name)
                    optimizer.learning_rate = learning_rate
                    
                    random.seed(config['computation']['RANDOM_SEED'])
                    np.random.seed(config['computation']['RANDOM_SEED'])  
                    tf.random.set_seed(config['computation']['RANDOM_SEED'])                      
                    model.compile(
                        optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss()#, jit_compile=True
                    )
                    
                    #print(model.get_weights())

                    return model
                    
                ak.graph.Graph._compile_keras_model = _compile_keras_model_adjusted           

                
                
                auto_model = ak.AutoModel(inputs=input_node, 
                                    outputs=output_node,
                                    loss=loss_function_name,
                                    metrics=metric_names,
                                    objective='val_loss',
                                    overwrite=True,
                                    tuner=config['i_net']['nas_optimizer'],#'hyperband',#"bayesian",'greedy', 'random'
                                    max_trials=config['i_net']['nas_trials'],
                                    directory=directory,
                                    seed=config['computation']['RANDOM_SEED'])

                ############################## PREDICTION ###############################
                print('TRAIN DATAS SHAPE: ', X_train.shape)

                #earlyStopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.01, verbose=0, mode='min', restore_best_weights=True)
                earlyStopping = CustomStopper(monitor='val_loss', patience=20, min_delta=0.01, verbose=0, mode='min', restore_best_weights=True, start_epoch = 25)
                
                
                random.seed(config['computation']['RANDOM_SEED'])
                np.random.seed(config['computation']['RANDOM_SEED'])  
                tf.random.set_seed(config['computation']['RANDOM_SEED'])           
                
                auto_model.fit(
                    x=X_train,
                    y=y_train_model,
                    validation_data=valid_data,
                    epochs=config['i_net']['epochs'],
                    batch_size=config['i_net']['batch_size'],
                    shuffle=False,
                    callbacks=[earlyStopping],
                    verbose=2,
                    workers=1, #10
                    use_multiprocessing=False, #True
                    )         

                history = auto_model.tuner.oracle.get_best_trials(min(config['i_net']['nas_trials'], 5))
                model = auto_model.export_model()
                
                model.save('./data/saved_models/' + config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + dt_string , save_format='tf')   
                
                if encoder_model is not None:
                    encoder_model.save('./data/saved_models/' + 'encoder_model_' +  config['i_net']['nas_type'] + '_' + str(config['i_net']['nas_trials']) + '_' + str(config['i_net']['data_reshape_version']) + dt_string , save_format='tf')


        else: 
            
            
            if not isinstance(config['i_net']['hidden_activation'], list):
                config['i_net']['hidden_activation'] = [config['i_net']['hidden_activation'] for _ in range(len(config['i_net']['dense_layers']))]

            tf.random.set_seed(config['computation']['RANDOM_SEED'])
                
            if config['i_net']['separate_weight_bias']:
                inputs = Input(shape=X_train.shape[1], 
                                   dtype=tf.float64,
                                   name='input')                    
                    
                #inputs = CastToFloat32()(inputs)
                    
                inputs_bias = crop(1, 0, bias_count)(inputs)
                inputs_weight = crop(1, bias_count, weight_count)(inputs)#crop(1, bias_count, bias_count+weight_count)(inputs)
                
                hidden_weight = tf.keras.layers.Dense(config['i_net']['dense_layers'][0], 
                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                      name='hidden1_weight_' + str(config['i_net']['dense_layers'][0]))(inputs_weight)
                hidden_weight = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][0],  
                                                           name='activation1_weight_' + config['i_net']['hidden_activation'][0])(hidden_weight)
                
                hidden_bias = tf.keras.layers.Dense(config['i_net']['dense_layers'][0], 
                                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                    name='hidden1_bias_' + str(config['i_net']['dense_layers'][0]))(inputs_bias)
                hidden_bias = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][0],  
                                                         name='activation1_bias_' + config['i_net']['hidden_activation'][0])(hidden_bias)
                
                
                if config['i_net']['dropout'][0] > 0:
                    hidden_weight = tf.keras.layers.Dropout(config['i_net']['dropout'][0], 
                                                            seed=config['computation']['RANDOM_SEED'],
                                                            name='dropout1_weight_' + str(config['i_net']['dropout'][0]))(hidden_weight)
                    
                    hidden_bias = tf.keras.layers.Dropout(config['i_net']['dropout'][0], 
                                                          seed=config['computation']['RANDOM_SEED'], 
                                                          name='dropout1_bias_' + str(config['i_net']['dropout'][0]))(hidden_bias)
                    
                for layer_index, neurons in enumerate(config['i_net']['dense_layers'][1:]):
                    hidden_weight = tf.keras.layers.Dense(neurons, 
                                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                          name='hidden_weight' + str(layer_index+2) + '_' + str(neurons))(hidden_weight)
                    hidden_weight = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][layer_index+1], 
                                                               name='activation_weight'  + str(layer_index+2) + '_' + config['i_net']['hidden_activation'][layer_index+1])(hidden_weight)

                    if config['i_net']['dropout'][layer_index+1] > 0:
                        hidden_weight = tf.keras.layers.Dropout(config['i_net']['dropout'][layer_index+1], 
                                                                seed=config['computation']['RANDOM_SEED'], 
                                                                name='dropout_weight' + str(layer_index+2) + '_' + str(config['i_net']['dropout'][layer_index+1]))(hidden_weight)

                        
                    hidden_bias = tf.keras.layers.Dense(neurons, 
                                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                        name='hidden_bias' + str(layer_index+2) + '_' + str(neurons))(hidden_bias)
                    hidden_bias = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][layer_index+1], 
                                                        name='activation_bias'  + str(layer_index+2) + '_' + config['i_net']['hidden_activation'][layer_index+1])(hidden_bias)

                    if config['i_net']['dropout'][layer_index+1] > 0:
                        hidden_bias = tf.keras.layers.Dropout(config['i_net']['dropout'][layer_index+1], 
                                                              seed=config['computation']['RANDOM_SEED'], 
                                                              name='dropout_bias' + str(layer_index+2) + '_' + str(config['i_net']['dropout'][layer_index+1]))(hidden_bias)

                        

                hidden = concatenate([hidden_bias, hidden_weight], name='hidden_combined')  
            else:
                inputs = Input(shape=X_train.shape[1], 
                                   name='input')                    

                hidden = tf.keras.layers.Dense(config['i_net']['dense_layers'][0], 
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                               name='hidden1_' + str(config['i_net']['dense_layers'][0]))(inputs)
                hidden = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][0],  
                                                    name='activation1_' + config['i_net']['hidden_activation'][0])(hidden)

                if config['i_net']['dropout'][0] > 0:
                    hidden = tf.keras.layers.Dropout(config['i_net']['dropout'][0], 
                                                     seed=config['computation']['RANDOM_SEED'], 
                                                     name='dropout1_' + str(config['i_net']['dropout'][0]))(hidden)

                for layer_index, neurons in enumerate(config['i_net']['dense_layers'][1:]):
                    hidden = tf.keras.layers.Dense(neurons, 
                                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                   name='hidden' + str(layer_index+2) + '_' + str(neurons))(hidden)
                    hidden = tf.keras.layers.Activation(activation=config['i_net']['hidden_activation'][layer_index+1], 
                                                        name='activation'  + str(layer_index+2) + '_' + config['i_net']['hidden_activation'][layer_index+1])(hidden)

                    if config['i_net']['dropout'][layer_index+1] > 0:
                        hidden = tf.keras.layers.Dropout(config['i_net']['dropout'][layer_index+1], 
                                                         seed=config['computation']['RANDOM_SEED'], 
                                                         name='dropout' + str(layer_index+2) + '_' + str(config['i_net']['dropout'][layer_index+1]))(hidden)

            internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
            leaf_node_num_ = 2 ** config['function_family']['maximum_depth']                    
                                
            if config['i_net']['function_representation_type'] == 1:
                if config['function_family']['dt_type'] == 'SDT':
                    outputs_coeff_neurons = internal_node_num_ * config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']))(hidden_outputs_coeff)
                    else:
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']))(hidden)        
                    outputs_list = [outputs_coeff]
                        
                    
                elif config['function_family']['dt_type'] == 'vanilla':   
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden_outputs_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='outputs_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                           
                    else:                    
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='outputs_coeff_' + str(outputs_coeff_neurons))(hidden)   
                    
                    outputs_list = [outputs_coeff]
                    
                    outputs_index_neurons = internal_node_num_ * config['function_family']['decision_sparsity']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_index = tf.keras.layers.Dense(outputs_index_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden_outputs_index_' + str(outputs_index_neurons))(hidden)
                        outputs_index = tf.keras.layers.Dense(outputs_index_neurons, 
                                                              activation='linear', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='outputs_index_' + str(outputs_index_neurons))(hidden_outputs_index)                                
                    else:                          
                        outputs_index = tf.keras.layers.Dense(outputs_index_neurons, 
                                                              activation='linear', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='outputs_index_' + str(outputs_index_neurons))(hidden)      

                    outputs_list.append(outputs_index)
                    
            elif config['i_net']['function_representation_type'] == 2:
                if config['function_family']['dt_type'] == 'SDT':                        
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity'] 
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                                
                    else:                               
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)

                    outputs_list = [outputs_coeff]

                    for outputs_index in range(internal_node_num_):
                        for var_index in range(config['function_family']['decision_sparsity']):
                            output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                            outputs_identifer_neurons = config['data']['number_of_variables']
                            if config['i_net']['additional_hidden']:
                                hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, 
                                                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                                 name='hidden_' + output_name)(hidden)
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                          name=output_name)(hidden_outputs_identifer)                               
                            else:                                  
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                          name=output_name)(hidden)
                            outputs_list.append(outputs_identifer)        
                    
                elif config['function_family']['dt_type'] == 'vanilla':                    
                    outputs_coeff_neurons = internal_node_num_ * config['function_family']['decision_sparsity'] 
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                        
                    else:
                        if True:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-3*x))
                                return x      
                            
                            def sigmoid_squeeze_inverse(x):
                                x = 1/(1+K.exp(3*x))
                                return x  
                            
                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/3))
                                return x      

                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze_inverse)})       
                            
                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                            
                        
                        else:
                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='sigmoid', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(internal_node_num_):
                        for var_index in range(config['function_family']['decision_sparsity']):
                            output_name = 'output_identifier' + str(outputs_index+1) + '_var' + str(var_index+1) + '_' + str(config['function_family']['decision_sparsity'])
                            outputs_identifer_neurons = config['data']['number_of_variables']
                            if config['i_net']['additional_hidden']:
                                hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, 
                                                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                                 name='hidden_' + output_name)(hidden)
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                          name=output_name)(hidden_outputs_identifer)                       
                            else:                                 
                                outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                          activation='softmax', 
                                                                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                          name=output_name)(hidden)
                            outputs_list.append(outputs_identifer)    

                
            elif config['i_net']['function_representation_type'] >= 3:                
                if config['function_family']['dt_type'] == 'SDT':
                    
                    outputs_coeff_neurons = internal_node_num_*config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden_output_coeff_' + str(outputs_coeff_neurons))(hidden)
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                      
                    else:                          
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              #activation='tanh', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                    outputs_list = [outputs_coeff]
                    
                    
                    for outputs_index in range(internal_node_num_):
                        output_name = 'output_identifier_' + str(outputs_index+1)
                        outputs_identifer_neurons = config['data']['number_of_variables']
                        if config['i_net']['additional_hidden']:
                            hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, 
                                                                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                             name='hidden_' + output_name)(hidden)
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                      name=output_name)(hidden_outputs_identifer)                     
                        else:                          
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                      name=output_name)(hidden)
                        outputs_list.append(outputs_identifer)    
              
                    
                elif config['function_family']['dt_type'] == 'vanilla':                    
                    
                    outputs_coeff_neurons = internal_node_num_*config['data']['number_of_variables']
                    if config['i_net']['additional_hidden']:
                        hidden_outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons*2, 
                                                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                     name='hidden' + 'output_coeff_' + str(outputs_coeff_neurons))(hidden)    
                        outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                              activation='sigmoid', 
                                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                              name='output_coeff_' + str(outputs_coeff_neurons))(hidden_outputs_coeff)                        
                    else:
                        
                        if config['i_net']['function_representation_type'] == 4:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-2*x))
                                return x                        

                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/2))
                                return x      
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                            
                        
                        elif config['i_net']['function_representation_type'] == 5:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-3*x))
                                return x                        

                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/3))
                                return x      
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                                      
                                
                        elif config['i_net']['function_representation_type'] == 6:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-5*x))
                                return x                        

                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/5))
                                return x      
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                                      
                                
                                
                                                                
                                
                        elif config['i_net']['function_representation_type'] == 7:
                            from keras import backend as K
                            def sigmoid_squeeze_inverse(x):
                                x = 1/(1+K.exp(3*x))
                                return x                        

                            def sigmoid_relaxed_inverse(x):
                                x = 1/(1+K.exp(x/3))
                                return x      
                            
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze_inverse)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                            
                            
                        elif config['i_net']['function_representation_type'] == 8:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-4*x))
                                return x                        

                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/4))
                                return x      
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                                      
                        elif config['i_net']['function_representation_type'] == 9:
                            from keras import backend as K
                            def sigmoid_squeeze(x):
                                x = 1/(1+K.exp(-2.5*x))
                                return x                        

                            def sigmoid_relaxed(x):
                                x = 1/(1+K.exp(-x/2.5))
                                return x      
                            
                            from keras.utils.generic_utils import get_custom_objects

                            get_custom_objects().update({'activation_special': Activation(sigmoid_squeeze)})                            

                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='activation_special', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)                                      
                                
                                
                                                                                               
                                                               
                                    
                            
                        elif config['i_net']['function_representation_type'] == 3:
                            outputs_coeff = tf.keras.layers.Dense(outputs_coeff_neurons, 
                                                                  activation='sigmoid', 
                                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                  name='output_coeff_' + str(outputs_coeff_neurons))(hidden)
                    outputs_list = [outputs_coeff]
                    
                    
                    for outputs_index in range(internal_node_num_):
                        output_name = 'output_identifier_' + str(outputs_index+1)
                        outputs_identifer_neurons = config['data']['number_of_variables']
                        if config['i_net']['additional_hidden']:
                            hidden_outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons*2, 
                                                                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                             name='hidden' + output_name)(hidden)                        
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                      name=output_name)(hidden_outputs_identifer)
                        else:
                            outputs_identifer = tf.keras.layers.Dense(outputs_identifer_neurons, 
                                                                      activation='softmax', 
                                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                      name=output_name)(hidden)                            
                        outputs_list.append(outputs_identifer)    

 
                
        
                
        
        
          
        
        
        
            if config['function_family']['dt_type'] == 'SDT':
                outputs_bias_neurons = internal_node_num_
                if config['i_net']['additional_hidden']:
                    hidden_outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons*2, 
                                                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                name='hidden_' + 'output_bias_' + str(outputs_bias_neurons))(hidden)    
                    outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons, 
                                                         #activation='tanh', 
                                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                         name='output_bias_' + str(outputs_bias_neurons))(hidden_outputs_bias)
                else:
                    outputs_bias = tf.keras.layers.Dense(outputs_bias_neurons, 
                                                         #activation='tanh', 
                                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                         name='output_bias_' + str(outputs_bias_neurons))(hidden)
                outputs_list.append(outputs_bias)     

                outputs_leaf_nodes_neurons = leaf_node_num_ * config['data']['num_classes']
                if config['i_net']['additional_hidden']:
                    hidden_outputs_bias = tf.keras.layers.Dense(outputs_leaf_nodes_neurons*2, 
                                                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                name='hidden_' + 'output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)    
                    outputs_bias = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                         #activation='tanh', 
                                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                         name='output_leaf_nodes_' + str(outputs_leaf_nodes_neurons))(hidden_outputs_bias)
                else:                
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                               #activation='tanh', 
                                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                               name='output_leaf_nodes_' + str(outputs_leaf_nodes_neurons))(hidden)
                outputs_list.append(outputs_leaf_nodes)     

                outputs = concatenate(outputs_list, name='output_combined')            
            elif config['function_family']['dt_type'] == 'vanilla':
                outputs_leaf_nodes_neurons = leaf_node_num_
                if config['i_net']['additional_hidden']:
                    hidden_outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons*2, 
                                                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                                      name='hidden_' + 'output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)    
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                               activation='sigmoid', 
                                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                               name='output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden_outputs_leaf_nodes)                    
                else:
                    outputs_leaf_nodes = tf.keras.layers.Dense(outputs_leaf_nodes_neurons, 
                                                               activation='sigmoid', 
                                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                                                               name='output_leaf_node_' + str(outputs_leaf_nodes_neurons))(hidden)
                outputs_list.append(outputs_leaf_nodes)    

                outputs = concatenate(outputs_list, name='output_combined')        
                    

            model = Model(inputs=inputs, outputs=outputs)
                    
            if config['i_net']['early_stopping']:
                callback_names.append('early_stopping')
                callback_names.append('reduce_lr_loss')
            
            callbacks = return_callbacks_from_string(callback_names, config)            

            optimizer = tf.keras.optimizers.get(config['i_net']['optimizer'])
            optimizer.learning_rate = config['i_net']['learning_rate']

            random.seed(config['computation']['RANDOM_SEED'])
            np.random.seed(config['computation']['RANDOM_SEED'])  
            tf.random.set_seed(config['computation']['RANDOM_SEED'])  
            
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=metrics
                         )
            #print(model.get_weights())
            verbosity = 2 #if n_jobs ==1 else 0
            
            random.seed(config['computation']['RANDOM_SEED'])
            np.random.seed(config['computation']['RANDOM_SEED'])  
            tf.random.set_seed(config['computation']['RANDOM_SEED'])                
            ############################## PREDICTION ###############################
            
            #print(model.get_weights())
            
            history = model.fit(X_train,
                      y_train_model,
                      epochs=config['i_net']['epochs'], 
                      batch_size=config['i_net']['batch_size'], 
                      shuffle=False,
                      validation_data=valid_data,
                      callbacks=callbacks,
                      verbose=verbosity,
                      workers=1,
                      use_multiprocessing=False,
                    )

            history = history.history
            
            
            
            model.save('./data/saved_models/' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version']), save_format='tf')
            if encoder_model is not None:
                encoder_model.save('./data/saved_models/' + 'encoder_model_' + paths_dict['path_identifier_interpretation_net'] + dt_string + '_reshape' + str(config['i_net']['data_reshape_version']), save_format='tf')        

            
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
                  #Dense(min(reduction_size*4, num_features//2), activation="relu"),
                  #Dense(min(reduction_size*2, num_features//2), activation="relu"),
                  Dense(min(reduction_size*3, num_features//2), activation="relu"),
                  Dense(reduction_size, activation="relu", name='sequential')
                ]
            )

            self.decoder = Sequential(
                [
                  #Dense(min(reduction_size*2, num_features//2), activation="relu"),
                  #Dense(min(reduction_size*4, num_features//2), activation="relu"),
                  Dense(min(reduction_size*3, num_features//2), activation="relu"),
                  Dense(num_features//2, activation="relu"),
                  Dense(num_features, activation="linear")
                ]
            )

        def call(self, inputs):

            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded
    
    if encoder_model is None:
        
        encoder_model = AutoEncoders(num_features=X_data.shape[1], reduction_size=2*config['data']['number_of_variables']*(2**config['function_family']['maximum_depth'])) #5

        encoder_model.compile(
            loss='mae',
            metrics=['mae'],
            optimizer='adam'
        )

        history = encoder_model.fit(
            X_data[100:], 
            X_data[100:], 
            epochs=250,
            batch_size=128, 
            validation_data=(X_data[:100], X_data[:100]),
            callbacks=return_callbacks_from_string('early_stopping'),
            verbose=2,
            workers=1,
            use_multiprocessing=False,
             )
    
    encoder_layer = encoder_model.encoder#auto_encoder.get_layer('sequential')
    X_data = encoder_layer.predict(X_data)    
    
    return X_data, X_data_flat, encoder_model
    
def restructure_data_cnn_lstm(X_data, config, subsequences=None):
    import multiprocessing
    import psutil
    
    from utilities.utility_functions import generate_base_model, shape_flat_network_parameters
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