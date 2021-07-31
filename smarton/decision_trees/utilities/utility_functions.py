#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
from tqdm import tqdm_notebook as tqdm
#import pickle
import numpy as np
from numpy import linspace
import pandas as pd
import scipy as sp

from functools import reduce
from more_itertools import random_product
import operator

import math

from joblib import Parallel, delayed
from collections.abc import Iterable
#from scipy.integrate import quad
import matplotlib.pyplot as plt 


#from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw
import time

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from IPython.display import display, Math, Latex, clear_output

import os
import shutil
import pickle
    
#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.DecisionTree_BASIC import *
#from utilities.utility_functions import *

from scipy.optimize import minimize
from scipy import optimize
import sympy as sym
from sympy import Symbol, sympify, lambdify, abc, SympifyError

# Function Generation 0 1 import
from sympy.sets.sets import Union
import math

from numba import jit, njit
import itertools 

from interruptingcow import timeout
from livelossplot import PlotLossesKerasTF
from sklearn.datasets import make_classification

                                    
#######################################################################################################################################################
#############################################################General Utility Functions#################################################################
#######################################################################################################################################################
                                                        
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def chunks(lst, chunksize):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunksize):
        yield lst[i:i + chunksize]

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
        
def return_float_tensor_representation(some_representation, dtype=tf.float32):
    if tf.is_tensor(some_representation):
        some_representation = tf.dtypes.cast(some_representation, dtype) 
    else:
        some_representation = tf.convert_to_tensor(some_representation)
        some_representation = tf.dtypes.cast(some_representation, dtype) 
        
    if not tf.is_tensor(some_representation):
        raise SystemExit('Given variable is no instance of ' + str(dtype) + ':' + str(some_representation))
     
    return some_representation


def sleep_minutes(minutes):
    time.sleep(int(60*minutes))
    
def sleep_hours(hours):
    time.sleep(int(60*60*hours))
    
    
    


def return_numpy_representation(some_representation):
    if isinstance(some_representation, pd.DataFrame):
        some_representation = some_representation.values
        some_representation = np.float32(some_representation)
        
    if isinstance(some_representation, list):
        some_representation = np.array(some_representation, dtype=np.float32)
        
    if isinstance(some_representation, np.ndarray):
   
        some_representation = np.float32(some_representation)
    else:
        raise SystemExit('Given variable is no instance of ' + str(np.ndarray) + ':' + str(some_representation))
    
    return some_representation


def mergeDict(dict1, dict2):
    #Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend(value)
            elif isinstance(dict1[key], list) and not isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend([value])
            elif not isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = [dict1[key]]
                newDict[key].extend(value)
            else:
                newDict[key] = [dict1[key], value]
    return newDict


def return_callbacks_from_string(callback_string_list):
    callbacks = [] if len(callback_string_list) > 0 else None
    #if 'plot_losses_callback' in callback_string_list:
        #callbacks.append(PlotLossesCallback())
    if 'reduce_lr_loss' in callback_string_list:
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=0, min_delta=0, mode='min') #epsilon
        callbacks.append(reduce_lr_loss)
    if 'early_stopping' in callback_string_list:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0, verbose=0, mode='min', restore_best_weights=True)
        callbacks.append(earlyStopping)        
    if 'plot_losses' in callback_string_list:
        plotLosses = PlotLossesKerasTF()
        callbacks.append(plotLosses) 
    #if not multi_epoch_analysis and samples_list == None: 
        #callbacks.append(TQDMNotebookCallback())        
    return callbacks


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def flatten_list(l):
    
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
                
    flat_l = flatten(l)
    
    return list(flat_l)


#######################################################################################################################################################
###########################Manual calculations for comparison of polynomials based on function values (no TF!)#########################################
#######################################################################################################################################################

    
def generate_paths(config, path_type='interpretation_net'):

    paths_dict = {}
    
                

        
    data_specification_string = ('_var' + str(config['data']['number_of_variables']) +
                                  '_class' + str(config['data']['num_classes']) +
                                  '_' + str(config['data']['function_generation_type']) +
                                  #'_' + str(config['data']['objective']) +
                                  '_xMax' + str(config['data']['x_max']) +
                                  '_xMin' + str(config['data']['x_min']) +
                                  '_xDist' + str(config['data']['x_distrib']) +
                                  
                                  '_depth' + str(config['function_family']['maximum_depth']) +
                                  '_' + ('fullyGrown' if config['function_family']['fully_grown'] else 'partiallyGrown')
                                 )

        
        

    if path_type == 'data_creation' or path_type == 'lambda_net': #Data Generation
  
        path_identifier_function_data = ('lNetSize' + str(config['data']['lambda_dataset_size']) +
                                         '_numDatasets' + str(config['data']['number_of_generated_datasets']) +
                                         data_specification_string)            

        paths_dict['path_identifier_function_data'] = path_identifier_function_data
    
    if path_type == 'lambda_net' or path_type == 'interpretation_net': #Lambda-Net
            
        
            
        lambda_layer_str = '-'.join([str(neurons) for neurons in config['lambda_net']['lambda_network_layers']])
        
        early_stopping_string = 'ES' + str(config['lambda_net']['early_stopping_min_delta_lambda']) if config['lambda_net']['early_stopping_lambda'] else ''
        lambda_init_string = 'noFixedInit' if config['lambda_net']['number_initializations_lambda'] == -1 else 'fixedInit' + str(config['lambda_net']['number_initializations_lambda']) + '-seed' + str(config['computation']['RANDOM_SEED'])
        lambda_noise_string = '_noise-' + config['data']['noise_injected_type'] + str(config['data']['noise_injected_level']) if config['data']['noise_injected_level'] > 0 else ''
        
        
        lambda_net_identifier = (
                                 lambda_layer_str + 
                                 '_e' + str(config['lambda_net']['epochs_lambda']) + early_stopping_string + 
                                 '_b' + str(config['lambda_net']['batch_lambda']) + 
                                 '_drop' + str(config['lambda_net']['dropout_lambda']) + 
                                 '_' + config['lambda_net']['optimizer_lambda'] + 
                                 '_' + config['lambda_net']['loss_lambda'] +
                                 '_' + lambda_init_string + 
                                 lambda_noise_string
                                )

        path_identifier_lambda_net_data = ('lNetSize' + str(config['data']['lambda_dataset_size']) +
                                           '_numLNets' + str(config['lambda_net']['number_of_trained_lambda_nets']) +
                                           data_specification_string + 
                                           
                                           '/' +
                                           lambda_net_identifier)
                                           

        paths_dict['path_identifier_lambda_net_data'] = path_identifier_lambda_net_data
    
    
    if path_type == 'interpretation_net': #Interpretation-Net   
            
        interpretation_network_layers_string = 'dense' + '-'.join([str(neurons) for neurons in config['i_net']['dense_layers']])

        if config['i_net']['convolution_layers'] != None:
            interpretation_network_layers_string += 'conv' + '-'.join([str(neurons) for neurons in config['i_net']['convolution_layers']])
        if config['i_net']['lstm_layers'] != None:
            interpretation_network_layers_string += 'lstm' + '-'.join([str(neurons) for neurons in config['i_net']['lstm_layers']])

        interpretation_net_identifier = '_' + interpretation_network_layers_string + '_drop' + '-'.join([str(dropout) for dropout in config['i_net']['dropout']]) + 'e' + str(config['i_net']['epochs']) + 'b' + str(config['i_net']['batch_size']) + '_' + config['i_net']['optimizer']
        
        path_identifier_interpretation_net = ('lNetSize' + str(config['data']['lambda_dataset_size']) +
                                                   '_numLNets' + str(config['lambda_net']['number_of_trained_lambda_nets']) +
                                                   data_specification_string + 

                                                   '/' +
                                                   lambda_net_identifier +
            
                                                   '/' +
                                                   'inet' + interpretation_net_identifier)
        
        
        paths_dict['path_identifier_interpretation_net'] = path_identifier_interpretation_net
        
    return paths_dict





def create_folders_inet(config):
    
    paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
    try:
        # Create target Directory
        os.makedirs('./data/plotting/' + paths_dict['path_identifier_interpretation_net'] + '/')
        os.makedirs('./data/results/' + paths_dict['path_identifier_interpretation_net'] + '/')
    except FileExistsError:
        pass
    

def generate_directory_structure():
    
    directory_names = ['parameters', 'plotting', 'saved_function_lists', 'results', 'saved_models', 'weights', 'weights_training']
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
        text_file = open('./data/.gitignore', 'w')
        text_file.write('*')
        text_file.close()  
        
    for directory_name in directory_names:
        path = './data/' + directory_name
        if not os.path.exists(path):
            os.makedirs(path)
            
            
def generate_lambda_net_directory(config):
    
    paths_dict = generate_paths(config, path_type = 'lambda_net')
    
    #clear files
    try:
        # Create target Directory
        os.makedirs('./data/weights/weights_' + paths_dict['path_identifier_lambda_net_data'])

    except FileExistsError:
        folder = './data/weights/weights_' + paths_dict['path_identifier_lambda_net_data']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e)) 
    try:
        # Create target Directory
        os.makedirs('./data/results/weights_' + paths_dict['path_identifier_lambda_net_data'])
    except FileExistsError:
        pass
    
    
######################################################################################################################################################################################################################
########################################################################################  RANDOM FUNCTION GENERATION FROM ############################################################################################ 
######################################################################################################################################################################################################################

######################################################################################################################################################################################################################
def generate_random_data_points(low, high, size, variables, distrib='uniform'):
    if distrib=='normal':
        list_of_data_points = []
        for _ in range(size):
            random_data_points = np.random.normal(loc=(low+high)/2, scale=(low+high)/4, size=variables)
            while max(random_data_points) > high and min(random_data_points) < low:
                random_poly = np.random.normal(loc=(low+high)/2, scale=1.0, size= variables)
            list_of_data_points.append(random_poly)
        list_of_data_points = np.array(list_of_polynomials)
        
    elif distrib=='uniform':
        list_of_data_points = np.random.uniform(low=low, high=high, size=(size, variables))
        
    return list_of_data_points
######################################################################################################################################################################################################################


def get_shaped_parameters_for_decision_tree(flat_parameters, config):
    
    input_dim = config['data']['number_of_variables']
    output_dim = config['data']['num_classes']
    internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
    leaf_node_num_ = 2 ** config['function_family']['maximum_depth']

    
    weights = flat_parameters[:input_dim*internal_node_num_]
    weights = tf.reshape(weights, (internal_node_num_, input_dim))


    biases = flat_parameters[input_dim*internal_node_num_:(input_dim+1)*internal_node_num_]

    leaf_probabilities = flat_parameters[(input_dim+1)*internal_node_num_:]
    leaf_probabilities = tf.transpose(tf.reshape(leaf_probabilities, (leaf_node_num_, output_dim)))

    return weights, biases, leaf_probabilities
    


def generate_decision_tree_from_array(parameter_array, config):
    tree = SDT(input_dim=config['data']['number_of_variables'],
               output_dim=config['data']['num_classes'],
               depth=config['function_family']['maximum_depth'],
               use_cuda=False,
               verbosity=0)
    
    tree.initialize_from_parameter_array(parameter_array)
    
    return tree


def generate_random_decision_tree(config, seed=42):
    
    #random.seed(seed)
    #np.random.seed(seed)
    
    if config['function_family']['fully_grown']:
        tree = SDT(input_dim=config['data']['number_of_variables'],#X_train.shape[1], 
                   output_dim=config['data']['num_classes'],#int(max(y_train))+1, 
                   depth=config['function_family']['maximum_depth'],
                   random_seed=seed,
                   use_cuda=False,
                   verbosity=0)#
        
    else: 
        raise SystemExit('Partially Grown Trees not implemented yet')
        
    
    return tree


def generate_random_data_points(config, seed):
    
    random.seed(seed)
    np.random.seed(seed)
    
    if config['data']['x_distrib']=='normal':
        list_of_data_points = []
        x_range = config['data']['x_max']-config['data']['x_min']
        for _ in range(config['data']['lambda_dataset_size']):
            random_data_point = np.random.normal(loc=x_range/2, scale=x_range/4, size=config['data']['number_of_variables'])
            while max(random_data_point) > config['data']['x_max'] or min(random_data_point) < config['data']['x_min']:
                random_data_point = np.random.normal(loc=x_range/2, scale=x_range, size=config['data']['number_of_variables'])
            list_of_data_points.append(random_data_point)
        list_of_data_points = np.array(list_of_data_points)
        
    elif config['data']['x_distrib']=='uniform':
        list_of_data_points = np.random.uniform(low=config['data']['x_min'], high=config['data']['x_max'], size=(config['data']['lambda_dataset_size'], config['data']['number_of_variables']))
        
    return list_of_data_points 


def generate_decision_tree_data_trained_make_classification(config, seed=42):
    
    decision_tree = generate_random_decision_tree(config, seed)
        
    X_data, y_data_tree = make_classification(n_samples=X_data.shape[0], 
                                                       n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                       n_informative=config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                       n_redundant=0, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                       n_repeated=0, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                       n_classes=config['data']['num_classes'], 
                                                       n_clusters_per_class=2, 
                                                       flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                       class_sep=1.0, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                       hypercube=True, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                       shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                       scale=1.0, #Multiply features by the specified value. 
                                                       shuffle=True, 
                                                       random_state=seed) 
    
    decision_tree = SDT(input_dim=config['data']['number_of_variables'],#X_train.shape[1], 
                   output_dim=config['data']['num_classes'],#int(max(y_train))+1, 
                   depth=config['function_family']['maximum_depth'],
                   random_seed=seed,
                   use_cuda=False,
                   verbosity=0)
    
    decision_tree.fit(X_data, y_data_tree, epochs=50)    
    
    y_data = decision_tree.predict_proba(X_data)
    counter = 1
    
    while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
        seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
        counter += 1
        
        decision_tree = generate_random_decision_tree(config, seed)
        y_data = decision_tree.predict_proba(X_data)    #predict_proba #predict    
    
    return decision_tree.to_array(), X_data, np.round(y_data), y_data 


def generate_decision_tree_data_trained(config, seed=42):
    
    decision_tree = generate_random_decision_tree(config, seed)
    
    X_data = generate_random_data_points(config, seed)
    
    y_data_tree = np.random.randint(0,2,X_data.shape[0])
    
    decision_tree = SDT(input_dim=config['data']['number_of_variables'],#X_train.shape[1], 
                   output_dim=config['data']['num_classes'],#int(max(y_train))+1, 
                   depth=config['function_family']['maximum_depth'],
                   random_seed=seed,
                   use_cuda=False,
                   verbosity=0)
    
    decision_tree.fit(X_data, y_data_tree, epochs=50)    
    
    y_data = decision_tree.predict_proba(X_data)
    counter = 1
    
    while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
        seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
        counter += 1
        
        decision_tree = generate_random_decision_tree(config, seed)
        y_data = decision_tree.predict_proba(X_data)    #predict_proba #predict
    
    return decision_tree.to_array(), X_data, np.round(y_data), y_data 



def generate_decision_tree_data(config, seed=42):
    
    decision_tree = generate_random_decision_tree(config, seed)
    
    X_data = generate_random_data_points(config, seed)
    
    y_data = decision_tree.predict_proba(X_data)
    counter = 1
    
    while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
        seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
        counter += 1
        
        decision_tree = generate_random_decision_tree(config, seed)
        y_data = decision_tree.predict_proba(X_data)    #predict_proba #predict
    
    return decision_tree.to_array(), X_data, np.round(y_data), y_data 


def generate_decision_tree_identifier(config):
    num_internal_nodes = 2 ** config['function_family']['maximum_depth'] - 1
    num_leaf_nodes = 2 ** config['function_family']['maximum_depth']
    
    filter_shape = (num_internal_nodes, config['data']['number_of_variables'])
    bias_shape = (num_internal_nodes, 1)
    
    leaf_probabilities_shape = (num_leaf_nodes, config['data']['num_classes'])
    
    decision_tree_identifier_list = []
    for filter_number in range(filter_shape[0]):
        for variable_number in range(filter_shape[1]):
            decision_tree_identifier_list.append('f' + str(filter_number) + 'v' + str(variable_number))
            
    for bias_number in range(bias_shape[0]):
        decision_tree_identifier_list.append('b' + str(bias_number))
            
    for leaf_probabilities_number in range(leaf_probabilities_shape[0]):
        for class_number in range(leaf_probabilities_shape[1]):
            decision_tree_identifier_list.append('lp' + str(leaf_probabilities_number) + 'c' + str(class_number))       
            
    return decision_tree_identifier_list



######################################################################################################################################################################################################################
###########################################################################################  LAMBDA NET UTILITY ################################################################################################ 
######################################################################################################################################################################################################################


        
def split_LambdaNetDataset(dataset, test_split, random_seed=42):
    
    from utilities.LambdaNet import LambdaNetDataset
    
    assert isinstance(dataset, LambdaNetDataset) 
    
    lambda_nets_list = dataset.lambda_net_list
    
    if isinstance(test_split, int) or isinstance(test_split, float):
        lambda_nets_train_list, lambda_nets_test_list = train_test_split(lambda_nets_list, test_size=test_split, random_state=random_seed)     
    elif isinstance(test_split, list):
        lambda_nets_test_list = [lambda_nets_list[i] for i in test_split]
        lambda_nets_train_list = list(set(lambda_nets_list) - set(lambda_nets_test_list))
        #lambda_nets_train_list = lambda_nets_list.copy()
        #for i in sorted(test_split, reverse=True):
        #    del lambda_nets_train_list[i]           
    assert len(lambda_nets_list) == len(lambda_nets_train_list) + len(lambda_nets_test_list)
    
    return LambdaNetDataset(lambda_nets_train_list), LambdaNetDataset(lambda_nets_test_list)
                                                                                                 
def generate_base_model(config): #without dropout
    
    output_neurons = 1 if config['data']['num_classes']==2 else config['data']['num_classes']
    output_activation = 'sigmoid' if config['data']['num_classes']==2 else 'softmax'
    
    model = Sequential()
        
    #kerase defaults: kernel_initializer='glorot_uniform', bias_initializer='zeros'               
    model.add(Dense(config['lambda_net']['lambda_network_layers'][0], activation='relu', input_dim=config['data']['number_of_variables'], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), bias_initializer='zeros'))
   
    if config['lambda_net']['dropout_lambda'] > 0:
        model.add(Dropout(config['lambda_net']['dropout_lambda']))

    for neurons in config['lambda_net']['lambda_network_layers'][1:]:
        model.add(Dense(neurons, 
                        activation='relu', 
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                        bias_initializer='zeros'))
        
        if config['lambda_net']['dropout_lambda'] > 0:
            model.add(Dropout(config['lambda_net']['dropout_lambda']))   
    
    model.add(Dense(output_neurons, 
                    activation=output_activation, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                    bias_initializer='zeros'))    
    
    return model




def shape_flat_network_parameters(flat_network_parameters, target_network_parameters):
    
    #äfrom utilities.utility_functions import flatten_list
    
    #def recursive_len(item):
    #    if type(item) == list:
    #        return sum(recursive_len(subitem) for subitem in item)
    #    else:
    #        return 1      
        
    shaped_network_parameters =[]
    start = 0  
    
    for parameters in target_network_parameters:
        target_shape = parameters.shape
        size = np.prod(target_shape)#recursive_len(el)#len(list(flatten_list(el)))
        shaped_parameters = np.reshape(flat_network_parameters[start:start+size], target_shape)
        shaped_network_parameters.append(shaped_parameters)
        start += size

    return shaped_network_parameters

def network_parameters_to_pred(weights, x, config, base_model=None):

    if base_model is None:
        base_model = generate_base_model(config)
    base_model_network_parameters = base_model.get_weights()
    
    # Shape weights (flat) into correct model structure
    shaped_network_parameters = shape_flat_network_parameters(weights, base_model_weights)
    
    model = tf.keras.models.clone_model(base_model)
    
    # Make prediction
    model.set_weights(shaped_network_parameters)
    y = model.predict(x).ravel()
    return y

    
def network_parameters_to_network(network_parameters, config, base_model=None):
    
    if base_model is None:
        model = generate_base_model(config)    
    else:
        model = tf.keras.models.clone_model(base_model)
    
    model_network_parameters = model.get_weights()    
 

    # Shape weights (flat) into correct model structure
    shaped_network_parameters = shape_flat_network_parameters(network_parameters, model_network_parameters)
    
    model.set_weights(shaped_network_parameters)
    
    model.compile(optimizer=config['lambda_net']['optimizer_lambda'],
                  loss='binary_crossentropy',#tf.keras.losses.get(config['lambda_net']['loss_lambda']),
                  metrics=[tf.keras.metrics.get("binary_accuracy"), tf.keras.metrics.get("accuracy")]
                 )
    
    return model  


def shaped_network_parameters_to_array(shaped_network_parameters, config):
    network_parameter_list = []
    for layer_weights, biases in pairwise(shaped_network_parameters):    #clf.get_weights()
        for neuron in layer_weights:
            for weight in neuron:
                network_parameter_list.append(weight)
        for bias in biases:
                network_parameter_list.append(bias)
                
    return np.array(network_parameter_list)



######################################################################################################################################################################################################################
###########################################################################################  PER NETWORK OPTIMIZATION ################################################################################################ 
######################################################################################################################################################################################################################



def per_network_poly_optimization_tf(per_network_dataset_size, 
                                  lambda_network_weights, 
                                  list_of_monomial_identifiers_numbers, 
                                  config, 
                                  optimizer = tf.optimizers.Adam,
                                  lr=0.05, 
                                  max_steps = 1000, 
                                  early_stopping=10, 
                                  restarts=5, 
                                  printing=True,
                                  return_error=False):
    
    
    from utilities.metrics import calculate_poly_fv_tf_wrapper
    from utilities.metrics import r2_keras_loss

    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################
            
    globals().update(config)
        
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)    
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)       
    

    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))
    
    weights_structure = base_model.get_weights()
    
    
    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))
    random_lambda_input_data = tf.dtypes.cast(tf.convert_to_tensor(random_lambda_input_data), tf.float32)
    list_of_monomial_identifiers_numbers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers_numbers), tf.float32)
    
    model_lambda_placeholder = tf.keras.models.clone_model(base_model)  
    
    dims = [np_arrays.shape for np_arrays in weights_structure]
    

    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)
    
    #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
    # build models
    start = 0
    layers = []
    for i in range(len(dims)//2):

        # set weights of layer
        index = i*2
        size = np.product(dims[index])
        weights_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
        start += size

        # set biases of layer
        index += 1
        size = np.product(dims[index])
        biases_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
        start += size


    lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_lambda_input_data))    
    

    
    ########################################### OPTIMIZATION ########################################################
        
    current_monomial_degree = tf.Variable(0, dtype=tf.int64)
    best_result = np.inf

    for current_iteration in range(restarts):
                
        @tf.function(experimental_compile=True) 
        def function_to_optimize():
            
            poly_optimize = poly_optimize_input[0]

            if interpretation_net_output_monomials != None:
                poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]
                poly_optimize_identifiers_list = []
                if sparse_poly_representation_version == 1:
                    for i in range(interpretation_net_output_monomials):
                        poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                        poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                    poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                else:
                    for i in range(interpretation_net_output_monomials):
                        for j in range(n):
                            poly_optimize_identifiers = tf.math.softmax(poly_optimize[i*n*j*(d+1)+interpretation_net_output_monomials:(i+1)*n*j*(d+1)+interpretation_net_output_monomials])
                            poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                    poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)                
                poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

            poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, current_monomial_degree, config=config), (random_lambda_input_data))

            error = None
            if inet_loss == 'mae':
                error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
            elif inet_loss == 'r2':
                error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
            else:
                raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

            error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)   

            return error 

    
            
        opt = optimizer(learning_rate=lr)
        
        poly_optimize_input = tf.Variable(tf.random.uniform([1, interpretation_net_output_shape]))
        
        stop_counter = 0
        best_result_iteration = np.inf

        for current_step in range(max_steps):
            if stop_counter>=early_stopping:
                break
            
            opt.minimize(function_to_optimize, var_list=[poly_optimize_input])
            current_result = function_to_optimize()
            if printing:
                clear_output(wait=True)
                print("Current best: {} \n Curr_res: {} \n Iteration {}, Step {}".format(best_result_iteration,current_result, current_iteration, current_step))
 
            stop_counter += 1
            if current_result < best_result_iteration:
                best_result_iteration = current_result
                stop_counter = 0
                best_poly_optimize_iteration = tf.identity(poly_optimize_input)
                
        if best_result_iteration < best_result:
            best_result = best_result_iteration
            best_poly_optimize = tf.identity(best_poly_optimize_iteration)
            

    per_network_poly = best_poly_optimize[0].numpy()
    
    if printing:
        print("Optimization terminated at {}".format(best_result))
        
    if return_error:
        return best_result, per_network_poly
    
    return per_network_poly



def per_network_poly_optimization_scipy(per_network_dataset_size, 
                                          lambda_network_weights, 
                                          list_of_monomial_identifiers_numbers, 
                                          config, 
                                          optimizer = 'Nelder-Mead',
                                          jac = None,
                                          max_steps = 1000, 
                                          restarts=5, 
                                          printing=True,
                                          return_error=False):

    from utilities.metrics import calculate_poly_fv_tf_wrapper

    def copy( self ):
        return tf.identity(self)
    tf.Tensor.copy = copy


    ########################################### GENERATE RELEVANT PARAMETERS FOR OPTIMIZATION ########################################################

    globals().update(config)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)    
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)       


    base_model = Sequential()

    base_model.add(Dense(lambda_network_layers[0], activation='relu', input_dim=n))

    for neurons in lambda_network_layers[1:]:
        base_model.add(Dense(neurons, activation='relu'))

    base_model.add(Dense(1))

    weights_structure = base_model.get_weights()


    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))
    random_lambda_input_data = tf.dtypes.cast(tf.convert_to_tensor(random_lambda_input_data), tf.float32)
    list_of_monomial_identifiers_numbers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers_numbers), tf.float32)

    model_lambda_placeholder = tf.keras.models.clone_model(base_model)  

    dims = [np_arrays.shape for np_arrays in weights_structure]


    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)

    #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
    # build models
    start = 0
    layers = []
    for i in range(len(dims)//2):

        # set weights of layer
        index = i*2
        size = np.product(dims[index])
        weights_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
        start += size

        # set biases of layer
        index += 1
        size = np.product(dims[index])
        biases_tf_true = tf.reshape(lambda_network_weights[start:start+size], dims[index])
        model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
        start += size


    lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_lambda_input_data))    



    ########################################### OPTIMIZATION ########################################################

    current_monomial_degree = tf.Variable(0, dtype=tf.int64)
    best_result = np.inf

    for current_iteration in range(restarts):

        def function_to_optimize_scipy_wrapper(current_monomial_degree):
            @tf.function(experimental_compile=True) 
            def function_to_optimize_scipy(poly_optimize_input):   

                #poly_optimize = tf.cast(tf.constant(poly_optimize_input), tf.float32)
                poly_optimize = tf.cast(poly_optimize_input, tf.float32)

                if interpretation_net_output_monomials != None:
                    poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]
                    poly_optimize_identifiers_list = []
                    if sparse_poly_representation_version == 1:
                        for i in range(interpretation_net_output_monomials):
                            poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                            poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                        poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                    else:
                        for i in range(interpretation_net_output_monomials):
                            for j in range(n):
                                poly_optimize_identifiers = tf.math.softmax(poly_optimize[i*n*j*(d+1)+interpretation_net_output_monomials:(i+1)*n*j*(d+1)+interpretation_net_output_monomials])
                                poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                        poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)                
                    poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

                poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, current_monomial_degree, config=config), (random_lambda_input_data))

                error = None
                if inet_loss == 'mae':
                    error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
                elif inet_loss == 'r2':
                    error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
                else:
                    raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

                error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)   

                return error
            return function_to_optimize_scipy


        poly_optimize_input = tf.random.uniform([1, interpretation_net_output_shape])    

        def function_to_optimize_scipy_grad_wrapper(current_monomial_degree):
            def function_to_optimize_scipy_grad(poly_optimize_input):

                error = function_to_optimize_scipy_wrapper(current_monomial_degree)(poly_optimize_input)
                error = error.numpy()
                return error
            return function_to_optimize_scipy_grad

        stop_counter = 0


        if jac=='fprime':
            jac = lambda x: optimize.approx_fprime(x, function_to_optimize_scipy_grad_wrapper(current_monomial_degree), 0.01)

        #tf.print(interpretation_net_output_monomials)
        #tf.print(config)        
        opt_res = minimize(function_to_optimize_scipy_wrapper(current_monomial_degree), poly_optimize_input, method=optimizer, jac=jac, options={'maxfun': None, 'maxiter': max_steps})
        print(opt_res)
        #opt_res = minimize(function_to_optimize_scipy_wrapper(current_monomial_degree), poly_optimize_input, method=optimizer, options={'maxfun': None, 'maxiter': max_steps})

        best_result_iteration = opt_res.fun
        best_poly_optimize_iteration = opt_res.x

        if best_result_iteration < best_result:
            best_result = best_result_iteration
            best_poly_optimize = best_poly_optimize_iteration

    per_network_poly = best_poly_optimize

    if printing:
        print("Optimization terminated at {}".format(best_result))

    if return_error:
        return best_result, per_network_poly
    
    return per_network_poly

