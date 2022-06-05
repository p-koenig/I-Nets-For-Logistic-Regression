#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
from tqdm.notebook import tqdm
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
import collections
from collections.abc import Iterable
#from scipy.integrate import quad
import matplotlib.pyplot as plt 
import datetime


#from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
#from similaritymeasures import frechet_dist, area_between_two_curves, dtw
import time

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
from utilities.InterpretationNet import *

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
from utilities.make_classification_distribution import make_classification_distribution
from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from prettytable import PrettyTable
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder

from copy import deepcopy

from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree as ctree
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xgboost as xgb

from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
                                    
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
    if minutes > 0:
        for _ in tqdm(range(minutes)):
            time.sleep(60)
        
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
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                newDict[key] = mergeDict(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
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


def return_callbacks_from_string(callback_string_list, config=None):
    
    from utilities.InterpretationNet import CustomStopper
    
    callbacks = [] if len(callback_string_list) > 0 else None
    #if 'plot_losses_callback' in callback_string_list:
        #callbacks.append(PlotLossesCallback())
    if 'reduce_lr_loss' in callback_string_list:
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=50, verbose=0, min_delta=0.001, mode='min') #epsilon #, factor=0.1
        #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=25, verbose=0, min_delta=0.001, mode='min') #epsilon #, factor=0.1
        
        callbacks.append(reduce_lr_loss)
    if 'early_stopping' in callback_string_list:
        #earlyStopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0.001, verbose=0, mode='min', restore_best_weights=True)
        earlyStopping = CustomStopper(monitor='val_loss', patience=100, min_delta=0.001, verbose=0, mode='min', restore_best_weights=True, start_epoch = 10)
        #earlyStopping = CustomStopper(monitor='val_loss', patience=25, min_delta=0.001, verbose=0, mode='min', restore_best_weights=False, start_epoch = 10)
        
        callbacks.append(earlyStopping)        
    if 'plot_losses' in callback_string_list:
        plotLosses = PlotLossesKerasTF()
        callbacks.append(plotLosses) 
    if 'tensorboard' in callback_string_list:
        paths_dict = generate_paths(config, path_type = 'interpretation_net')
        log_dir = './data/logging/' + paths_dict['path_identifier_interpretation_net'] + '/tensorboard'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #log_dir = './data/logging/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback) 

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

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
#######################################################################################################################################################
###########################Manual calculations for comparison of polynomials based on function values (no TF!)#########################################
#######################################################################################################################################################

    
def generate_paths(config, path_type='interpretation_net'):

    paths_dict = {}    
    
    try:
        dt_type = config['data']['dt_type_train'] if config['data']['dt_type_train'] is not None else config['function_family']['dt_type']
    except:
        dt_type = config['function_family']['dt_type']
     
    try:
        maximum_depth = config['data']['maximum_depth_train'] if config['data']['maximum_depth_train'] is not None else config['function_family']['maximum_depth']
    except:
        maximum_depth = config['function_family']['maximum_depth']
        
    try:
        decision_sparsity = config['data']['decision_sparsity_train'] if config['data']['decision_sparsity_train'] is not None else config['function_family']['decision_sparsity']
    except:
        decision_sparsity = config['function_family']['decision_sparsity']       
        

        
    data_noise = '' if config['data']['data_noise'] is None else '_dNoise' + str(config['data']['data_noise'])
                
    decision_sparsity = -1 if decision_sparsity == config['data']['number_of_variables'] else decision_sparsity
     
    categorical_sting = ''
    if len(config['data']['categorical_indices']) > 0:
        categorical_sting = '_cat' + '-'.join(str(e) for e in config['data']['categorical_indices'])
        
    random_parameters_distribution_string = '_randParamDist' if config['data']['random_parameters_distribution'] else ''
    max_distributions_per_class_string = '_maxDistClass' + str(config['data']['max_distributions_per_class']) if config['data']['max_distributions_per_class'] is not None else ''
    
    distrib_param_max_str = ''
    try:
        distrib_param_max_str = '_distribParamMax' + str(config['data']['distrib_param_max'])
    except:
        pass
        
    fixed_class_probability_str = ''
    try:
        fixed_class_probability_str = '_randClassProb' if not config['data']['fixed_class_probability'] else ''
    except:
        pass
        
    weighted_data_generation_str = ''
    try:
        weighted_data_generation_str = '_weightedFeatures' if config['data']['weighted_data_generation'] else ''
    except:
        pass

    data_generation_filtering_str = ''
    try:
        data_generation_filtering_str = '_filterGen' if config['data']['data_generation_filtering'] else ''
    except:
        pass    
        
    data_generation_linearly_separable_str = ''
    if not config['data']['data_generation_filtering']:
        try:
            data_generation_linearly_separable_str = '_exLinSep' if config['data']['exclude_linearly_seperable'] else ''
        except:
            pass            
        
    data_generation_shift_str = ''
    try:
        data_generation_shift_str = '_shifted' if config['data']['shift_distrib'] else ''
    except:
        pass   
    
    balanced_data_str = ''
    try:
        data_generation_shift_str = '_noBalance' if not config['data']['balanced_data'] else ''
    except:
        pass       
    
    
    
    data_generation_distrib_str = ''
    #print(config['data']['distribution_list'])
    #print(config['data']['distribution_list'][0])
    #print([string for string in config['data']['distribution_list'][0]])
    try:
        if config['data']['distrib_by_feature']:
            data_generation_distrib_str = '-'.join([string[:2] for string in config['data']['distribution_list'][0]]) if 'distribution' in config['data']['function_generation_type'] else ''
        else: 
            data_generation_distrib_str = '-'.join([string[:2] for string in config['data']['distribution_list']]) if 'distribution' in config['data']['function_generation_type'] else ''     
    except:
        data_generation_distrib_str = '-'.join([string[:2] for string in config['data']['distribution_list']]) if 'distribution' in config['data']['function_generation_type'] else ''     

    
    
    dt_str = (
              '_depth' + str(maximum_depth) +
              '_beta' + str(config['function_family']['beta']) +
              '_decisionSpars' +  str(decision_sparsity) + 
              '_' + str(dt_type) +
              '_' + ('fullyGrown' if config['function_family']['fully_grown'] else 'partiallyGrown')
             ) if config['data']['function_generation_type'] != 'make_classification' else ''

    data_specification_string = (
                                  '_var' + str(config['data']['number_of_variables']) +
                                  '_class' + str(config['data']['num_classes']) +
                                  '_' + str(config['data']['function_generation_type']) +
                                  #'_' + str(config['data']['objective']) +
                                  '_xMax' + str(config['data']['x_max']) +
                                  '_xMin' + str(config['data']['x_min']) +
                                  '_xDist' + str(config['data']['x_distrib']) +
                                  data_noise +
                                  categorical_sting +
                                  random_parameters_distribution_string + 
                                  max_distributions_per_class_string +       
                                  distrib_param_max_str +
                                  fixed_class_probability_str +
                                  data_generation_shift_str +
                                  weighted_data_generation_str +
                                  data_generation_filtering_str +
                                  data_generation_linearly_separable_str +
                                  data_generation_shift_str + 
                                  data_generation_distrib_str + 
                                  dt_str
                                 )

    if path_type == 'data_creation' or path_type == 'lambda_net' or path_type == 'interpretation_net': #Data Generation
  
        path_identifier_function_data = ('lNetSize' + str(config['data']['lambda_dataset_size']) +
                                         '_numDatasets' + str(config['data']['number_of_generated_datasets']) +
                                         data_specification_string)            

        paths_dict['path_identifier_function_data'] = path_identifier_function_data
        
    if path_type == 'lambda_net' or path_type == 'interpretation_net': #Lambda-Net
            
        
            
        lambda_layer_str = '-'.join([str(neurons) for neurons in config['lambda_net']['lambda_network_layers']])
        
        early_stopping_string = 'ES' + str(config['lambda_net']['early_stopping_min_delta_lambda']) if config['lambda_net']['early_stopping_lambda'] else ''
        lambda_init_string = 'noFixedInit' if config['lambda_net']['number_initializations_lambda'] == -1 else 'fixedInit' + str(config['lambda_net']['number_initializations_lambda']) + '-seed' + str(config['computation']['RANDOM_SEED'])
        lambda_noise_string = '_noise-' + config['data']['noise_injected_type'] + str(config['data']['noise_injected_level']) if config['data']['noise_injected_level'] > 0 else ''
        
        lambda_batchnorm_str = ''
        try:
            lambda_batchnorm_str = '_batchnorm' if config['lambda_net']['use_batchnorm_lambda'] else ''
        except:
            pass  
    
        
        lambda_net_identifier = (
                                 lambda_layer_str + 
                                 '_e' + str(config['lambda_net']['epochs_lambda']) + early_stopping_string + 
                                 '_b' + str(config['lambda_net']['batch_lambda']) + 
                                 '_drop' + str(config['lambda_net']['dropout_lambda']) + 
                                 '_' + config['lambda_net']['optimizer_lambda'] + 
                                 '_' + config['lambda_net']['loss_lambda'] +
                                 '_' + lambda_init_string + 
                                 lambda_batchnorm_str +
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
            
        if config['i_net']['additional_hidden']:
            interpretation_network_layers_string += '_addHidden'
            
        function_representation_type_string = '_funcRep' + str(config['i_net']['function_representation_type'])
        
        data_reshape_version_string = '_reshape'+ str(config['i_net']['data_reshape_version'])
        
        interpretation_net_identifier = '_' + interpretation_network_layers_string + '_drop' + '-'.join([str(dropout) for dropout in config['i_net']['dropout']]) + 'e' + str(config['i_net']['epochs']) + 'b' + str(config['i_net']['batch_size']) + '_' + config['i_net']['optimizer'] + function_representation_type_string + data_reshape_version_string
        
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
        os.makedirs('./data/logging/' + paths_dict['path_identifier_interpretation_net'] + '/')
        os.makedirs('./data/plotting/' + paths_dict['path_identifier_interpretation_net'] + '/')
        os.makedirs('./data/results/' + paths_dict['path_identifier_interpretation_net'] + '/')
    except FileExistsError:
        pass
    

def generate_directory_structure():
    
    directory_names = ['parameters', 'plotting', 'saved_function_lists', 'results', 'saved_models', 'weights', 'weights_training', 'logging']
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
    
    
def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)
    
######################################################################################################################################################################################################################
########################################################################################  RANDOM FUNCTION GENERATION FROM ############################################################################################ 
######################################################################################################################################################################################################################

def plot_tree_from_parameters(paramerter_array, config):
    
    if config['function_family']['dt_type'] == 'SDT':
        some_tree = generate_decision_tree_from_array(paramerter_array, config)
        return some_tree.plot_tree()
    
    elif config['function_family']['dt_type'] == 'vanilla':
        image, nodes = anytree_decision_tree_from_parameters(paramerter_array, config=config)
        return image

def get_parameters_from_sklearn_decision_tree(tree, config, printing=False):
    
    from sklearn.tree import plot_tree
    from sklearn.tree import _tree
    from math import log2
    import queue
    
    if printing:
        plt.figure(figsize=(24,12))  # set plot size (denoted in inches)
        plot_tree(tree, fontsize=12)
        plt.show()
     
    def level_to_pre(arr,ind,new_arr):
        if ind>=len(arr): return new_arr #nodes at ind don't exist
        new_arr.append(arr[ind]) #append to back of the array
        new_arr = level_to_pre(arr,ind*2+1,new_arr) #recursive call to left
        new_arr = level_to_pre(arr,ind*2+2,new_arr) #recursive call to right
        return new_arr    
    
    def pre_to_level(arr):
        def left_tree_size(n):
            if n<=1: return 0
            l = int(log2(n+1)) #l = no of completely filled levels
            ans = 2**(l-1)
            last_level_nodes = min(n-2**l+1,ans)
            return ans + last_level_nodes -1       

        que = queue.Queue()
        que.put((0,len(arr)))
        ans = [] #this will be answer
        while not que.empty():
            iroot,size = que.get() #index of root and size of subtree
            if iroot>=len(arr) or size==0: continue ##nodes at iroot don't exist
            else : ans.append(arr[iroot]) #append to back of output array
            sz_of_left = left_tree_size(size) 
            que.put((iroot+1,sz_of_left)) #insert left sub-tree info to que
            que.put((iroot+1+sz_of_left,size-sz_of_left-1)) #right sub-tree info 

        return ans     
    
    features = [i for i in range(config['data']['number_of_variables'])]
    
    tree_ = tree.tree_
    features = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    if printing:
        print(features)
    
    split_values = []
    split_features = []
    
    leaf_probabilities = []
    
    def recurse(node, depth):        
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = features[node]
            threshold = tree_.threshold[node]
            split_values.append(threshold)
            split_features.append(feature)
                       
            if printing:
                print("{}if {} <= {}:".format(indent, feature, threshold))
            recurse(tree_.children_left[node], depth + 1)
            if printing:
                print("{}else:  # if {} > {}".format(indent, feature, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            
            missing_depth_subtree = config['function_family']['maximum_depth'] - (depth-1) 
            
            if missing_depth_subtree > 0:
                internal_node_num_ = 2 ** missing_depth_subtree - 1
                leaf_node_num_ = 2 ** missing_depth_subtree 
                
                subtree_internal_nodes = [-1 for i in range(internal_node_num_)]
                
                class_distribution = tree_.value[node][0]
                subtree_leaf_nodes = [class_distribution[0]/(class_distribution[0] + class_distribution[1]) for i in range(leaf_node_num_)]
                
                subtree_level = np.hstack([subtree_internal_nodes, subtree_leaf_nodes])
                subtree_pre = np.array(level_to_pre(subtree_level, 0, []))
                
                subtree_internal_nodes_level = subtree_pre[subtree_pre == -1]
                subtree_leaf_nodes_level = subtree_pre[subtree_pre != -1]
                           
                #split_features.extend(subtree_level)
                split_features.extend(subtree_internal_nodes_level)
                split_values.extend(subtree_internal_nodes_level)
                leaf_probabilities.extend(subtree_leaf_nodes_level)  

            else:       
                class_distribution = tree_.value[node][0]
                leaf_probabilities.append(class_distribution[0]/(class_distribution[0] + class_distribution[1]))
                #leaf_probabilities.append(class_distribution)
                            
            if printing:
                print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
    split_values = pre_to_level(split_values)
    split_features = pre_to_level(split_features)
    
    parameter_array = np.hstack([split_values, split_features, leaf_probabilities])
    
    return parameter_array




def get_shaped_parameters_for_decision_tree(flat_parameters, config, eager_execution=False):
    
    input_dim = config['data']['number_of_variables']
    output_dim = config['data']['num_classes']
    internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
    leaf_node_num_ = 2 ** config['function_family']['maximum_depth']
    
    if 'i_net' not in config.keys():
        config['i_net'] = {'function_representation_type': 1}

    if config['i_net']['function_representation_type'] == 1:
        if config['function_family']['dt_type'] == 'SDT':
                   

            weights = flat_parameters[:input_dim*internal_node_num_]
            weights = tf.reshape(weights, (internal_node_num_, input_dim))

            
            #print("config['data']['number_of_variables'], config['function_family']['decision_sparsity']", config['data']['number_of_variables'], config['function_family']['decision_sparsity'])
            if config['function_family']['decision_sparsity'] != -1 and config['function_family']['decision_sparsity'] !=  config['data']['number_of_variables']:
                vals_list, idx_list = tf.nn.top_k(tf.abs(weights), k=config['function_family']['decision_sparsity'], sorted=False)
                idx_list = tf.cast(idx_list, tf.int64)

                sparse_index_list = []
                for i, idx in enumerate(tf.unstack(idx_list)):
                    #idx = tf.sort(idx, direction='ASCENDING')
                    for ind in tf.unstack(idx):
                        sparse_index = tf.stack([tf.constant(i, dtype=tf.int64), ind], axis=0)
                        sparse_index_list.append(sparse_index)
                sparse_index_list = tf.stack(sparse_index_list)

                sparse_tensor = tf.sparse.SparseTensor(indices=sparse_index_list, values=tf.squeeze(tf.reshape(vals_list, (1, -1))), dense_shape=weights.shape)
                if eager_execution:
                    dense_tensor = tf.sparse.to_dense(tf.sparse.reorder(sparse_tensor))
                else:
                    dense_tensor = tf.sparse.to_dense(sparse_tensor)#tf.sparse.to_dense(tf.sparse.reorder(sparse_tensor))
                weights = dense_tensor

            biases = flat_parameters[input_dim*internal_node_num_:(input_dim+1)*internal_node_num_]

            leaf_probabilities = flat_parameters[(input_dim+1)*internal_node_num_:]
            leaf_probabilities = tf.transpose(tf.reshape(leaf_probabilities, (leaf_node_num_, output_dim)))
            
            #tf.print(weights, biases, leaf_probabilities)
            
            return weights, biases, leaf_probabilities
            
        elif config['function_family']['dt_type'] == 'vanilla':
            splits_coeff = flat_parameters[:config['function_family']['decision_sparsity']*internal_node_num_]
            splits_coeff = tf.clip_by_value(splits_coeff, clip_value_min=config['data']['x_min'], clip_value_max=config['data']['x_max'])
            splits_coeff_list = tf.split(splits_coeff, internal_node_num_)
            splits_index = tf.cast(tf.clip_by_value(tf.round(flat_parameters[config['function_family']['decision_sparsity']*internal_node_num_:(config['function_family']['decision_sparsity']*internal_node_num_)*2]), clip_value_min=0, clip_value_max=config['data']['number_of_variables']-1), tf.int64)
            splits_index_list = tf.split(splits_index, internal_node_num_)
            
            splits_list = []
            for values_node, indices_node in zip(splits_coeff_list, splits_index_list):
                sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(indices_node, axis=1), values=values_node, dense_shape=[input_dim])
                dense_tensor = tf.sparse.to_dense(sparse_tensor)
                splits_list.append(dense_tensor)             
            
            splits = tf.stack(splits_list)            
            
            
            leaf_classes = flat_parameters[(config['function_family']['decision_sparsity']*internal_node_num_)*2:]  
            leaf_classes = tf.clip_by_value(leaf_classes, clip_value_min=0, clip_value_max=1)
            #tf.print(splits, leaf_classes)
            
            return splits, leaf_classes
        
    elif config['i_net']['function_representation_type'] == 2:
        if config['function_family']['dt_type'] == 'SDT':
            weights_coeff = flat_parameters[:internal_node_num_ * config['function_family']['decision_sparsity']]

            weights_index_array = flat_parameters[internal_node_num_ * config['function_family']['decision_sparsity']:(internal_node_num_ * config['function_family']['decision_sparsity'])+(internal_node_num_ * config['function_family']['decision_sparsity'] * config['data']['number_of_variables'])]
            
            biases = flat_parameters[(internal_node_num_ * config['function_family']['decision_sparsity'])+(internal_node_num_ * config['function_family']['decision_sparsity'] * config['data']['number_of_variables']):(internal_node_num_ * config['function_family']['decision_sparsity'])+(internal_node_num_ * config['function_family']['decision_sparsity'] * config['data']['number_of_variables']) + internal_node_num_]
            
            leaf_probabilities = flat_parameters[(internal_node_num_ * config['function_family']['decision_sparsity'])+(internal_node_num_ * config['function_family']['decision_sparsity'] * config['data']['number_of_variables']) + internal_node_num_:]
            leaf_probabilities = tf.transpose(tf.reshape(leaf_probabilities, (leaf_node_num_, output_dim)))

            weights_coeff_list_by_internal_node = tf.split(weights_coeff, internal_node_num_)


            weights_index_list_by_internal_node = tf.split(weights_index_array, internal_node_num_)
            weights_index_list_by_internal_node_by_decision_sparsity = []
            for tensor in weights_index_list_by_internal_node:
                weights_index_list_by_internal_node_by_decision_sparsity.append(tf.split(tensor, config['function_family']['decision_sparsity']))
            weights_index_list_by_internal_node_by_decision_sparsity_argmax = tf.split(tf.argmax(weights_index_list_by_internal_node_by_decision_sparsity, axis=2), internal_node_num_)
            weights_index_list_by_internal_node_by_decision_sparsity_argmax_new = []
            for tensor in weights_index_list_by_internal_node_by_decision_sparsity_argmax:
                weights_index_list_by_internal_node_by_decision_sparsity_argmax_new.append(tf.squeeze(tensor, axis=0))
            weights_index_list_by_internal_node_by_decision_sparsity_argmax = weights_index_list_by_internal_node_by_decision_sparsity_argmax_new
            dense_tensor_list = []
                                    
            if False: #duplicates in predicted index not considered/summarized
                for indices_node, values_node in zip(weights_index_list_by_internal_node_by_decision_sparsity_argmax,  weights_coeff_list_by_internal_node):
                    sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(indices_node, axis=1), values=values_node, dense_shape=[input_dim])
                    if eager_execution == True:
                        dense_tensor = tf.sparse.to_dense(tf.sparse.reorder(sparse_tensor))
                    else:
                        dense_tensor = tf.sparse.to_dense(sparse_tensor)
                    dense_tensor_list.append(dense_tensor)
            else:
                dense_tensor_list = []
                for indices_node, values_node in zip(weights_index_list_by_internal_node_by_decision_sparsity_argmax,  weights_coeff_list_by_internal_node):
                    if False:
                        dense_tensor = []
                        for i in range(config['data']['number_of_variables']):
                            index_identifier = tf.where(tf.equal(indices_node, i))
                            values_by_index = tf.reduce_sum(tf.gather_nd(values_node, index_identifier))
                            dense_tensor.append(values_by_index)
                    else:
                        dense_tensor = []#[0 for _ in range(config['data']['number_of_variables'])]
                        for i in range(config['data']['number_of_variables']):
                            index_identifier = []
                            for j, variable_index in enumerate(tf.unstack(indices_node)):
                                if tf.equal(variable_index, i):
                                    index_identifier.append(j)
                            index_identifier = tf.cast(tf.expand_dims(tf.stack(index_identifier), axis=1), tf.int64)
                            values_by_index = tf.reduce_sum(tf.gather_nd(values_node, index_identifier))
                            dense_tensor.append(values_by_index)
                    dense_tensor = tf.stack(dense_tensor)
                    dense_tensor_list.append(dense_tensor)

            weights = tf.stack(dense_tensor_list) 
            #tf.print(weights, biases, leaf_probabilities)
            return weights, biases, leaf_probabilities
        
        elif config['function_family']['dt_type'] == 'vanilla':
            split_values_num_params = internal_node_num_ * config['function_family']['decision_sparsity']
            split_index_num_params = config['data']['number_of_variables'] *  config['function_family']['decision_sparsity'] * internal_node_num_
            leaf_classes_num_params = leaf_node_num_ #* config['data']['num_classes']

            split_values = flat_parameters[:split_values_num_params]
            split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

            split_index_array = flat_parameters[split_values_num_params:split_values_num_params+split_index_num_params]    
            split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)
            split_index_list_by_internal_node_by_decision_sparsity = []
            for tensor in split_index_list_by_internal_node:
                split_tensor = tf.split(tensor, config['function_family']['decision_sparsity'])
                split_index_list_by_internal_node_by_decision_sparsity.append(split_tensor)
            split_index_list_by_internal_node_by_decision_sparsity_argmax = tf.split(tf.argmax(split_index_list_by_internal_node_by_decision_sparsity, axis=2), internal_node_num_)
            split_index_list_by_internal_node_by_decision_sparsity_argmax_new = []
            for tensor in split_index_list_by_internal_node_by_decision_sparsity_argmax:
                tensor_squeeze = tf.squeeze(tensor, axis=0)
                split_index_list_by_internal_node_by_decision_sparsity_argmax_new.append(tensor_squeeze)
            split_index_list_by_internal_node_by_decision_sparsity_argmax = split_index_list_by_internal_node_by_decision_sparsity_argmax_new    
            dense_tensor_list = []
            for indices_node, values_node in zip(split_index_list_by_internal_node_by_decision_sparsity_argmax,  split_values_list_by_internal_node):
                sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(indices_node, axis=1), values=values_node, dense_shape=[input_dim])
                dense_tensor = tf.sparse.to_dense(sparse_tensor)
                dense_tensor_list.append(dense_tensor) 
            splits = tf.stack(dense_tensor_list)

            leaf_classes_array = flat_parameters[split_values_num_params+split_index_num_params:]  
            split_index_list_by_leaf_node = tf.split(leaf_classes_array, leaf_node_num_)
            #leaf_classes_list = []
            #for tensor in split_index_list_by_leaf_node:
                #argmax = tf.argmax(tensor)
                #argsort = tf.argsort(tensor, direction='DESCENDING')
                #leaf_classes_list.append(argsort[0])
                #leaf_classes_list.append(argsort[1])

            leaf_classes = tf.squeeze(tf.stack(split_index_list_by_leaf_node))#tf.stack(leaf_classes_list)
            
            #tf.print(splits, leaf_classes)
            return splits, leaf_classes

    elif config['i_net']['function_representation_type'] >= 3:
        if config['function_family']['dt_type'] == 'SDT':    
            
            split_values_num_params = config['data']['number_of_variables'] * internal_node_num_#config['function_family']['decision_sparsity']
            split_index_num_params = config['data']['number_of_variables'] * internal_node_num_
            leaf_classes_num_params = leaf_node_num_ #* config['data']['num_classes']

            split_values = flat_parameters[:split_values_num_params]
            split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

            split_index_array = flat_parameters[split_values_num_params:split_values_num_params+split_index_num_params]    
            split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)         
            
            
            biases = flat_parameters[split_values_num_params+split_index_num_params:split_values_num_params+split_index_num_params+internal_node_num_]
            
            split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(split_index_list_by_internal_node)
            
            weights = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))
            
            leaf_probabilities = flat_parameters[split_values_num_params+split_index_num_params+internal_node_num_:]
            leaf_probabilities = tf.transpose(tf.reshape(leaf_probabilities, (leaf_node_num_, output_dim)))
            
            return weights, biases, leaf_probabilities
        elif config['function_family']['dt_type'] == 'vanilla':    
            split_values_num_params = config['data']['number_of_variables'] * internal_node_num_#config['function_family']['decision_sparsity']
            split_index_num_params = config['data']['number_of_variables'] * internal_node_num_
            leaf_classes_num_params = leaf_node_num_ #* config['data']['num_classes']

            split_values = flat_parameters[:split_values_num_params]
            split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

            split_index_array = flat_parameters[split_values_num_params:split_values_num_params+split_index_num_params]    
            split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)         
            
            split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(split_index_list_by_internal_node)
            
            splits = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))
            
            leaf_classes_array = flat_parameters[split_values_num_params+split_index_num_params:]  
            split_index_list_by_leaf_node = tf.split(leaf_classes_array, leaf_node_num_)

            leaf_classes = tf.squeeze(tf.stack(split_index_list_by_leaf_node))
            
            return splits, leaf_classes
        
    return None

    

def generate_decision_tree_from_array(parameter_array, config):
    
    from utilities.DecisionTree_BASIC import SDT
    
    if config['function_family']['dt_type'] == 'SDT': 
        
        tree = SDT(input_dim=config['data']['number_of_variables'],
                   output_dim=config['data']['num_classes'],
                   depth=config['function_family']['maximum_depth'],
                   beta=config['function_family']['beta'],
                   decision_sparsity=config['function_family']['decision_sparsity'],
                   use_cuda=False,
                   verbosity=0)

        tree.initialize_from_parameter_array(parameter_array)
        
    elif config['function_family']['dt_type'] == 'vanilla': 
        #raise SystemExit('Cant inizialize vanilla DT') 
        return None
    
    return tree




def generate_random_data_points_custom(low, 
                                       high, 
                                       size, 
                                       variables, 
                                       categorical_indices=None, 
                                       seed=None, 
                                       distrib=None, 
                                       random_parameters=False, 
                                       parameters=None, 
                                       distrib_param_max=1, 
                                       config=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    if isinstance(distrib, list):
        distributions_per_class = config['data']['max_distributions_per_class']
        if True:
            list_of_data_points, _, _, _ = generate_dataset_from_distributions(distribution_list = [distrib], 
                                                    number_of_variables = variables, 
                                                    number_of_samples = size, 
                                                    distributions_per_class =  distributions_per_class, 
                                                    seed = seed, 
                                                    flip_percentage=0.0, 
                                                    data_noise=0.0, 
                                                    random_parameters=True, 
                                                    #distribution_dict_list=None,
                                                    config=config)      


        else:

            list_of_data_points = []
            
            for j in range(variables):
                distrib_by_variable = np.random.choice(distrib)
                if parameters == None:
                    value_1 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)
                    value_2 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2) 

                    parameter_by_distribution = {
                        'normal': {
                            'loc': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },
                        'uniform': {
                            'low': np.minimum(value_1, value_2),
                            'high': np.maximum(value_1, value_2),
                        },
                        'gamma': {
                            'shape': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },        
                        'exponential': {
                            'scale': np.random.uniform(0, distrib_param_max),
                        },        
                        'beta': {
                            'a': np.random.uniform(0, distrib_param_max),
                            'b': np.random.uniform(0, distrib_param_max),
                        },
                        'binomial': {
                            'n': 100,
                            'p': np.random.uniform(0, 1),
                        },
                        'poisson': {
                            'lam': np.random.uniform(0, distrib_param_max),
                        },      
                        'lognormal': {
                            'mean': np.random.uniform(0, distrib_param_max),
                            'sigma': np.random.uniform(0, distrib_param_max),
                        },             
                        'f': {
                            'dfnum': np.random.uniform(0, distrib_param_max),
                            'dfden': np.random.uniform(0, distrib_param_max),
                        },
                        'logistic': {
                            'loc': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },
                        'weibull': {
                            'a': np.random.uniform(0, distrib_param_max),
                        },  
                        'standarduniform': {
                            'low': 0,
                            'high': 1,                            
                        }, 
                        'standardnormal': {
                            'loc': 0,
                            'scale': 1,
                        }, 
                    }     
                else:
                    parameter_by_distribution = {
                        distrib: parameters[j]
                    }             

                list_of_data_points_variable = None 


                if random_parameters == True and parameters is None:

                    list_of_data_points_variable, _ = get_distribution_data_from_string(distribution_name=distrib_by_variable, size=size, seed=seed+j, random_parameters=random_parameters, distrib_param_max=distrib_param_max)      

                elif distrib_by_variable == 'uniform':
                    list_of_data_points_variable = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=size)
                elif distrib_by_variable == 'normal':
                    list_of_data_points_variable = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=size) 
                elif distrib_by_variable == 'gamma':
                    list_of_data_points_variable = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=size)
                elif distrib_by_variable == 'exponential':
                    list_of_data_points_variable = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=(size, variables))
                elif distrib_by_variable == 'beta':
                    list_of_data_points_variable = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=size)
                elif distrib_by_variable == 'binomial':
                    list_of_data_points_variable = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=size)       
                elif distrib_by_variable == 'poisson':
                    list_of_data_points_variable = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=size)
                elif distrib_by_variable == 'lognormal':
                    list_of_data_points_variable = np.random.lognormal(parameter_by_distribution['lognormal']['mean'],parameter_by_distribution['lognormal']['sigma'], size=size)
                elif distrib_by_variable == 'f':
                    list_of_data_points_variable = np.random.f(parameter_by_distribution['f']['dfnum'], parameter_by_distribution['f']['dfden'], size=size)
                elif distrib_by_variable == 'logistic':
                    list_of_data_points_variable = np.random.logistic(parameter_by_distribution['logistic']['loc'], parameter_by_distribution['logistic']['scale'], size=size)
                elif distrib_by_variable == 'weibull':
                    list_of_data_points_variable = np.random.weibull(parameter_by_distribution['weibull']['a'], size=size)
                elif distrib_by_variable == 'standarduniform':
                    list_of_data_points_variable = np.random.uniform(parameter_by_distribution['standarduniform']['low'], parameter_by_distribution['standarduniform']['high'], size=size)
                elif distrib_by_variable == 'standardnormal':
                    list_of_data_points_variable = np.random.normal(parameter_by_distribution['standardnormal']['loc'], parameter_by_distribution['standardnormal']['scale'], size=size) 
                    
                list_of_data_points.append(list_of_data_points_variable)

            list_of_data_points = np.vstack(list_of_data_points).T#np.hstack(list_of_data_points)

            list_of_data_points = np.nan_to_num(list_of_data_points)
            
            list_of_data_points_scaled = []
            for i, column in enumerate(list_of_data_points.T):
                scaler = MinMaxScaler(feature_range=(low, high))
                scaler.fit(column.reshape(-1, 1))
                #list_of_data_points[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
                list_of_data_points_scaled.append(scaler.transform(column.reshape(-1, 1)).ravel())
            list_of_data_points = np.array(list_of_data_points_scaled).T


            if categorical_indices is not None:
                for categorical_index in categorical_indices:
                    list_of_data_points[:,categorical_index] = np.round(list_of_data_points[:,categorical_index])                 


        
    else:

        if parameters == None:
            value_1 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)
            value_2 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2) 

            parameter_by_distribution = {
                'normal': {
                    'loc': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },
                'uniform': {
                    'low': np.minimum(value_1, value_2),
                    'high': np.maximum(value_1, value_2),
                },
                'gamma': {
                    'shape': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },        
                'exponential': {
                    'scale': np.random.uniform(0, distrib_param_max),
                },        
                'beta': {
                    'a': np.random.uniform(0, distrib_param_max),
                    'b': np.random.uniform(0, distrib_param_max),
                },
                'binomial': {
                    'n': 100,
                    'p': np.random.uniform(0, 1),
                },
                'poisson': {
                    'lam': np.random.uniform(0, distrib_param_max),
                },      
                'lognormal': {
                    'mean': np.random.uniform(0, distrib_param_max),
                    'sigma': np.random.uniform(0, distrib_param_max),
                },             
                'f': {
                    'dfnum': np.random.uniform(0, distrib_param_max),
                    'dfden': np.random.uniform(0, distrib_param_max),
                },
                'logistic': {
                    'loc': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },
                'weibull': {
                    'a': np.random.uniform(0, distrib_param_max),
                },             

            }     
        else:
            parameter_by_distribution = {
                distrib: parameters
            }             

        list_of_data_points = None 


        if random_parameters == True and parameters is None:
            list_of_data_points, _ = get_distribution_data_from_string(distribution_name=distrib, size=(size, variables), seed=seed, random_parameters=random_parameters, distrib_param_max=distrib_param_max)        
        elif distrib == 'uniform':
            list_of_data_points = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=(size, variables))
        elif distrib == 'normal':
            list_of_data_points = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=(size, variables)) 
        elif distrib == 'gamma':
            list_of_data_points = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=(size, variables))
        elif distrib == 'exponential':
            list_of_data_points = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=(size, variables))
        elif distrib == 'beta':
            list_of_data_points = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=(size, variables))
        elif distrib == 'binomial':
            list_of_data_points = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=(size, variables))       
        elif distrib == 'poisson':
            list_of_data_points = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=(size, variables))
        elif distrib == 'lognormal':
            list_of_data_points = np.random.lognormal(parameter_by_distribution['lognormal']['mean'],parameter_by_distribution['lognormal']['sigma'], size=(size, variables))
        elif distrib == 'f':
            list_of_data_points = np.random.f(parameter_by_distribution['f']['dfnum'], parameter_by_distribution['f']['dfden'], size=(size, variables))
        elif distrib == 'logistic':
            list_of_data_points = np.random.logistic(parameter_by_distribution['logistic']['loc'], parameter_by_distribution['logistic']['scale'], size=(size, variables))
        elif distrib == 'weibull':
            list_of_data_points = np.random.weibull(parameter_by_distribution['weibull']['a'], size=(size, variables))

        list_of_data_points = np.nan_to_num(list_of_data_points)

        list_of_data_points_scaled = []
        for i, column in enumerate(list_of_data_points.T):
            scaler = MinMaxScaler(feature_range=(low, high))
            scaler.fit(column.reshape(-1, 1))
            #list_of_data_points[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
            list_of_data_points_scaled.append(scaler.transform(column.reshape(-1, 1)).ravel())
        list_of_data_points = np.array(list_of_data_points_scaled).T


        if categorical_indices is not None:
            for categorical_index in categorical_indices:
                list_of_data_points[:,categorical_index] = np.round(list_of_data_points[:,categorical_index])       
        
    return list_of_data_points



def generate_random_data_points(config, seed, parameters=None):
            
    low = config['data']['x_min'] 
    high = config['data']['x_max']
    size = config['data']['lambda_dataset_size'] 
    variables = config['data']['number_of_variables'] 
    categorical_indices = config['data']['categorical_indices']
    distrib=config['data']['x_distrib']
    #random_parameters = config['data']['random_parameters_trained']
    
    random_parameters=False
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
        
    if isinstance(distrib, list):
        if True:
            list_of_data_points, _, _, _ = generate_dataset_from_distributions(distribution_list = [distrib], 
                                                    number_of_variables = variables, 
                                                    number_of_samples = size, 
                                                    distributions_per_class =  config['data']['max_distributions_per_class'], 
                                                    seed = seed, 
                                                    flip_percentage=0.0, 
                                                    data_noise=0.0, 
                                                    random_parameters=True, 
                                                    #distribution_dict_list=None,
                                                    config=config)                
            
        else:
            list_of_data_points = []
            for j in range(variables):
                distribution_by_variable = np.random.choice(distrib)

                try:
                    distrib_param_max = config['data']['distrib_param_max']
                except:
                    distrib_param_max = 1          

                if parameters == None:
                    value_1 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)
                    value_2 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)         
                    parameter_by_distribution = {
                        'normal': {
                            'loc': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },
                        'uniform': {
                            'low': np.minimum(value_1, value_2),
                            'high': np.maximum(value_1, value_2),
                        },
                        'gamma': {
                            'shape': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },        
                        'exponential': {
                            'scale': np.random.uniform(0, distrib_param_max),
                        },        
                        'beta': {
                            'a': np.random.uniform(0, distrib_param_max),
                            'b': np.random.uniform(0, distrib_param_max),
                        },
                        'binomial': {
                            'n': 100,
                            'p': np.random.uniform(0, 1),
                        },
                        'poisson': {
                            'lam': np.random.uniform(0, distrib_param_max),
                        },
                        'lognormal': {
                            'mean': np.random.uniform(0, distrib_param_max),
                            'sigma': np.random.uniform(0, distrib_param_max),
                        },             
                        'f': {
                            'dfnum': np.random.uniform(0, distrib_param_max),
                            'dfden': np.random.uniform(0, distrib_param_max),
                        },
                        'logistic': {
                            'loc': np.random.uniform(0, distrib_param_max),
                            'scale': np.random.uniform(0, distrib_param_max),
                        },
                        'weibull': {
                            'a': np.random.uniform(0, distrib_param_max),
                        },        

                    }           
                else:
                    parameter_by_distribution = {
                        distrib: parameters[j]
                    }

                list_of_data_points_variable = None

                if random_parameters == True and parameters is None:
                    list_of_data_points_variable, _ = get_distribution_data_from_string(distribution_name=distribution_by_variable, size=size, seed=seed+j, random_parameters=random_parameters, distrib_param_max=distrib_param_max)        
                elif distribution_by_variable == 'uniform':
                    list_of_data_points_variable = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=size)
                elif distribution_by_variable == 'normal':
                    list_of_data_points_variable = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=size) 
                elif distribution_by_variable == 'gamma':
                    list_of_data_points_variable = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=size)
                elif distribution_by_variable == 'exponential':
                    list_of_data_points_variable = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=size)
                elif distribution_by_variable == 'beta':
                    list_of_data_points_variable = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=size)
                elif distribution_by_variable == 'binomial':
                    list_of_data_points_variable = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=size)       
                elif distribution_by_variable == 'poisson':
                    list_of_data_points_variable = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=size)
                elif distribution_by_variable == 'lognormal':
                    list_of_data_points_variable = np.random.lognormal(parameter_by_distribution['lognormal']['mean'],parameter_by_distribution['lognormal']['sigma'], size=size)
                elif distribution_by_variable == 'f':
                    list_of_data_points_variable = np.random.f(parameter_by_distribution['f']['dfnum'], parameter_by_distribution['f']['dfden'], size=size)
                elif distribution_by_variable == 'logistic':
                    list_of_data_points_variable = np.random.logistic(parameter_by_distribution['logistic']['loc'], parameter_by_distribution['logistic']['scale'], size=size)
                elif distribution_by_variable == 'weibull':
                    list_of_data_points_variable = np.random.weibull(parameter_by_distribution['weibull']['a'], size=size)

                list_of_data_points.append(list_of_data_points_variable)

            list_of_data_points = np.vstack(list_of_data_points).T#np.hstack(list_of_data_points) 

            list_of_data_points = np.nan_to_num(list_of_data_points)

            list_of_data_points_scaled = []
            for i, column in enumerate(list_of_data_points.T):
                scaler = MinMaxScaler(feature_range=(low, high))
                scaler.fit(column.reshape(-1, 1))
                #list_of_data_points[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
                list_of_data_points_scaled.append(scaler.transform(column.reshape(-1, 1)).ravel())
            list_of_data_points = np.array(list_of_data_points_scaled).T


            if categorical_indices is not None:
                for categorical_index in categorical_indices:
                    list_of_data_points[:,categorical_index] = np.round(list_of_data_points[:,categorical_index])                   





    
    
    else:


        try:
            distrib_param_max = config['data']['distrib_param_max']
        except:
            distrib_param_max = 1          

        if parameters == None:
            value_1 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)
            value_2 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)         
            parameter_by_distribution = {
                'normal': {
                    'loc': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },
                'uniform': {
                    'low': np.minimum(value_1, value_2),
                    'high': np.maximum(value_1, value_2),
                },
                'gamma': {
                    'shape': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },        
                'exponential': {
                    'scale': np.random.uniform(0, distrib_param_max),
                },        
                'beta': {
                    'a': np.random.uniform(0, distrib_param_max),
                    'b': np.random.uniform(0, distrib_param_max),
                },
                'binomial': {
                    'n': 100,
                    'p': np.random.uniform(0, 1),
                },
                'poisson': {
                    'lam': np.random.uniform(0, distrib_param_max),
                },
                'lognormal': {
                    'mean': np.random.uniform(0, distrib_param_max),
                    'sigma': np.random.uniform(0, distrib_param_max),
                },             
                'f': {
                    'dfnum': np.random.uniform(0, distrib_param_max),
                    'dfden': np.random.uniform(0, distrib_param_max),
                },
                'logistic': {
                    'loc': np.random.uniform(0, distrib_param_max),
                    'scale': np.random.uniform(0, distrib_param_max),
                },
                'weibull': {
                    'a': np.random.uniform(0, distrib_param_max),
                },        

            }           
        else:
            parameter_by_distribution = {
                distrib: parameters
            }

        list_of_data_points = None

        if random_parameters == True and parameters is None:
            list_of_data_points, _ = get_distribution_data_from_string(distribution_name=distrib, size=(size, variables), seed=seed, random_parameters=random_parameters, distrib_param_max=distrib_param_max)        
        elif distrib == 'uniform':
            list_of_data_points = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=(size, variables))
        elif distrib == 'normal':
            list_of_data_points = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=(size, variables)) 
        elif distrib == 'gamma':
            list_of_data_points = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=(size, variables))
        elif distrib == 'exponential':
            list_of_data_points = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=(size, variables))
        elif distrib == 'beta':
            list_of_data_points = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=(size, variables))
        elif distrib == 'binomial':
            list_of_data_points = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=(size, variables))       
        elif distrib == 'poisson':
            list_of_data_points = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=(size, variables))
        elif distrib == 'lognormal':
            list_of_data_points = np.random.lognormal(parameter_by_distribution['lognormal']['mean'],parameter_by_distribution['lognormal']['sigma'], size=(size, variables))
        elif distrib == 'f':
            list_of_data_points = np.random.f(parameter_by_distribution['f']['dfnum'], parameter_by_distribution['f']['dfden'], size=(size, variables))
        elif distrib == 'logistic':
            list_of_data_points = np.random.logistic(parameter_by_distribution['logistic']['loc'], parameter_by_distribution['logistic']['scale'], size=(size, variables))
        elif distrib == 'weibull':
            list_of_data_points = np.random.weibull(parameter_by_distribution['weibull']['a'], size=(size, variables))

        list_of_data_points = np.nan_to_num(list_of_data_points)

        list_of_data_points_scaled = []
        for i, column in enumerate(list_of_data_points.T):
            scaler = MinMaxScaler(feature_range=(low, high))
            scaler.fit(column.reshape(-1, 1))
            #list_of_data_points[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
            list_of_data_points_scaled.append(scaler.transform(column.reshape(-1, 1)).ravel())
        list_of_data_points = np.array(list_of_data_points_scaled).T


        if categorical_indices is not None:
            for categorical_index in categorical_indices:
                list_of_data_points[:,categorical_index] = np.round(list_of_data_points[:,categorical_index])                   



        
    return list_of_data_points 


def generate_random_decision_tree(config, seed=42, verbosity=0):
    
    from utilities.DecisionTree_BASIC import SDT   
    from sklearn.tree import DecisionTreeClassifier
    #random.seed(seed)
    #np.random.seed(seed)
    
    if config['function_family']['dt_type'] == 'SDT':    
        if config['function_family']['fully_grown']:
            tree = SDT(input_dim=config['data']['number_of_variables'],#X_train.shape[1], 
                       output_dim=config['data']['num_classes'],#int(max(y_train))+1, 
                       depth=config['function_family']['maximum_depth'],
                       beta=config['function_family']['beta'],
                       decision_sparsity=config['function_family']['decision_sparsity'],
                       random_seed=seed,
                       use_cuda=False,
                       verbosity=verbosity)#
            
            return tree

        else: 
            raise SystemExit('Partially Grown Trees not implemented yet')
    elif config['function_family']['dt_type'] == 'vanilla': 
    
        tree = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
        
        return tree
    
    return None



def generate_data_random_decision_tree(config, seed=42):
    
    X_data = generate_random_data_points(config, seed)
    
    if config['function_family']['dt_type'] == 'SDT':    
        decision_tree = generate_random_decision_tree(config, seed)

        y_data = decision_tree.predict_proba(X_data)
        counter = 1

        while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
            seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
            counter += 1

            decision_tree = generate_random_decision_tree(config, seed)
            
            y_data = decision_tree.predict_proba(X_data)    #predict_proba #predict

        return decision_tree.to_array(), X_data, np.round(y_data), y_data 

    elif config['function_family']['dt_type'] == 'vanilla': 
        
        config_dt = deepcopy(config)
        config_dt['i_net'] = {'function_representation_type': 1}
        config_dt['i_net']['function_representation_type'] = 1
        config_dt['data']['categorical_indices'] = []        
        
        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth']  

        np.random.seed(seed)
        random.seed(seed)        
        inner_nodes_split_value = []
        for _ in range(internal_node_num_):
            random_number = np.random.uniform(0, 1)
            inner_nodes_split_value.append(random_number)      
            
        inner_nodes_split_feature = []
        for _ in range(internal_node_num_):
            random_number = np.random.randint(0, config['data']['number_of_variables'])
            inner_nodes_split_feature.append(random_number)  

        leaf_nodes = []
        for _ in range(leaf_node_num_):
            random_number = np.random.uniform(0, 1)
            leaf_nodes.append(random_number)        
        
        decision_tree = inner_nodes_split_value
        decision_tree.extend(inner_nodes_split_feature)
        decision_tree.extend(leaf_nodes)
        
        decision_tree = np.array(decision_tree)
        
        #print('decision_tree.shape', decision_tree.shape)
        
        y_data, _  = calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(X_data, config_dt)(decision_tree)
        y_data = y_data.numpy()
        
        counter = 1

        while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
            seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
            np.random.seed(seed)
            random.seed(seed)            
            
            inner_nodes_split_value = []
            for _ in range(internal_node_num_):
                random_number = np.random.uniform(0, 1)
                inner_nodes_split_value.append(random_number)      

            inner_nodes_split_feature = []
            for _ in range(internal_node_num_):
                random_number = np.random.randint(0, config['data']['number_of_variables'])
                inner_nodes_split_feature.append(random_number)  

            leaf_nodes = []
            for _ in range(leaf_node_num_):
                random_number = np.random.uniform(0, 1)
                leaf_nodes.append(random_number)        

            decision_tree = inner_nodes_split_value
            decision_tree.extend(inner_nodes_split_feature)
            decision_tree.extend(leaf_nodes)

            decision_tree = np.array(decision_tree)

            #print('decision_tree.shape', decision_tree.shape)

            y_data, _  = calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(X_data, config_dt)(decision_tree)
            y_data = y_data.numpy()
        
            
        return decision_tree, X_data, np.round(y_data), y_data 
    
    return None


def generate_data_random_decision_tree_trained(config, seed=42):
    
    decision_tree = generate_random_decision_tree(config, seed)

    X_data = generate_random_data_points(config, seed)

    np.random.seed(seed)
    random.seed(seed)
    y_data_tree = np.random.randint(0,2,X_data.shape[0])
        
    if config['function_family']['dt_type'] == 'SDT':    
        
        decision_tree.fit(X_data, y_data_tree, epochs=50)    
        
        y_data = decision_tree.predict_proba(X_data)
        counter = 1

        while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
            seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
            counter += 1

            decision_tree = generate_random_decision_tree(config, seed)
            y_data = decision_tree.predict_proba(X_data)    #predict_proba #predict

        return decision_tree.to_array(), X_data, np.round(y_data), y_data 

    elif config['function_family']['dt_type'] == 'vanilla': 

        decision_tree.fit(X_data, y_data_tree)    

        y_data = decision_tree.predict(X_data)
        counter = 1

        while np.unique(np.round(y_data)).shape[0] == 1 or np.min(np.unique(np.round(y_data), return_counts=True)[1]) < config['data']['lambda_dataset_size']/4:
            seed = seed+(config['data']['number_of_generated_datasets'] * counter)    
            counter += 1

            decision_tree = generate_random_decision_tree(config, seed)
            
            X_data = generate_random_data_points(config, seed)

            y_data_tree = np.random.randint(0,2,X_data.shape[0])  
            
            decision_tree.fit(X_data, y_data_tree)  
            
            y_data = decision_tree.predict(X_data)    #predict_proba #predict

        #placeholder = [0 for i in range((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables'] + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes'])]

        return get_parameters_from_sklearn_decision_tree(decision_tree, config), X_data, np.round(y_data), y_data 

    return None







def generate_data_make_classification_decision_tree_trained(config, seed=42):
           
    informative = config['data']['number_of_variables']#np.random.randint(config['data']['number_of_variables']//2, high=config['data']['number_of_variables']+1) #config['data']['number_of_variables']
    redundant = 0#np.random.randint(0, high=config['data']['number_of_variables']-informative+1) #0
    repeated = 0#config['data']['number_of_variables']-informative-redundant # 0

    n_clusters_per_class =  max(2, np.random.randint(0, high=informative//2+1)) #2

    X_data, y_data_tree = make_classification(n_samples=config['data']['lambda_dataset_size'], 
                                                       n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                       n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                       n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                       n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                       n_classes=config['data']['num_classes'], 
                                                       n_clusters_per_class=n_clusters_per_class, 
                                                       #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                       #class_sep=1.0, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                       #hypercube=False, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                       #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                       #scale=1.0, #Multiply features by the specified value. 
                                                       shuffle=True, 
                                                       random_state=seed)    

    for i, column in enumerate(X_data.T):
        scaler = MinMaxScaler()
        scaler.fit(column.reshape(-1, 1))
        X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
    
    if config['data']['categorical_indices'] is not None:
        for categorical_index in config['data']['categorical_indices']:
            X_data[:,categorical_index] = np.round(X_data[:,categorical_index])      
            
    decision_tree = generate_random_decision_tree(config, seed)
        
    if config['function_family']['dt_type'] == 'SDT':   
        decision_tree.fit(X_data, y_data_tree, epochs=50)    

        y_data = decision_tree.predict_proba(X_data)

        return decision_tree.to_array(), X_data, np.round(y_data), y_data     
    
    
    elif config['function_family']['dt_type'] == 'vanilla': 
        decision_tree.fit(X_data, y_data_tree)    

        y_data = decision_tree.predict(X_data)    

        #placeholder = [0 for i in range((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables'] + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes'])]

        return get_parameters_from_sklearn_decision_tree(decision_tree, config), X_data, np.round(y_data), y_data 
    
    return None


def generate_data_make_classification(config, seed=42):
            
    informative = config['data']['number_of_variables']#np.random.randint(config['data']['number_of_variables']//2, high=config['data']['number_of_variables']+1) #config['data']['number_of_variables']
    redundant = 0#np.random.randint(0, high=config['data']['number_of_variables']-informative+1) #0
    repeated = 0#config['data']['number_of_variables']-informative-redundant # 0

    n_clusters_per_class =  max(2, np.random.randint(0, high=informative//2+1)) #2

    X_data, y_data = make_classification(n_samples=config['data']['lambda_dataset_size'], 
                                                       n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                       n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                       n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                       n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                       n_classes=config['data']['num_classes'], 
                                                       n_clusters_per_class=n_clusters_per_class, 
                                                       #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                       #class_sep=1.0, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                       #hypercube=False, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                       #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                       #scale=1.0, #Multiply features by the specified value. 
                                                       shuffle=True, 
                                                       random_state=seed) 
    
    for i, column in enumerate(X_data.T):
        scaler = MinMaxScaler()
        scaler.fit(column.reshape(-1, 1))
        X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
    
    if config['data']['categorical_indices'] is not None:
        for categorical_index in config['data']['categorical_indices']:
            X_data[:,categorical_index] = np.round(X_data[:,categorical_index])    
            
    function_representation_length = ( 
       ((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables']) + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes']
  if config['function_family']['dt_type'] == 'SDT'
  else ((2 ** config['function_family']['maximum_depth'] - 1) * config['function_family']['decision_sparsity']) * 2 + (2 ** config['function_family']['maximum_depth']) if config['function_family']['dt_type'] == 'vanilla'
  else None
                                                            ) 
    
    placeholder = [0 for i in range(function_representation_length)]
        
    return placeholder, X_data, np.round(y_data), y_data 





def generate_data_make_classification_distribution_decision_tree_trained(config, seed=42):
           
    np.random.seed(seed)
    informative = 3#np.random.randint(config['data']['number_of_variables']//2, high=config['data']['number_of_variables']+1) #config['data']['number_of_variables']
    redundant = np.random.randint(0, high=config['data']['number_of_variables']-informative+1) #0
    repeated = config['data']['number_of_variables']-informative-redundant # 0

    n_clusters_per_class = min(informative//2+1, config['data']['max_distributions_per_class'])#max(2, np.random.randint(0, high=informative//2+1)) #2

    X_data, y_data, distribution_parameter_list  = make_classification_distribution(n_samples=config['data']['lambda_dataset_size'], 
                                                       n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                       n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                       n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                       n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                       n_classes=config['data']['num_classes'], 
                                                       n_clusters_per_class=n_clusters_per_class, 
                                                       #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                       class_sep=0.5, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                       hypercube=True, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                       #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                       #scale=1.0, #Multiply features by the specified value. 
                                                       shuffle=True, 
                                                       random_state=seed,
                                                       random_parameters=config['data']['random_parameters_distribution'],
                                                       distrib_param_max=config['data']['distrib_param_max']
                                                       ) 

    for i, column in enumerate(X_data.T):
        scaler = MinMaxScaler()
        scaler.fit(column.reshape(-1, 1))
        X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
    
    if config['data']['categorical_indices'] is not None:
        for categorical_index in config['data']['categorical_indices']:
            X_data[:,categorical_index] = np.round(X_data[:,categorical_index])      
            
    decision_tree = generate_random_decision_tree(config, seed)
        
    if config['function_family']['dt_type'] == 'SDT':   
        decision_tree.fit(X_data, y_data_tree, epochs=50)    

        y_data = decision_tree.predict_proba(X_data)

        return decision_tree.to_array(), X_data, np.round(y_data), y_data     
    
    
    elif config['function_family']['dt_type'] == 'vanilla': 
        decision_tree.fit(X_data, y_data_tree)    

        y_data = decision_tree.predict(X_data)    

        #placeholder = [0 for i in range((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables'] + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes'])]

        return get_parameters_from_sklearn_decision_tree(decision_tree, config), X_data, np.round(y_data), y_data, distribution_parameter_list 
    
    return None


def generate_data_make_classification_distribution(config, seed=42):
           
    np.random.seed(seed)
    informative = 3#np.random.randint(config['data']['number_of_variables']//2, high=config['data']['number_of_variables']+1) #config['data']['number_of_variables']
    redundant = np.random.randint(0, high=config['data']['number_of_variables']-informative+1) #0
    repeated = config['data']['number_of_variables']-informative-redundant # 0

    n_clusters_per_class = min(informative//2+1, config['data']['max_distributions_per_class'])#max(2, np.random.randint(0, high=informative//2+1)) #2

    X_data, y_data, distribution_parameter_list  = make_classification_distribution(n_samples=config['data']['lambda_dataset_size'], 
                                                       n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                       n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                       n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                       n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                       n_classes=config['data']['num_classes'], 
                                                       n_clusters_per_class=n_clusters_per_class, 
                                                       #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                       class_sep=0.5, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                       hypercube=True, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                       #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                       #scale=1.0, #Multiply features by the specified value. 
                                                       shuffle=True, 
                                                       random_state=seed,
                                                       random_parameters=config['data']['random_parameters_distribution'],
                                                       distrib_param_max=config['data']['distrib_param_max']
                                                       ) 
    
    for i, column in enumerate(X_data.T):
        scaler = MinMaxScaler()
        scaler.fit(column.reshape(-1, 1))
        X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
    
    if config['data']['categorical_indices'] is not None:
        for categorical_index in config['data']['categorical_indices']:
            X_data[:,categorical_index] = np.round(X_data[:,categorical_index])    
            
    function_representation_length = ( 
       ((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables']) + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes']
  if config['function_family']['dt_type'] == 'SDT'
  else ((2 ** config['function_family']['maximum_depth'] - 1) * config['function_family']['decision_sparsity']) * 2 + (2 ** config['function_family']['maximum_depth']) if config['function_family']['dt_type'] == 'vanilla'
  else None
                                                            ) 
    
    placeholder = [0 for i in range(function_representation_length)]
        
    return placeholder, X_data, np.round(y_data), y_data, distribution_parameter_list

def generate_data_distribution_trained(config, 
                                      seed=42, 
                                      max_distributions_per_class=0, 
                                      random_parameters=False, 
                                      data_noise = 0):
           
    random.seed(seed)
    distributions_per_class = max_distributions_per_class#random.randint(1, max_distributions_per_class) if max_distributions_per_class != 0 else max_distributions_per_class
    
    X_data, y_data_tree, distribution_parameter_list, _ = generate_dataset_from_distributions(distribution_list = config['data']['distribution_list'], 
                                                               number_of_variables = config['data']['number_of_variables'], 
                                                               number_of_samples = config['data']['lambda_dataset_size'], 
                                                               distributions_per_class = distributions_per_class, 
                                                               seed = seed, 
                                                               data_noise = data_noise,
                                                               random_parameters=random_parameters,
                                                               config=config)        
        

            
    decision_tree = generate_random_decision_tree(config, seed)
        
    if config['function_family']['dt_type'] == 'SDT':   
        decision_tree.fit(X_data, y_data_tree, epochs=50)    

        y_data = decision_tree.predict_proba(X_data)

        return decision_tree.to_array(), X_data, np.round(y_data), y_data     
    
    
    elif config['function_family']['dt_type'] == 'vanilla': 
        decision_tree.fit(X_data, y_data_tree)    

        y_data = decision_tree.predict(X_data)    

        #placeholder = [0 for i in range((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables'] + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes'])]

        return get_parameters_from_sklearn_decision_tree(decision_tree, config), X_data, np.round(y_data), y_data, distribution_parameter_list
    
    return None


def generate_data_distribution(config, 
                              seed=42, 
                              max_distributions_per_class=0, 
                              random_parameters=False, 
                              data_noise = 0):
        
    random.seed(seed)
    distributions_per_class = max_distributions_per_class#random.randint(1, max_distributions_per_class) if max_distributions_per_class != 0 else max_distributions_per_class
     
    X_data, y_data, distribution_parameter_list, _ = generate_dataset_from_distributions(distribution_list = config['data']['distribution_list'], 
                                                               number_of_variables = config['data']['number_of_variables'], 
                                                               number_of_samples = config['data']['lambda_dataset_size'], 
                                                               distributions_per_class = distributions_per_class, 
                                                               seed = seed, 
                                                               data_noise = data_noise,
                                                               random_parameters=random_parameters,
                                                                config=config)
    
    
    function_representation_length = ( 
       ((2 ** config['function_family']['maximum_depth'] - 1) * config['data']['number_of_variables']) + (2 ** config['function_family']['maximum_depth'] - 1) + (2 ** config['function_family']['maximum_depth']) * config['data']['num_classes']
  if config['function_family']['dt_type'] == 'SDT'
  else ((2 ** config['function_family']['maximum_depth'] - 1) * config['function_family']['decision_sparsity']) * 2 + (2 ** config['function_family']['maximum_depth']) if config['function_family']['dt_type'] == 'vanilla'
  else None
                                                            ) 
    
    placeholder = [0 for i in range(function_representation_length)]
        
    return placeholder, X_data, np.round(y_data), y_data , distribution_parameter_list








def anytree_decision_tree_from_parameters(dt_parameter_array, config, normalizer_list=None, path='./data/plotting/temp.png'):
    
    from anytree import Node, RenderTree
    from anytree.exporter import DotExporter
    
    splits, leaf_classes = get_shaped_parameters_for_decision_tree(dt_parameter_array, config, eager_execution=True)

    splits = splits.numpy()
    leaf_classes = leaf_classes.numpy()
    
    
    if normalizer_list is not None: 
        transpose = splits.transpose()
        transpose_normalized = []
        for i, column in enumerate(transpose):
            column_new = column
            if len(column_new[column_new != 0]) != 0:
                column_new[column_new != 0] = normalizer_list[i].inverse_transform(column[column != 0].reshape(-1, 1)).ravel()
            #column_new = normalizer_list[i].inverse_transform(column.reshape(-1, 1)).ravel()
            transpose_normalized.append(column_new)
        splits = np.array(transpose_normalized).transpose()

    splits_by_layer = []
    for i in range(config['function_family']['maximum_depth']+1):
        start = 2**i - 1
        end = 2**(i+1) -1
        splits_by_layer.append(splits[start:end])

    nodes = {
    }
    #tree = Tree()
    for i, splits in enumerate(splits_by_layer):
        for j, split in enumerate(splits):
            if i == 0:
                current_node_id = int(2**i - 1 + j)
                name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                split_variable = np.argmax(np.abs(split))
                split_value = np.round(split[split_variable], 3)
                split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)
                
                nodes[name] = Node(name=name, display_name=split_description)
                
                #tree.create_node(tag=split_description, identifier=name, data=None)            
            else:
                current_node_id = int(2**i - 1 + j)
                name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                parent_node_id = int(np.floor((current_node_id-1)/2))
                parent_name = 'n' + str(parent_node_id)
                split_variable = np.argmax(np.abs(split))
                split_value = np.round(split[split_variable], 3)
                split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)
                
                nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)
                
                #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)
                
    for j, leaf_class in enumerate(leaf_classes):
        i = config['function_family']['maximum_depth']
        current_node_id = int(2**i - 1 + j)
        name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
        parent_node_id = int(np.floor((current_node_id-1)/2))
        parent_name = 'n' + str(parent_node_id)
        #split_variable = np.argmax(np.abs(split))
        #split_value = np.round(split[split_variable], 3)
        split_description = str(np.round((1-leaf_class), 3))#'x' + str(split_variable) + ' <= '  + str(split_value)
        nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)
        #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)        
    
        DotExporter(nodes['n0'], nodeattrfunc=lambda node: 'label="{}"'.format(node.display_name)).to_picture(path)
        
        
    return Image(path), nodes#nodes#tree



def treelib_decision_tree_from_parameters(dt_parameter_array, config, normalizer_list=None):
    
    from treelib import Node, Tree
    
    splits, leaf_classes = get_shaped_parameters_for_decision_tree(dt_parameter_array, config, eager_execution=True)

    splits = splits.numpy()
    leaf_classes = leaf_classes.numpy()
    if normalizer_list is not None: 
        transpose = splits.transpose()
        transpose_normalized = []
        for i, column in enumerate(transpose):
            column_new = column
            if len(column_new[column_new != 0]) != 0:
                column_new[column_new != 0] = normalizer_list[i].inverse_transform(column[column != 0].reshape(-1, 1)).ravel()            
            transpose_normalized.append(column_new)
        splits = np.array(transpose_normalized).transpose()
        
    splits_by_layer = []
    for i in range(config['function_family']['maximum_depth']+1):
        start = 2**i - 1
        end = 2**(i+1) -1
        splits_by_layer.append(splits[start:end])

    tree = Tree()
    for i, splits in enumerate(splits_by_layer):
        for j, split in enumerate(splits):
            if i == 0:
                current_node_id = int(2**i - 1 + j)
                name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                split_variable = np.argmax(np.abs(split))
                split_value = np.round(split[split_variable], 3)
                split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)
                tree.create_node(tag=split_description, identifier=name, data=None)            
            else:
                current_node_id = int(2**i - 1 + j)
                name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                parent_node_id = int(np.floor((current_node_id-1)/2))
                parent_name = 'n' + str(parent_node_id)
                split_variable = np.argmax(np.abs(split))
                split_value = np.round(split[split_variable], 3)
                split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)
                tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)
                
    for j, leaf_class in enumerate(leaf_classes):
        i = config['function_family']['maximum_depth']
        current_node_id = int(2**i - 1 + j)
        name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
        parent_node_id = int(np.floor((current_node_id-1)/2))
        parent_name = 'n' + str(parent_node_id)
        #split_variable = np.argmax(np.abs(split))
        #split_value = np.round(split[split_variable], 3)
        split_description = str(np.round((1-leaf_class), 3))#'x' + str(split_variable) + ' <= '  + str(split_value)
        tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)        
    
    return tree




def generate_decision_tree_identifier(config):
    
    if config['function_family']['dt_type'] == 'SDT':
    
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
            
    elif config['function_family']['dt_type'] == 'vanilla':
        
        num_internal_nodes = 2 ** config['function_family']['maximum_depth'] - 1
        num_leaf_nodes = 2 ** config['function_family']['maximum_depth']

        decision_tree_identifier_list = []
        for internal_number in range(num_internal_nodes):
            decision_tree_identifier_list.append('feat' + str(internal_number))

        for internal_number in range(num_internal_nodes):
            decision_tree_identifier_list.append('split' + str(internal_number))
            
        for leaf_probabilities_number in range(num_leaf_nodes):
            decision_tree_identifier_list.append('lp' + str(leaf_probabilities_number))               
            
    return decision_tree_identifier_list



def dt_array_to_sklearn(vanilla_dt_array, config,X_data, y_data, printing=False):
#def dt_array_to_sklearn(vanilla_dt_array, config, printing=False):
    
    """
    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.
    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.
    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.
    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].
    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].
    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.
    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.
    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.
    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.
    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.
    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """    
    
    from math import log2
    import queue
    from sklearn.tree import DecisionTreeClassifier
    def gini(p):
        return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))    

    def level_to_pre(arr,ind,new_arr):
        if ind>=len(arr): return new_arr #nodes at ind don't exist
        new_arr.append(arr[ind]) #append to back of the array
        new_arr = level_to_pre(arr,ind*2+1,new_arr) #recursive call to left
        new_arr = level_to_pre(arr,ind*2+2,new_arr) #recursive call to right
        return new_arr

    def pre_to_level(arr):
        def left_tree_size(n):
            if n<=1: return 0
            l = int(log2(n+1)) #l = no of completely filled levels
            ans = 2**(l-1)
            last_level_nodes = min(n-2**l+1,ans)
            return ans + last_level_nodes -1       

        que = queue.Queue()
        que.put((0,len(arr)))
        ans = [] #this will be answer
        while not que.empty():
            iroot,size = que.get() #index of root and size of subtree
            if iroot>=len(arr) or size==0: continue ##nodes at iroot don't exist
            else : ans.append(arr[iroot]) #append to back of output array
            sz_of_left = left_tree_size(size) 
            que.put((iroot+1,sz_of_left)) #insert left sub-tree info to que
            que.put((iroot+1+sz_of_left,size-sz_of_left-1)) #right sub-tree info 

        return ans    
    
    splits, leaf_classes= get_shaped_parameters_for_decision_tree(vanilla_dt_array, config, eager_execution=True)
    
    if printing:
        print('splits', splits)
        print('leaf_classes', leaf_classes)
        
    internal_node_num = 2 ** config['function_family']['maximum_depth'] -1    
    leaf_node_num = 2 ** config['function_family']['maximum_depth']    
    n_nodes = internal_node_num + leaf_node_num

    indices_list = [i for i in range(internal_node_num + leaf_node_num)]
    pre_order_from_level = np.array(level_to_pre(indices_list, 0, []))

    level_order_from_pre = np.array(pre_to_level(indices_list))
    children_left = []
    children_right = []
    counter = 0
    for i in pre_order_from_level:#pre_order_from_level:
        left = 2*i+1 
        right = 2*i+2 
        if left < n_nodes:
            children_left.append(level_order_from_pre[left])
        else:
            children_left.append(-1)
        if left < n_nodes:
            children_right.append(level_order_from_pre[right])
        else:
            children_right.append(-1)            
            
        #try:
        #    children_left.append(level_order_from_pre[left])
        #except:
        #    children_left.append(-1)
        #try:
        #    children_right.append(level_order_from_pre[right])
        #except:
        #    children_right.append(-1)            
        
    children_left = np.array(children_left)
    children_right = np.array(children_right)
    
    #print('children_left', children_left.shape, children_left)
    #print('children_right', children_right.shape, children_right)
    
    indices_list = [i for i in range(internal_node_num+leaf_node_num)]
    new_order = np.array(level_to_pre(indices_list, 0, []))
    
    feature = [np.argmax(np.abs(split)) for split in splits]
    feature.extend([-2 for i in range(leaf_node_num)])
    feature = np.array(feature)[new_order]
    threshold = [split[np.argmax(np.abs(split))] for split in splits]
    threshold.extend([-2 for i in range(leaf_node_num)])
    threshold = np.round(np.array(threshold)[new_order], 3) 
    
    samples = 1000
    value_list = []
    n_node_samples_list = []
    impurity_list = []
    
    value_list_previous = None
    for current_depth in reversed(range(1, (config['function_family']['maximum_depth']+1)+1)):
        internal_node_num_current_depth = (2 ** current_depth - 1) - (2 ** (current_depth-1) - 1)
        #print(internal_node_num_current_depth)
        #n_node_samples = [samples for _ in range(internal_node_num_current_depth)]
        if current_depth > config['function_family']['maximum_depth']: #is leaf
            values = []
            impurity = []
            n_node_samples = []
            for i, leaf_class in enumerate(leaf_classes):
                current_value = [int(np.round(samples*leaf_class.numpy())), int(np.round(samples*(1-leaf_class.numpy())))] 
                curent_impurity = gini(current_value[0]/sum(current_value))
                
                values.append(current_value)
                impurity.append(curent_impurity)
                n_node_samples.append(sum(current_value))
            #values = [[0, samples] for _ in range(internal_node_num_current_depth)]
            #impurity = [0.5 for _ in range(internal_node_num_current_depth)]
        else:
            value_list_previous_left = value_list_previous[::2]
            value_list_previous_right = value_list_previous[1::2]
            samples_sum_list = np.add(value_list_previous_left, value_list_previous_right)
            
            values = [samples_sum for samples_sum in samples_sum_list]
            impurity = [gini(value[0]/sum(value)) for value in values]
            n_node_samples = [sum(value) for value in values]

            samples = samples*2
        
        value_list_previous = values
        
        n_node_samples_list[0:0] = n_node_samples
        value_list[0:0] = values
        impurity_list[0:0] = impurity        
        #n_node_samples_list.extend(n_node_samples)
        #value_list.extend(values)
        #impurity_list.extend(impurity)
        
        
        
    value = np.expand_dims(np.array(value_list), axis=1) #shape [node_count, n_outputs, max_n_classes]; number of samples for each class
    value = np.round(value[new_order].astype(np.float64), 3)
    impurity =  np.array(impurity_list) #
    impurity = impurity[new_order].astype(np.float64)
    n_node_samples = np.array(n_node_samples_list) #number of samples at each node
    n_node_samples = n_node_samples[new_order]
    weighted_n_node_samples = 1 * np.array(n_node_samples_list) #same as tree_n_node_samples, but weighted    
    weighted_n_node_samples = np.round(weighted_n_node_samples.astype(np.float64), 3)
    
    if printing:
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print("The binary tree structure has {n} nodes and has "
              "the following tree structure:\n".format(n=n_nodes))
        for i in range(n_nodes):
            if is_leaves[i]:
                print("{space}node={node} is a leaf node.".format(space=node_depth[i] * "\t", node=i))
            else:
                print("{space}node={node} is a split node: "
                      "go to node {left} if X[:, {feature}] <= {threshold} "
                      "else to node {right}.".format(
                          space=node_depth[i] * "\t",
                          node=i,
                          left=children_left[i],
                          feature=feature[i],
                          threshold=threshold[i],
                          right=children_right[i]))    

        
    clf=DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
    #y_data = [i for i in range(config['data']['num_classes'])]
    #X_data = [[0 for i in range(config['data']['number_of_variables'])] for _ in range(config['data']['num_classes'])]
    clf.fit(X_data, y_data)
    if printing:    
        print('-------------------------------------------------------------')
        print(clf.tree_.value.dtype, value.dtype)
        print(clf.tree_.impurity.dtype, impurity.dtype)
        print(clf.tree_.n_node_samples.dtype, n_node_samples.dtype)
        print(clf.tree_.weighted_n_node_samples.dtype, weighted_n_node_samples.dtype)
        print(clf.tree_.children_left.dtype, children_left.dtype)
        print(clf.tree_.children_right.dtype, children_right.dtype)
        print(clf.tree_.feature.dtype, feature.dtype)
        print(clf.tree_.threshold.dtype, threshold.dtype)      
        print('-------------------------------------------------------------')
        print('clf.tree_.value', clf.tree_.value)
        print('clf.tree_.impurity', clf.tree_.impurity)
        print('clf.tree_.n_node_samples', clf.tree_.n_node_samples)
        print('clf.tree_.weighted_n_node_samples', clf.tree_.weighted_n_node_samples)
        print('clf.tree_.children_left', clf.tree_.children_left)
        print('clf.tree_.children_right', clf.tree_.children_right)
        print('clf.tree_.feature', clf.tree_.feature)
        print('clf.tree_.threshold', clf.tree_.threshold)            
        print('-------------------------------------------------------------')    
    
    #print(type(clf.tree_.node_count), type(n_nodes))
    #print(type(clf.tree_.capacity), type(n_nodes))    
    clf.tree_.node_count = n_nodes
    clf.tree_.capacity = n_nodes
    
    #print(type(clf.tree_.node_count), type(n_nodes))
    #print(type(clf.tree_.capacity), type(n_nodes))       
    
    #print(clf.tree_.value, np.array(clf.tree_.value.shape))
    #print(value, np.array(value).shape)
    #TODO: FÜR VALUES NICHT IMMER 50/50 BEI INNER UND 100/0 BEI LEAF, SONDERN: BEI LEAFS ANFANGEN UND DANN DEN PFADEN ENTLANG HOCH-ADDIEREN FÜR JEDEN PARENT NODE
    if printing:    
        print('-------------------------------------------------------------')
        print(clf.tree_.value.dtype, value.dtype)
        print(clf.tree_.impurity.dtype, impurity.dtype)
        print(clf.tree_.n_node_samples.dtype, n_node_samples.dtype)
        print(clf.tree_.weighted_n_node_samples.dtype, weighted_n_node_samples.dtype)
        print(clf.tree_.children_left.dtype, children_left.dtype)
        print(clf.tree_.children_right.dtype, children_right.dtype)
        print(clf.tree_.feature.dtype, feature.dtype)
        print(clf.tree_.threshold.dtype, threshold.dtype)
    for i in indices_list:
        clf.tree_.children_left[i] = children_left[i]
        clf.tree_.children_right[i] = children_right[i]            
        clf.tree_.value[i] = value[i]
        clf.tree_.impurity[i] = impurity[i]
        clf.tree_.n_node_samples[i] = n_node_samples[i]
        clf.tree_.weighted_n_node_samples[i] = weighted_n_node_samples[i]
        clf.tree_.feature[i] = feature[i]
        clf.tree_.threshold[i] = threshold[i]      
    if printing:    
        print('-------------------------------------------------------------')
        print('value', value)
        print('impurity', impurity)
        print('n_node_samples', n_node_samples)
        print('weighted_n_node_samples', weighted_n_node_samples)
        print('children_left', children_left)
        print('children_right', children_right)
        print('feature', feature)
        print('threshold', threshold)            
        print('-------------------------------------------------------------')        
        print('-------------------------------------------------------------')
        print('clf.tree_.value', clf.tree_.value)
        print('clf.tree_.impurity', clf.tree_.impurity)
        print('clf.tree_.n_node_samples', clf.tree_.n_node_samples)
        print('clf.tree_.weighted_n_node_samples', clf.tree_.weighted_n_node_samples)
        print('clf.tree_.children_left', clf.tree_.children_left)
        print('clf.tree_.children_right', clf.tree_.children_right)
        print('clf.tree_.feature', clf.tree_.feature)
        print('clf.tree_.threshold', clf.tree_.threshold)            
        print('-------------------------------------------------------------')
    return clf



def get_number_of_function_parameters(dt_type, maximum_depth, number_of_variables, number_of_classes):
    
    number_of_function_parameters = None
    
    if dt_type == 'SDT':
        number_of_function_parameters = (2 ** maximum_depth - 1) * (number_of_variables + 1) + (2 ** maximum_depth) * number_of_classes
    elif dt_type == 'vanilla':
        number_of_function_parameters = (2 ** maximum_depth - 1) * 2 + (2 ** maximum_depth)

    return number_of_function_parameters




######################################################################################################################################################################################################################
###########################################################################################  REAL WORLD & SYNTHETIC EVALUATION ################################################################################################ 
######################################################################################################################################################################################################################
def get_distribution_data_from_string(distribution_name, size, seed=None, parameters=None, random_parameters=False, distrib_param_max=1, class_identifier=None):
        

    random.seed(seed)
    np.random.seed(seed)
        
    if parameters == None:
        value_1 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)
        value_2 = np.random.uniform(0, distrib_param_max)#np.random.uniform(-0.2, 1.2)   
        
        if random_parameters:
            if class_identifier is None:
                parameter_by_distribution = {
                    'normal': {
                        'loc': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },
                    'uniform': {
                        'low': np.minimum(value_1, value_2),
                        'high': np.maximum(value_1, value_2),
                    },
                    'gamma': {
                        'shape': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },        
                    'exponential': {
                        'scale': np.random.uniform(0, distrib_param_max),
                    },        
                    'beta': {
                        'a': np.random.uniform(0, distrib_param_max),
                        'b': np.random.uniform(0, distrib_param_max),
                    },
                    'binomial': {
                        'n': 100,
                        'p': np.random.uniform(0, 1),
                    },
                    'poisson': {
                        'lam': np.random.uniform(0, distrib_param_max),
                    },
                    'lognormal': {
                        'mean': np.random.uniform(0, distrib_param_max),
                        'sigma': np.random.uniform(0, distrib_param_max),
                    },             
                    'f': {
                        'dfnum': np.random.uniform(0, distrib_param_max),
                        'dfden': np.random.uniform(0, distrib_param_max),
                    },
                    'logistic': {
                        'loc': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },
                    'weibull': {
                        'a': np.random.uniform(0, distrib_param_max),
                    },    
                    'standardnormal': {
                        'loc': 0,
                        'scale': 1,
                    },      
                    'standarduniform': {
                        'low': 0,
                        'high': 1,
                    },                    

                }
            elif class_identifier == 0:
                parameter_by_distribution = {
                    'normal': {
                        'loc': np.random.uniform(0, 2),
                        'scale': np.random.uniform(0, 3),
                    },
                    'uniform': {
                        'low': np.random.uniform(0, 0.4),
                        'high': np.random.uniform(0.7, 0.9),
                    },
                    'gamma': {
                        'shape': np.random.uniform(0, 0.5),
                        'scale': np.random.uniform(2, 3),
                    },        
                    'exponential': {
                        'scale': np.random.uniform(0, -1),
                    },        
                    'beta': {
                        'a': np.random.uniform(0, 0.25),
                        'b': np.random.uniform(1, 2),
                    },
                    'binomial': {
                        'n': 100,
                        'p': np.random.uniform(0, 1),
                    },
                    'poisson': {
                        'lam': np.random.uniform(0, -1),
                    },        

                }
            elif class_identifier == 1:
                parameter_by_distribution = {
                    'normal': {
                        'loc': np.random.uniform(1, 3),
                        'scale': np.random.uniform(0, 3),
                    },
                    'uniform': {
                        'low': np.random.uniform(0.1, 0.3),
                        'high': np.random.uniform(0.6, 1),
                    },
                    'gamma': {
                        'shape': np.random.uniform(0.5, 1.5),
                        'scale': np.random.uniform(0.5, 1),
                    },        
                    'exponential': {
                        'scale': np.random.uniform(0, -1),
                    },        
                    'beta': {
                        'a': np.random.uniform(0.5, 1.5),
                        'b': np.random.uniform(5, 10),
                    },
                    'binomial': {
                        'n': 100,
                        'p': np.random.uniform(0, 1),
                    },
                    'poisson': {
                        'lam': np.random.uniform(0, -1),
                    },              

                }
                
                
            
            random.seed(seed)
            np.random.seed(seed)
        else:        
            
            if class_identifier is None:
                parameter_by_distribution = {
                    'normal': {
                        'loc': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },
                    'uniform': {
                        'low': np.minimum(value_1, value_2),
                        'high': np.maximum(value_1, value_2),
                    },
                    'gamma': {
                        'shape': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },        
                    'exponential': {
                        'scale': np.random.uniform(0, distrib_param_max),
                    },        
                    'beta': {
                        'a': np.random.uniform(0, distrib_param_max),
                        'b': np.random.uniform(0, distrib_param_max),
                    },
                    'binomial': {
                        'n': 100,
                        'p': np.random.uniform(0, 1),
                    },
                    'poisson': {
                        'lam': np.random.uniform(0, distrib_param_max),
                    },   
                    'lognormal': {
                        'mean': np.random.uniform(0, distrib_param_max),
                        'sigma': np.random.uniform(0, distrib_param_max),
                    },             
                    'f': {
                        'dfnum': np.random.uniform(0, distrib_param_max),
                        'dfden': np.random.uniform(0, distrib_param_max),
                    },
                    'logistic': {
                        'loc': np.random.uniform(0, distrib_param_max),
                        'scale': np.random.uniform(0, distrib_param_max),
                    },
                    'weibull': {
                        'a': np.random.uniform(0, distrib_param_max),
                    },    
                    'standardnormal': {
                        'loc': 0,
                        'scale': 1,
                    },      
                    'standarduniform': {
                        'low': 0,
                        'high': 1,
                    },                      

                }              
            
            else:
                parameter_by_distribution = {
                    'normal': {
                        'loc': 1.5,#distrib_param_max/2,
                        'scale': 1.5,#distrib_param_max/2,
                    },
                    'uniform': {
                        'low': 0,
                        'high': 1,#distrib_param_max/2,
                    },
                    'gamma': {
                        'shape': 0.75,#distrib_param_max/2,#2,
                        'scale': 1.5,#distrib_param_max/2,#2,
                    },        
                    'exponential': {
                        'scale': distrib_param_max/2,
                    },        
                    'beta': {
                        'a': 0.75,#distrib_param_max/2,#2,
                        'b': 5,#distrib_param_max/2,#5,
                    },
                    'binomial': {
                        'n': 100,
                        'p': 0.5,
                    },
                    'poisson': {
                        'lam': distrib_param_max/2,#1,
                    },        

                }           
        
        
    else:
        parameter_by_distribution = {
            distribution_name: parameters
        } 
      
    if distribution_name == 'normal':
        return np.nan_to_num(np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=size)), parameter_by_distribution['normal']
    elif distribution_name == 'uniform':
        return np.nan_to_num(np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=size)), parameter_by_distribution['uniform']
    elif distribution_name == 'gamma':     
        return np.nan_to_num(np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=size)), parameter_by_distribution['gamma']
    elif distribution_name == 'exponential':    
        return np.nan_to_num(np.random.exponential(parameter_by_distribution['exponential']['scale'], size=size)), parameter_by_distribution['exponential']   
    elif distribution_name == 'beta':
        return np.nan_to_num(np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=size)), parameter_by_distribution['beta']
    elif distribution_name == 'binomial':   
        return np.nan_to_num(np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=size)), parameter_by_distribution['binomial']  
    elif distribution_name == 'poisson':
        return np.nan_to_num(np.random.poisson(parameter_by_distribution['poisson']['lam'], size=size)), parameter_by_distribution['poisson'] 
    elif distribution_name == 'lognormal':
        return np.nan_to_num(np.random.lognormal(parameter_by_distribution['lognormal']['mean'],parameter_by_distribution['lognormal']['sigma'], size=size)), parameter_by_distribution['lognormal']  
    elif distribution_name == 'f':
        return np.nan_to_num(np.random.f(parameter_by_distribution['f']['dfnum'], parameter_by_distribution['f']['dfden'], size=size)), parameter_by_distribution['f']  
    elif distribution_name == 'logistic':
        return np.nan_to_num(np.random.logistic(parameter_by_distribution['logistic']['loc'], parameter_by_distribution['logistic']['scale'], size=size)), parameter_by_distribution['logistic']  
    elif distribution_name == 'weibull':
        return np.nan_to_num(np.random.weibull(parameter_by_distribution['weibull']['a'], size=size)), parameter_by_distribution['weibull']    
    elif distribution_name == 'standarduniform':
        return np.nan_to_num(np.random.uniform(parameter_by_distribution['standarduniform']['low'], parameter_by_distribution['standarduniform']['high'], size=size)), parameter_by_distribution['standarduniform']    
    elif distribution_name == 'standardnormal':
        return np.nan_to_num(np.random.normal(parameter_by_distribution['standardnormal']['loc'], parameter_by_distribution['standardnormal']['scale'], size=size)), parameter_by_distribution['standardnormal']    

    
    
    return None, None
    

def distribution_evaluation_interpretation_net_synthetic_data(loss_function, 
                                                               metrics,
                                                               #model,
                                                               config,
                                                               distribution_list_evaluation,
                                                               identifier,
                                                               lambda_net_parameters_train,
                                                               mean_train_parameters,
                                                               std_train_parameters,
                                                               distances_dict={},
                                                               max_distributions_per_class=0,
                                                               flip_percentage=0.0,
                                                               data_noise=0.0,
                                                               verbose=0,
                                                               random_parameters=None,
                                                               backend='loky'):

        
    inet_evaluation_result_dict_complete_by_distribution = {}
    inet_evaluation_result_dict_mean_by_distribution = {}

    parallel_inet_evaluation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend=backend) #loky #sequential multiprocessing
    evaluation_results_by_dataset = parallel_inet_evaluation(delayed(distribution_evaluation_single_model_synthetic_data)(loss_function, 
                                                                                                               metrics,
                                                                                                               #model,
                                                                                                               config,
                                                                                                               distribution_list_evaluation = distribution_list_evaluation,
                                                                                                               lambda_net_parameters_train=lambda_net_parameters_train,
                                                                                                               mean_train_parameters=mean_train_parameters,
                                                                                                               std_train_parameters=std_train_parameters,    
                                                                                                               data_seed=i,
                                                                                                               max_distributions_per_class = max_distributions_per_class,
                                                                                                               flip_percentage=flip_percentage,
                                                                                                               data_noise=data_noise,
                                                                                                               verbose=verbose,
                                                                                                               random_parameters=random_parameters) for i in range(config['i_net']['test_size'])) 




    del parallel_inet_evaluation

    distribution_parameter_list_list = [result[-1] for result in evaluation_results_by_dataset]
    
    test_network_list = []
    model_history_list = []
    normalizer_list_list = []
    data_dict_list = []
            
    distances_dict_list_unstructured = np.array([[None] * config['i_net']['test_size']] * len(distribution_list_evaluation))
    inet_evaluation_results = np.array([[None] * config['i_net']['test_size']] * len(distribution_list_evaluation))
    dt_inet_list = np.array([[None] * config['i_net']['test_size']] * len(distribution_list_evaluation))
    dt_distilled_list_list = np.array([[None] * config['i_net']['test_size']] * len(distribution_list_evaluation))

    for i, evaluation_result_by_dataset in enumerate(evaluation_results_by_dataset):
        (distances_dict_by_dataset, _, results_list_by_dataset, dt_inet_by_dataset, dt_distilled_list_by_dataset, data_dict_by_dataset, normalizer_list_by_dataset, test_network_by_dataset, model_history_by_dataset, _) = evaluation_result_by_dataset
        test_network_list.append(test_network_by_dataset)
        model_history_list.append(model_history_by_dataset)
        normalizer_list_list.append(normalizer_list_by_dataset)
        data_dict_list.append(data_dict_by_dataset)

        for j, (distances_dict_unstructured_by_distribution, results_list_by_distribution, dt_inet_by_distribution, dt_distilled_list_by_distribution) in enumerate(zip(distances_dict_by_dataset, results_list_by_dataset, dt_inet_by_dataset, dt_distilled_list_by_dataset)):
            distances_dict_list_unstructured[j][i] = distances_dict_unstructured_by_distribution
            inet_evaluation_results[j][i] = results_list_by_distribution#[0]
            dt_inet_list[j][i] = dt_inet_by_distribution#[0]
            dt_distilled_list_list[j][i] = dt_distilled_list_by_distribution#[0]

    
    for distribution, distances_dict_list_unstructured_by_distribution, inet_evaluation_results_by_distribution, dt_inet_list_by_distribution, dt_distilled_list_list_by_distribution in zip(distribution_list_evaluation, distances_dict_list_unstructured, inet_evaluation_results, dt_inet_list, dt_distilled_list_list):      
        inet_evaluation_result_dict_by_distribution= None
        for some_dict in inet_evaluation_results_by_distribution:
            if inet_evaluation_result_dict_by_distribution == None:
                inet_evaluation_result_dict_by_distribution = some_dict
            else:
                inet_evaluation_result_dict_by_distribution = mergeDict(inet_evaluation_result_dict_by_distribution, some_dict)

        inet_evaluation_result_dict_mean_by_distribution[str(distribution)] = {}

        for key_l1, values_l1 in inet_evaluation_result_dict_by_distribution.items():
            if key_l1 != 'function_values':
                if isinstance(values_l1, dict):
                    inet_evaluation_result_dict_mean_by_distribution[str(distribution)][key_l1] = {}
                    for key_l2, values_l2 in values_l1.items():
                        inet_evaluation_result_dict_mean_by_distribution[str(distribution)][key_l1][key_l2] = np.mean(values_l2)
                        inet_evaluation_result_dict_mean_by_distribution[str(distribution)][key_l1][key_l2 + '_median'] = np.median(values_l2)   

        inet_evaluation_result_dict_complete_by_distribution[str(distribution)] = inet_evaluation_result_dict_by_distribution        
        
        
    inet_evaluation_result_dict = None    
    for some_dict in inet_evaluation_results.ravel():
        if inet_evaluation_result_dict == None:
            inet_evaluation_result_dict = some_dict
        else:
            inet_evaluation_result_dict = mergeDict(inet_evaluation_result_dict, some_dict)

    inet_evaluation_result_dict_mean = {}

    for key_l1, values_l1 in inet_evaluation_result_dict.items():
        if key_l1 != 'function_values':
            if isinstance(values_l1, dict):
                inet_evaluation_result_dict_mean[key_l1] = {}
                for key_l2, values_l2 in values_l1.items():
                    inet_evaluation_result_dict_mean[key_l1][key_l2] = np.mean(values_l2)
                    inet_evaluation_result_dict_mean[key_l1][key_l2 + '_median'] = np.median(values_l2)                                                     

    distances_dict_list = None                                              
    for distances_dict_single in distances_dict_list_unstructured.ravel():
        if distances_dict_list == None:
            distances_dict_list = distances_dict_single
        else:
            distances_dict_list = mergeDict(distances_dict_list, distances_dict_single)     

    distances_dict[identifier] = {}

    for key, value in distances_dict_list.items():
        distances_dict[identifier][key] = np.mean(value)   
            
    return (distances_dict, 
            inet_evaluation_result_dict, 
            inet_evaluation_result_dict_complete_by_distribution, 
            inet_evaluation_result_dict_mean, 
            inet_evaluation_result_dict_mean_by_distribution, 
            inet_evaluation_results, 
            dt_inet_list, 
            dt_distilled_list_list, 
            data_dict_list, 
            normalizer_list_list,
            test_network_list,
            model_history_list,
            distribution_parameter_list_list)
         
    
def generate_dataset_from_distributions(distribution_list, 
                                        number_of_variables, 
                                        number_of_samples, 
                                        distributions_per_class = 0, 
                                        seed = None, 
                                        flip_percentage=0.0, 
                                        data_noise=0.0, 
                                        random_parameters=None, 
                                        distribution_dict_list=None,
                                        config=None):  

    def split_into(n, p):
        return np.floor([n/p + 1] * (n%p) + [n/p] * (p - n%p)).astype(np.int64)
                    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        if config['data']['distrib_by_feature']:
            distribution_list = distribution_list[0]
    except:
        pass

    X_data_list = []
    feature_weights_list = []
    distribution_parameter_list = []

    samples_class_0 = int(np.floor(number_of_samples/2))
    samples_class_1 = number_of_samples - samples_class_0     
    
    try:
        distrib_param_max = config['data']['distrib_param_max']
    except:
        distrib_param_max = 1       
    if distribution_dict_list is not None:
        
        if config is not None:
            distributions_per_class_original = config['data']['max_distributions_per_class']
        else:
            distributions_per_class_original = distributions_per_class
                        
        for i in range(number_of_variables):
            distribution_name = list(distribution_dict_list[i].keys())[0]
            try:
                distributions_per_class = len(list(distribution_dict_list[i][distribution_name]['class_0'].values())[0])
            except:
                distributions_per_class = 1

            if config['data']['fixed_class_probability']:
                samples_class_0_distrib = samples_class_0
                samples_class_1_distrib = samples_class_1                 
            else:
                class_0_distrib_ratio = distribution_dict_list[i][distribution_name]['samples_class_0']/config['data']['lambda_dataset_size']
                samples_class_0_distrib = np.round(number_of_samples * class_0_distrib_ratio).astype(np.int64)
                samples_class_1_distrib = number_of_samples - samples_class_0_distrib                 
                
            if distributions_per_class_original == 0:
                pass
            else:
                class_0_data_list = [None] * (distributions_per_class)
                distribution_parameter_0_list = [None] * (distributions_per_class)
                class_1_data_list = [None] * (distributions_per_class)
                distribution_parameter_1_list = [None] * (distributions_per_class)

                samples_class_0_distrib_list = split_into(samples_class_0_distrib , distributions_per_class)
                samples_class_1_distrib_list = split_into(samples_class_1_distrib , distributions_per_class)
                
            feature_weight = distribution_dict_list[i][distribution_name]['feature_weight_0']#np.random.uniform(0, 1)

            for j in range(distributions_per_class):

                distribution_parameter_0 = {}
                for key, value in distribution_dict_list[i][distribution_name]['class_0'].items():
                    if distributions_per_class > 1:
                        distribution_parameter_0[key] = value[j]
                    else:
                        distribution_parameter_0[key] = value

                distribution_parameter_1 = {}
                for key, value in distribution_dict_list[i][distribution_name]['class_1'].items():
                    if distributions_per_class > 1:
                        distribution_parameter_1[key] = value[j]
                    else:
                        distribution_parameter_1[key] = value
                        
                if distributions_per_class_original == 0:
                    feature_data, distribution_parameters = get_distribution_data_from_string(distribution_name, number_of_samples, seed=((seed+1)*(i+1)) % (2**32 - 1), parameters=distribution_parameter_0, random_parameters=False, distrib_param_max=distrib_param_max)  

                    
                else:                    
                    class_0_data_list[j], distribution_parameter_0_list[j] = get_distribution_data_from_string(distribution_name, samples_class_0_distrib_list[j], seed=((seed+1)*(i+1)*(j+1)) % (2**32 - 1), parameters=distribution_parameter_0, random_parameters=False, distrib_param_max=distrib_param_max)                    
                    class_1_data_list[j], distribution_parameter_1_list[j] = get_distribution_data_from_string(distribution_name, samples_class_1_distrib_list[j], seed=(1_000_000_000+(seed+1)*(i+1)*(j+1)) % (2**32 - 1), parameters=distribution_parameter_1, random_parameters=False, distrib_param_max=distrib_param_max)                    

            if distributions_per_class_original != 0:
                if distribution_dict_list[i][distribution_name]['seed_shuffeling'] is not None:
                    seed = distribution_dict_list[i][distribution_name]['seed_shuffeling']
                
                class_0_data = np.hstack(class_0_data_list)
                class_1_data = np.hstack(class_1_data_list)

                feature_0_weights = np.full_like(a=class_0_data, fill_value=feature_weight, dtype=np.float32)
                feature_1_weights = np.full_like(a=class_1_data, fill_value=-feature_weight, dtype=np.float32)  
                
                feature_data = np.hstack([class_0_data, class_1_data])
                
                feature_weights = np.hstack([feature_0_weights, feature_1_weights])
                
                def make_batch(iterable, n=1):
                    l = len(iterable)
                    for ndx in range(0, l, n):
                        yield iterable[ndx:min(ndx + n, l)]                
                
                if config['data']['weighted_data_generation']:
                    if config['data']['fixed_class_probability']:
                        batch_size_0 = 100
                        batch_size_1 = 100
                    else:
                        max_samples = max(class_0_data.shape[0], class_1_data.shape[0])

                        fraction = np.ceil(feature_data.shape[0]/100*2)

                        batch_size_0 = int(np.ceil(class_0_data.shape[0] / fraction))
                        batch_size_1 = int(np.ceil(class_0_data.shape[0] / fraction))                 
                    
                    batch_list_feature_data = []
                    for batch_0, batch_1 in zip(make_batch(class_0_data, batch_size_0), make_batch(class_1_data, batch_size_1)):
                        batch = np.hstack([batch_0, batch_1])
                        #print(batch.shape)
                        np.random.seed(seed+i)
                        np.random.shuffle(batch)
                        batch_list_feature_data.append(batch)
                    feature_data = np.hstack(batch_list_feature_data)
                    #print(feature_data.shape)
                    
                    #print('feature_weights', feature_weights)
                    batch_list_feature_weights = []
                    for batch_0, batch_1 in zip(make_batch(feature_0_weights, batch_size_0), make_batch(feature_1_weights, batch_size_1)):
                        batch = np.hstack([batch_0, batch_1])
                        #print(batch.shape)
                        np.random.seed(seed+i)
                        np.random.shuffle(batch)
                        batch_list_feature_weights.append(batch)
                    feature_weights = np.hstack(batch_list_feature_weights)                           
                    #print(feature_weights.shape)
                    
                distribution_parameter_0 = None
                for distribution_parameter in distribution_parameter_0_list:
                    if distribution_parameter_0 is None:
                        distribution_parameter_0 = distribution_parameter
                    else:
                        distribution_parameter_0 = mergeDict(distribution_parameter_0, distribution_parameter)

                distribution_parameter_1 = None 
                for distribution_parameter in distribution_parameter_1_list:
                    if distribution_parameter_1 is None:
                        distribution_parameter_1 = distribution_parameter
                    else:
                        distribution_parameter_1 = mergeDict(distribution_parameter_1, distribution_parameter)

                distribution_parameter = {
                    distribution_name: {
                        'class_0': distribution_parameter_0,
                        'class_1': distribution_parameter_1,
                        'samples_class_0': samples_class_0_distrib,
                        'feature_weight_0': feature_weight,
                        'seed_shuffeling': seed,
                    }
                }
                
                
            else:
                distribution_parameter = {
                    distribution_name: {
                        'class_0': distribution_parameters,
                        'class_1': distribution_parameters,
                        'samples_class_0': samples_class_0,
                        #'feature_weight_0': feature_weight,
                    }
                }
                
            distribution_parameter_list.append(distribution_parameter)

            X_data_list.append(feature_data)
            feature_weights_list.append(feature_weights)

        X_data = np.vstack(X_data_list).T
        feature_weights = np.vstack(feature_weights_list).T
        
        if distributions_per_class_original == 0:
            X_data = np.sort(X_data, axis=0)

        X_data = X_data + X_data * np.random.uniform(-data_noise, data_noise, X_data.shape) #np.random.uniform(-flip_percentage, flip_percentage, X_data.shape)#         
        X_data, normalizer_list = normalize_real_world_data(X_data)
        
        if config['data']['weighted_data_generation']:
            feature_weights_reduced = np.sum(feature_weights, axis=1)
            threshold = np.median(feature_weights_reduced)

            y_data = deepcopy(feature_weights_reduced)
            y_data[feature_weights_reduced >= threshold] = 1
            y_data[feature_weights_reduced < threshold] = 0     
        else:
            if config['data']['balanced_data']:
                y_data = np.hstack([[0]*samples_class_0, [1]*samples_class_1])
            else:
                ratio = np.random.uniform(0.25, 0.75)
                total_samples = samples_class_0 + samples_class_1
                samples_class_0_y_data = int(np.floor(samples_class_0 * ratio))
                samples_class_1_y_data = total_samples - samples_class_0_y_data
                y_data = np.hstack([[0]*samples_class_0_y_data, [1]*samples_class_1_y_data])            
                    
        idx = np.random.choice(y_data.shape[0], int(y_data.shape[0]*flip_percentage), replace=False)
        y_data[idx] = (y_data[idx] + 1) % 2              
   

    ################################################
    else:   
        #accuracy_single_split = 1
        #accuracy_max_split = 0
              
        #accuracy_single_split_threshold = 0.90
        #accuracy_max_split_threshold = 0.70
        #accuracy_diff_threshold = 0.10

        condition = True
        #while accuracy_single_split > accuracy_single_split_threshold or (accuracy_max_split-accuracy_single_split) < accuracy_diff_threshold or accuracy_max_split < accuracy_max_split_threshold and accuracy_max_split > 0.95: 
        while condition:
            
            X_data_list = []
            distribution_parameter_list = []   
                
            #list1, list2 = zip(*sorted(zip(list1, list2)))
                
            if distributions_per_class == 0:

                for i in range(number_of_variables):
                    #samples_class_0 = np.random.randint(1, int(np.floor(number_of_samples/2)))
                    #samples_class_1 = number_of_samples - samples_class_0                             
                    distribution_name = np.random.choice(distribution_list)
                    #data_list = [None]
                    #distribution_parameter_list = [None]

                    feature_data, distribution_parameter = get_distribution_data_from_string(distribution_name, samples_class_0+samples_class_1, seed=((seed+1)*(i+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max)


                    #feature_data = np.sort(feature_data, axis=0)
                    distribution_parameter = {distribution_name: distribution_parameter}

                    distribution_parameter_list.append(distribution_parameter)
                    X_data_list.append(feature_data)    

                X_data = np.vstack(X_data_list).T
                #print(X_data[:,:3])
                X_data = np.sort(X_data, axis=0)
                #print(X_data[:,:3])
                X_data = X_data + X_data * np.random.uniform(-data_noise, data_noise, X_data.shape) #np.random.uniform(-flip_percentage, flip_percentage, X_data.shape)#

                #print(X_data[:,:3])
                X_data, normalizer_list = normalize_real_world_data(X_data)
                #print(X_data[:,:3])
                if config['data']['balanced_data']:
                    y_data = np.hstack([[0]*samples_class_0, [1]*samples_class_1])
                else:
                    ratio = np.random.uniform(0.25, 0.75)
                    total_samples = samples_class_0 + samples_class_1
                    samples_class_0_y_data = int(np.floor(samples_class_0 * ratio))
                    samples_class_1_y_data = total_samples - samples_class_0_y_data
                    y_data = np.hstack([[0]*samples_class_0_y_data, [1]*samples_class_1_y_data])                
                

                idx = np.random.choice(y_data.shape[0], int(y_data.shape[0]*flip_percentage), replace=False)
                y_data[idx] = (y_data[idx] + 1) % 2  
                

            else:                
                distribution_parameter_0_list_list = []
                distribution_parameter_1_list_list = []
                
                samples_class_0_distrib_list_list = []
                samples_class_1_distrib_list_list = []
                
                for i in range(number_of_variables):   
                    
                    if config['data']['weighted_data_generation']:
                        np.random.seed(seed + i)
                        feature_weight = np.random.uniform(0, 1)
                    else:
                        feature_weight = 1
                
                    if config['data']['fixed_class_probability']:
                        samples_class_0_distrib = samples_class_0
                        samples_class_1_distrib = samples_class_1                 
                    else:
                        samples_class_0_distrib = np.random.randint(1, number_of_samples)
                        samples_class_1_distrib = number_of_samples - samples_class_0_distrib                         

                    #print('samples_class_0_distrib', samples_class_0_distrib)
                    #print('samples_class_1_distrib', samples_class_1_distrib)
                    #print('samples_class_0_distrib', samples_class_0_distrib, 'samples_class_1_distrib', samples_class_1_distrib)

                    distribution_name = np.random.choice(distribution_list)

                    class_0_data_list = [None] * (distributions_per_class)
                    distribution_parameter_0_list = [None] * (distributions_per_class)
                    class_1_data_list = [None] * (distributions_per_class)
                    distribution_parameter_1_list = [None] * (distributions_per_class)

                    samples_class_0_distrib_list = split_into(samples_class_0_distrib, distributions_per_class)
                    samples_class_1_distrib_list = split_into(samples_class_1_distrib, distributions_per_class)

                    #print('samples_class_0_distrib_list', samples_class_0_distrib_list)
                    #print('samples_class_1_distrib_list', samples_class_1_distrib_list)                    
                    for j in range(distributions_per_class):
                        
                        if not config['data']['shift_distrib']:#condition or config['data']['number_of_generated_datasets'] == 11111:
                            if True:
                                class_0_data_list[j], distribution_parameter_0_list[j] = get_distribution_data_from_string(distribution_name, samples_class_0_distrib_list[j], seed=((seed+1)*(i+1)*(j+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max)
                                class_1_data_list[j], distribution_parameter_1_list[j] = get_distribution_data_from_string(distribution_name, samples_class_1_distrib_list[j], seed=(1_000_000_000+(seed+1)*(i+1)*(j+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max)                        
                            elif False:#True:
                                class_0_data_list[j], distribution_parameter_0_list[j] = get_distribution_data_from_string(distribution_name, samples_class_0_distrib_list[j], seed=((seed+1)*(i+1)*(j+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max, class_identifier=0)
                                class_1_data_list[j], distribution_parameter_1_list[j] = get_distribution_data_from_string(distribution_name, samples_class_1_distrib_list[j], seed=(1_000_000_000+(seed+1)*(i+1)*(j+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max, class_identifier=1)
                        else:
                            class_0_data_list[j], distribution_parameter_0_list[j] = get_distribution_data_from_string(distribution_name, samples_class_0_distrib_list[j], seed=((seed+1)*(i+1)*(j+1)) % (2**32 - 1), random_parameters=random_parameters, distrib_param_max=distrib_param_max)
                            distribution_parameter_new = {}                 

                            #!!!!!NOT ALWAYS CHANGE ALL PARAMETERS (RANDOM 0, 1, 2)!!!!!!!
                            if False:
                                no_parameters = len(list(distribution_parameter_0_list[j].keys()))
                                no_parameters_to_change = np.random.randint(0, no_parameters)
                                parameters_to_change = np.random.choice([num for num in range(no_parameters)], no_parameters_to_change, replace=False)
                            elif False:
                                no_parameters = len(list(distribution_parameter_0_list[j].keys()))
                                #no_parameters_to_change = np.random.randint(0, no_parameters)
                                parameters_to_change = np.random.choice([num for num in range(no_parameters)], no_parameters, replace=False)
                            elif False:
                                no_parameters = len(list(distribution_parameter_0_list[j].keys()))
                                #no_parameters_to_change = np.random.randint(0, no_parameters)
                                parameters_to_change = np.random.choice([num for num in range(no_parameters)], 1, replace=False)    
                            else:
                                parameters_to_change = [0,1]

                            #print(parameters_to_change)
                            for index, (key, value) in enumerate(distribution_parameter_0_list[j].items()):
                                if index in parameters_to_change:
                                    #multiplier = np.random.choice([-0.5, 0.5])
                                    multiplier = np.random.uniform(-0.5, 0.5)

                                    new_value = value + value*multiplier
                                    new_value_clipped = np.clip(new_value, 0, config['data']['distrib_param_max'])

                                    if key == 'p':
                                        distribution_parameter_new[key] =  np.clip(new_value_clipped, 0, 1)
                                    else:
                                        distribution_parameter_new[key] =  new_value_clipped
                                else:
                                    distribution_parameter_new[key] =  value

                            class_1_data_list[j], distribution_parameter_1_list[j] = get_distribution_data_from_string(distribution_name, samples_class_1_distrib_list[j], seed=(1_000_000_000+(seed+1)*(i+1)*(j+1)) % (2**32 - 1), parameters=distribution_parameter_new, random_parameters=random_parameters, distrib_param_max=distrib_param_max)                    


                    class_0_data = np.hstack(class_0_data_list)
                    class_1_data = np.hstack(class_1_data_list)
                    
                    feature_0_weights = np.full_like(a=class_0_data, fill_value=feature_weight, dtype=np.float32)
                    feature_1_weights = np.full_like(a=class_1_data, fill_value=-feature_weight, dtype=np.float32)                      

                    distribution_parameter_0 = None
                    for distribution_parameter in distribution_parameter_0_list:
                        if distribution_parameter_0 is None:
                            distribution_parameter_0 = distribution_parameter
                        else:
                            distribution_parameter_0 = mergeDict(distribution_parameter_0, distribution_parameter)

                    distribution_parameter_1 = None 
                    for distribution_parameter in distribution_parameter_1_list:
                        if distribution_parameter_1 is None:
                            distribution_parameter_1 = distribution_parameter
                        else:
                            distribution_parameter_1 = mergeDict(distribution_parameter_1, distribution_parameter)
                    distribution_parameter = {
                        distribution_name: {
                            'class_0': distribution_parameter_0,
                            'class_1': distribution_parameter_1,
                            'samples_class_0': samples_class_0_distrib,
                            'feature_weight_0': feature_weight,
                            'seed_shuffeling': seed,
                        }
                    }
                    distribution_parameter_list.append(distribution_parameter)

                    feature_data = np.hstack([class_0_data, class_1_data])
                    feature_weights = np.hstack([feature_0_weights, feature_1_weights])

                    def make_batch(iterable, n=1):
                        l = len(iterable)
                        for ndx in range(0, l, n):
                            yield iterable[ndx:min(ndx + n, l)]                

                    if config['data']['weighted_data_generation']:
                        if config['data']['fixed_class_probability']:
                            batch_size_0 = 100
                            batch_size_1 = 100
                        else:
                            if False:
                                #max_samples = max(class_0_data.shape[0], class_1_data.shape[0])
                                #print('max_samples', max_samples)
                                fraction = min(class_0_data.shape[0], class_1_data.shape[0])/max(class_0_data.shape[0], class_1_data.shape[0]) #np.ceil(max_samples/100)
                                print('fraction', fraction)
                                if class_0_data.shape[0] > class_1_data.shape[0]:
                                    batch_size_0 = int(np.ceil(100/fraction))#int(np.ceil(class_0_data.shape[0] / fraction))
                                    batch_size_1 = 100#int(np.ceil(class_1_data.shape[0] / fraction))
                                else:
                                    batch_size_0 = 100#int(np.ceil(class_0_data.shape[0] / fraction))
                                    batch_size_1 = int(np.ceil(100/fraction))#int(np.ceil(class_1_data.shape[0] / fraction))                                
                                print('batch_size_0', batch_size_0, len(list(make_batch(class_0_data, batch_size_0))))
                                print('batch_size_1', batch_size_1, len(list(make_batch(class_1_data, batch_size_1))))
                            else:
                                pass
                            
                            
                        batch_list_feature_data = []
                        for batch_0, batch_1 in zip(make_batch(class_0_data, batch_size_0), make_batch(class_1_data, batch_size_1)):
                            batch = np.hstack([batch_0, batch_1])
                            np.random.seed(seed+i)
                            np.random.shuffle(batch)
                            batch_list_feature_data.append(batch)
                        feature_data = np.hstack(batch_list_feature_data)

                        #print('feature_weights', feature_weights)
                        batch_list_feature_weights = []
                        for batch_0, batch_1 in zip(make_batch(feature_0_weights, batch_size_0), make_batch(feature_1_weights, batch_size_1)):
                            batch = np.hstack([batch_0, batch_1])
                            np.random.seed(seed+i)
                            np.random.shuffle(batch)
                            batch_list_feature_weights.append(batch)
                        feature_weights = np.hstack(batch_list_feature_weights)                           
                                       
                    
                    X_data_list.append(feature_data)
                    feature_weights_list.append(feature_weights)

                    distribution_parameter_0_list_list.append(distribution_parameter_0_list)
                    distribution_parameter_1_list_list.append(distribution_parameter_1_list)

                    samples_class_0_distrib_list_list.append(samples_class_0_distrib_list)
                    samples_class_1_distrib_list_list.append(samples_class_1_distrib_list)          
                
                
                X_data = np.vstack(X_data_list).T.astype(np.float64)
                feature_weights = np.vstack(feature_weights_list).T

                X_data = X_data + X_data * np.random.uniform(-data_noise, data_noise, X_data.shape) #np.random.uniform(-flip_percentage, flip_percentage, X_data.shape)#         
                X_data, normalizer_list = normalize_real_world_data(X_data)

                if config['data']['weighted_data_generation']:
                    feature_weights_reduced = np.sum(feature_weights, axis=1)
                    threshold = np.median(feature_weights_reduced)
                    
                    y_data = deepcopy(feature_weights_reduced)
                    y_data[feature_weights_reduced >= threshold] = 1
                    y_data[feature_weights_reduced < threshold] = 0     
                else:
                    if config['data']['balanced_data']:
                        y_data = np.hstack([[0]*samples_class_0, [1]*samples_class_1])
                    else:
                        ratio = np.random.uniform(0.25, 0.75)
                        total_samples = samples_class_0 + samples_class_1
                        samples_class_0_y_data = int(np.floor(samples_class_0 * ratio))
                        samples_class_1_y_data = total_samples - samples_class_0_y_data
                        y_data = np.hstack([[0]*samples_class_0_y_data, [1]*samples_class_1_y_data])

                idx = np.random.choice(y_data.shape[0], int(y_data.shape[0]*flip_percentage), replace=False)
                y_data[idx] = (y_data[idx] + 1) % 2  
            
            
            if config['data']['exclude_linearly_seperable'] or config['data']['data_generation_filtering']:
                threshold = 0.9
                
                perceptron_test = True
                linear_discriminant_test = False
                dt_test = False
                
                if perceptron_test:
                    from sklearn.linear_model import Perceptron
                    perceptron = Perceptron(random_state = 0)
                    perceptron.fit(X_data, y_data)
                    y_data_perceptron = np.round(perceptron.predict(X_data)).astype(int) 

                    accuracy_perceptron = accuracy_score(y_data, y_data_perceptron)
                    
                    if config['data']['data_generation_filtering']:
                        condition = accuracy_perceptron > threshold
                    elif config['data']['exclude_linearly_seperable'] and not config['data']['data_generation_filtering']:
                        condition = accuracy_perceptron == 1
                        
                    if condition:
                        rand_int = np.random.randint(100_000, 1_000_000)
                        seed = (seed + rand_int) % (2**32 - 1)
                    else:
                        seed = config['computation']['RANDOM_SEED']                    
                elif linear_discriminant_test:
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    linear_discriminant =  LinearDiscriminantAnalysis(solver='eigen', tol=0, shrinkage=0)
                    linear_discriminant.fit(X_data, y_data)
                    y_data_linear_discriminant = np.round(linear_discriminant.predict(X_data)).astype(int) 

                    accuracy_linear_discriminant = accuracy_score(y_data, y_data_linear_discriminant)
                    condition = accuracy_linear_discriminant == 1

                    if False:
                        from sklearn.linear_model import Perceptron
                        perceptron = Perceptron(random_state = 0)
                        perceptron.fit(X_data, y_data)
                        y_data_perceptron = np.round(perceptron.predict(X_data)).astype(int) 

                        accuracy_perceptron = accuracy_score(y_data, y_data_perceptron)                        
                        print(accuracy_perceptron, accuracy_linear_discriminant)
                    if condition:
                        rand_int = np.random.randint(100_000, 1_000_000)
                        seed = (seed + rand_int) % (2**32 - 1)
                    else:
                        seed = config['computation']['RANDOM_SEED']                      
                        
                elif dt_test:
                    dt_model = DecisionTreeClassifier(max_depth = 2)
                    dt_model.fit(X_data, y_data)
                    
                    dt_model_preds = dt_model.predict(X_data)
                    accuracy_dt_model = accuracy_score(y_data, dt_model_preds)
                    
                    condition = accuracy_dt_model > threshold     
                    
                    if condition:
                        rand_int = np.random.randint(100_000, 1_000_000)
                        seed = (seed + rand_int) % (2**32 - 1)
                    else:
                        seed = config['computation']['RANDOM_SEED']          
            else:
                condition = False

                
                
                
        #print(distributions_per_class, distribution_parameter_list)
        
    #print('samples_class_0_distrib', samples_class_0_distrib, 'samples_class_1_distrib', samples_class_1_distrib)
    #print('samples_class_0', samples_class_0, 'samples_class_1', samples_class_1)
        
    #print('distribution_parameter_list', distribution_parameter_list)
        
    return X_data, y_data, distribution_parameter_list, normalizer_list


def distribution_evaluation_single_model_synthetic_data(loss_function, 
                                                        metrics,
                                                        #model,
                                                        config,
                                                        distribution_list_evaluation,
                                                        lambda_net_parameters_train,
                                                        mean_train_parameters,
                                                        std_train_parameters,    
                                                        data_seed=42,
                                                        max_distributions_per_class = 0,
                                                        random_parameters=None,
                                                        flip_percentage=0.0,
                                                        data_noise=0.0,
                                                        verbose=0
                                                        ):
    from utilities.InterpretationNet import load_inet
    model = load_inet(loss_function, metrics, config)
    if 'make_class' in config['data']['function_generation_type']:
        
        np.random.seed(data_seed)
        informative = 3#np.random.randint(config['data']['number_of_variables']//2, high=config['data']['number_of_variables']+1) #config['data']['number_of_variables']
        redundant = np.random.randint(0, high=config['data']['number_of_variables']-informative+1) #0
        repeated = config['data']['number_of_variables']-informative-redundant # 0

        n_clusters_per_class = min(informative//2+1, config['data']['max_distributions_per_class'])#max(2, np.random.randint(0, high=informative//2+1)) #2
        X_data, y_data, distribution_parameter_list  = make_classification_distribution(n_samples=config['data']['lambda_dataset_size'], 
                                                           n_features=config['data']['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                                           n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                                           n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                                           n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                                           n_classes=config['data']['num_classes'], 
                                                           n_clusters_per_class=n_clusters_per_class, 
                                                           #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                                           class_sep=0.5, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                                           hypercube=True, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                                           #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                                           #scale=1.0, #Multiply features by the specified value. 
                                                           shuffle=True, 
                                                           random_state=data_seed,
                                                           random_parameters=config['data']['random_parameters_distribution'],
                                                           distrib_param_max=config['data']['distrib_param_max']
                                                           )         
        
                
        normalizer_list = []
        for i, column in enumerate(X_data.T):
            scaler = MinMaxScaler()
            scaler.fit(column.reshape(-1, 1))
            X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)

    else:
        random.seed(data_seed)
        distributions_per_class = max_distributions_per_class#random.randint(1, max_distributions_per_class) if max_distributions_per_class != 0 else max_distributions_per_class
        
        X_data, y_data, distribution_parameter_list, normalizer_list = generate_dataset_from_distributions(config['data']['distribution_list'], 
                                                                                          config['data']['number_of_variables'], 
                                                                                          config['data']['lambda_dataset_size'],
                                                                                          distributions_per_class = distributions_per_class, 
                                                                                          seed=data_seed,
                                                                                          flip_percentage=flip_percentage,
                                                                                          data_noise=data_noise,
                                                                                          random_parameters=random_parameters,
                                                                                          config=config)
        

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_test_valid(X_data, y_data, valid_frac=0.25, test_frac=0.1, seed=config['computation']['RANDOM_SEED'])
    
    test_network, model_history = train_network_real_world_data(X_train, y_train, X_valid, y_valid, config, verbose=verbose)  
    
    distances_dict_list = []
    evaluation_result_dict_list = [] 
    results_list_list = []
    dt_inet_list = []
    dt_distilled_list_list = []
    
    for distribution_training in distribution_list_evaluation: #distribution_list
        evaluation_result_dict, results_list, test_network_parameters, dt_inet, dt_distilled_list = evaluate_network_real_world_data(model,
                                                                                                    test_network, 
                                                                                                    X_train, 
                                                                                                    X_test, 
                                                                                                    dataset_size_list=[10000, 'TRAINDATA', 'STANDARDUNIFORM', 'STANDARDNORMAL'],
                                                                                                    config=config,
                                                                                                    distribution=distribution_training)


        results_list_extended = results_list[0]

        results_list_extended['dt_scores']['soft_binary_crossentropy_train_data'] = results_list[-3]['dt_scores']['soft_binary_crossentropy']
        results_list_extended['dt_scores']['binary_crossentropy_train_data'] = results_list[-3]['dt_scores']['binary_crossentropy']
        results_list_extended['dt_scores']['accuracy_train_data'] = results_list[-3]['dt_scores']['accuracy']
        results_list_extended['dt_scores']['f1_score_train_data'] = results_list[-3]['dt_scores']['f1_score']
        results_list_extended['dt_scores']['roc_auc_score_train_data'] = results_list[-3]['dt_scores']['roc_auc_score']
        

        results_list_extended['dt_scores']['soft_binary_crossentropy_uniform_data'] = results_list[-2]['dt_scores']['soft_binary_crossentropy']
        results_list_extended['dt_scores']['binary_crossentropy_uniform_data'] = results_list[-2]['dt_scores']['binary_crossentropy']
        results_list_extended['dt_scores']['accuracy_uniform_data'] = results_list[-2]['dt_scores']['accuracy']
        results_list_extended['dt_scores']['f1_score_uniform_data'] = results_list[-2]['dt_scores']['f1_score']
        results_list_extended['dt_scores']['roc_auc_uniform_data'] = results_list[-2]['dt_scores']['roc_auc_score']
        

        results_list_extended['dt_scores']['soft_binary_crossentropy_normal_data'] = results_list[-1]['dt_scores']['soft_binary_crossentropy']
        results_list_extended['dt_scores']['binary_crossentropy_normal_data'] = results_list[-1]['dt_scores']['binary_crossentropy']
        results_list_extended['dt_scores']['accuracy_normal_data'] = results_list[-1]['dt_scores']['accuracy']
        results_list_extended['dt_scores']['f1_score_normal_data'] = results_list[-1]['dt_scores']['f1_score']   
        results_list_extended['dt_scores']['roc_auc_score_normal_data'] = results_list[-1]['dt_scores']['roc_auc_score']        

        results_list = results_list_extended
        evaluation_result_dict = results_list

        test_network_parameters = test_network_parameters#[:1]
        dt_inet = dt_inet#[:1]
        dt_distilled_list = dt_distilled_list#[:1]

        distances_dict = calculate_network_distance(mean=mean_train_parameters, 
                                                               std=std_train_parameters, 
                                                               network_parameters=test_network_parameters, 
                                                               lambda_net_parameters_train=lambda_net_parameters_train, 
                                                               config=config)

        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'X_test': X_test,
            'y_test': y_test,
        }
        
        distances_dict_list.append(distances_dict)
        evaluation_result_dict_list.append(evaluation_result_dict)
        results_list_list.append(results_list)
        dt_inet_list.append(dt_inet)
        dt_distilled_list_list.append(dt_distilled_list)    
        
    return (distances_dict_list, 
            evaluation_result_dict_list, 
            results_list_list, 
            dt_inet_list, 
            dt_distilled_list_list, 
            data_dict, 
            normalizer_list, 
            test_network.get_weights(), 
            model_history.history,
            distribution_parameter_list)




def evaluate_real_world_dataset(model,
                                dataset_size_list,
                                mean_train_parameters,
                                std_train_parameters,
                                lambda_net_parameters_train,
                                X_data, 
                                y_data, 
                                nominal_features, 
                                ordinal_features, 
                                config,
                                distribution_list_evaluation,
                                config_train_network=None,
                                force_rebalance=False):
    
    start_evaluate_network_complete = time.time()
    
    random.seed(config['computation']['RANDOM_SEED'])
    np.random.seed(config['computation']['RANDOM_SEED'])
    tf.random.set_seed(config['computation']['RANDOM_SEED'])
    
    print('Original Data Shape (selected): ', X_data.shape)
    
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), nominal_features)], remainder='passthrough', sparse_threshold=0)
    transformer.fit(X_data)

    X_data = transformer.transform(X_data)
    X_data = pd.DataFrame(X_data, columns=transformer.get_feature_names())

    for ordinal_feature in ordinal_features:
        X_data[ordinal_feature] = OrdinalEncoder().fit_transform(X_data[ordinal_feature].values.reshape(-1, 1)).flatten()

    X_data = X_data.astype(np.float64)
    
    print('Original Data Shape (encoded): ', X_data.shape)
    print('Original Data Class Distribution: ', y_data[y_data>=0.5].shape[0], ' (true) /', y_data[y_data<0.5].shape[0], ' (false)')
    
    if not config['i_net']['force_evaluate_real_world'] and X_data.shape[1] != config['data']['number_of_variables']:

        evaluation_result_dict_placeholder =  {
                                    'dt_scores': {},
                                    'inet_scores': {},  
                                    'model_scores': {},
                               } 
                
        for identifier in distribution_list_evaluation:
            evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['accuracy_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['f1_score_' + str(identifier)] = [np.nan] * len(dataset_size_list)    
            evaluation_result_dict_placeholder['dt_scores']['roc_auc_score_' + str(identifier)] = [np.nan] * len(dataset_size_list)    
            
            
            evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy_data_random_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy_data_random_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['accuracy_data_random_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['f1_score_data_random_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            evaluation_result_dict_placeholder['dt_scores']['roc_auc_score_data_random_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            
            
            evaluation_result_dict_placeholder['dt_scores']['runtime_' + str(identifier)] = [np.nan] * len(dataset_size_list)
            
            
        evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['accuracy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['f1_score'] = [np.nan] * len(dataset_size_list)        
        evaluation_result_dict_placeholder['dt_scores']['roc_auc_score'] = [np.nan] * len(dataset_size_list)        
    
        evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy_data_random'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy_data_random'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['accuracy_data_random'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['f1_score_data_random'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['roc_auc_score_data_random'] = [np.nan] * len(dataset_size_list)
        
        evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['accuracy_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['f1_score_std'] = [np.nan] * len(dataset_size_list) 
        evaluation_result_dict_placeholder['dt_scores']['roc_auc_score_std'] = [np.nan] * len(dataset_size_list)        
        
        evaluation_result_dict_placeholder['dt_scores']['soft_binary_crossentropy_data_random_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['binary_crossentropy_data_random_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['accuracy_data_random_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['f1_score_data_random_std'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['dt_scores']['roc_auc_score_data_random_std'] = [np.nan] * len(dataset_size_list)
        
        evaluation_result_dict_placeholder['dt_scores']['runtime'] = [np.nan] * len(dataset_size_list)
        
        
        evaluation_result_dict_placeholder['inet_scores']['soft_binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['inet_scores']['binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['inet_scores']['accuracy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['inet_scores']['f1_score'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['inet_scores']['roc_auc_score'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['inet_scores']['runtime'] = [np.nan] * len(dataset_size_list)        
                          
        evaluation_result_dict_placeholder['model_scores']['soft_binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['model_scores']['binary_crossentropy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['model_scores']['accuracy'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['model_scores']['f1_score'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['model_scores']['roc_auc_score'] = [np.nan] * len(dataset_size_list)
        evaluation_result_dict_placeholder['model_scores']['runtime'] = [np.nan] * len(dataset_size_list)        
                                                          

        results_placeholder =  {
                                'dt_scores': {},
                                'inet_scores': {},
                                'model_scores': {},
                               }  
        
        for identifier in distribution_list_evaluation:
            results_placeholder['dt_scores']['soft_binary_crossentropy_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['binary_crossentropy_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['accuracy_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['f1_score_' + str(identifier)] = np.nan  
            results_placeholder['dt_scores']['roc_auc_score_' + str(identifier)] = np.nan            
            
            results_placeholder['dt_scores']['soft_binary_crossentropy_data_random_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['binary_crossentropy_data_random_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['accuracy_data_random_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['f1_score_data_random_' + str(identifier)] = np.nan
            results_placeholder['dt_scores']['roc_auc_score_data_random_' + str(identifier)] = np.nan
            
            results_placeholder['dt_scores']['runtime_' + str(identifier)] = np.nan
        
        results_placeholder['dt_scores']['soft_binary_crossentropy'] = np.nan
        results_placeholder['dt_scores']['binary_crossentropy'] = np.nan
        results_placeholder['dt_scores']['accuracy'] = np.nan
        results_placeholder['dt_scores']['f1_score'] = np.nan
        results_placeholder['dt_scores']['roc_auc_score'] = np.nan
        
        results_placeholder['dt_scores']['soft_binary_crossentropy_data_random'] = np.nan
        results_placeholder['dt_scores']['binary_crossentropy_data_random'] = np.nan
        results_placeholder['dt_scores']['accuracy_data_random'] = np.nan
        results_placeholder['dt_scores']['f1_score_data_random'] = np.nan
        results_placeholder['dt_scores']['roc_auc_score_data_random'] = np.nan
        
        results_placeholder['dt_scores']['soft_binary_crossentropy_std'] = np.nan
        results_placeholder['dt_scores']['binary_crossentropy_std'] = np.nan
        results_placeholder['dt_scores']['accuracy_std'] = np.nan
        results_placeholder['dt_scores']['f1_score_std'] = np.nan
        results_placeholder['dt_scores']['roc_auc_score_std'] = np.nan
        
        results_placeholder['dt_scores']['soft_binary_crossentropy_data_random_std'] = np.nan
        results_placeholder['dt_scores']['binary_crossentropy_data_random_std'] = np.nan
        results_placeholder['dt_scores']['accuracy_data_random_std'] = np.nan
        results_placeholder['dt_scores']['f1_score_data_random_std'] = np.nan
        results_placeholder['dt_scores']['roc_auc_score_data_random_std'] = np.nan
        
        results_placeholder['dt_scores']['runtime'] = np.nan
        
        
        results_placeholder['inet_scores']['soft_binary_crossentropy'] = np.nan
        results_placeholder['inet_scores']['binary_crossentropy'] = np.nan
        results_placeholder['inet_scores']['accuracy'] = np.nan
        results_placeholder['inet_scores']['f1_score'] = np.nan
        results_placeholder['inet_scores']['roc_auc_score'] = np.nan
        results_placeholder['inet_scores']['runtime'] = np.nan 
        
        results_placeholder['model_scores']['soft_binary_crossentropy'] = np.nan
        results_placeholder['model_scores']['binary_crossentropy'] = np.nan
        results_placeholder['model_scores']['accuracy'] = np.nan
        results_placeholder['model_scores']['f1_score'] = np.nan
        results_placeholder['model_scores']['roc_auc_score'] = np.nan
        results_placeholder['model_scores']['runtime'] = np.nan         
        
        distances_dict_placeholder = {
                'z_score_aggregate': np.nan,
                'distance_to_initialization_aggregate': np.nan,
                'distance_to_sample_average': np.nan,
                'distance_to_sample_min': np.nan,
                'max_distance_to_neuron_average': np.nan,
                'max_distance_to_neuron_min': np.nan,     
            }     

        return distances_dict_placeholder, evaluation_result_dict_placeholder, [results_placeholder for _ in range(len(dataset_size_list))], None, None, None, None, None
    
    X_data = adjust_data_to_number_of_variables(X_data=X_data, 
                                                y_data=y_data, 
                                                number_of_variables=config['data']['number_of_variables'],
                                                seed=config['computation']['RANDOM_SEED'])
    X_data, normalizer_list = normalize_real_world_data(X_data)

    (X_train, 
     y_train, 
     X_valid, 
     y_valid, 
     X_test, 
     y_test) = split_train_test_valid(X_data, 
                                      y_data, 
                                      seed=config['computation']['RANDOM_SEED'],
                                      verbose=1)

    if force_rebalance:
        X_train, y_train = rebalance_data(X_train, 
                                          y_train, 
                                          balance_ratio=0.5, 
                                          strategy=config['i_net']['resampling_strategy'])    
    else:
    
        X_train, y_train = rebalance_data(X_train, 
                                          y_train, 
                                          balance_ratio=config['i_net']['resampling_threshold'], 
                                          strategy=config['i_net']['resampling_strategy'])

    if config_train_network == None:
        config_train_network = config

    start_network_training = time.time()
 
    test_network, model_history = train_network_real_world_data(X_train, 
                                                                y_train, 
                                                                X_valid, 
                                                                y_valid, 
                                                                config_train_network,
                                                                verbose=1)
      
    end_network_training = time.time()
    runtime_model = end_network_training-start_network_training
    print('Training Network: ', format(runtime_model))      
    
    evaluation_result_dict = {
                              'dt_scores': {},
                              'inet_scores': {},
                              'model_scores': {},
                             }
    
    dt_distilled_list = []
    #results_list = []
    
    results_list = []
    for _ in range(len(dataset_size_list)):
        result_dict = {
                          'dt_scores': {},
                          'inet_scores': {},
                          'model_scores': {},
                         }
        results_list.append(result_dict)
    
    soft_binary_crossentropy_list = []
    binary_crossentropy_list = []
    accuracy_list = []
    f1_score_list = []
    roc_auc_score_list = []
    
    soft_binary_crossentropy_list_data_random = []
    binary_crossentropy_list_data_random = []
    accuracy_list_data_random = []    
    f1_score_list_data_random = []
    roc_auc_score_list_data_random = []
    
    runtime_list = []
    
    

    parallel_distrib_evaluation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=0, backend='loky') #loky #sequential multiprocessing

    test_network_parameters_array = shaped_network_parameters_to_array(test_network.get_weights(), config)
    
    start_evaluate_network = time.time()
    
    parallel_results = parallel_distrib_evaluation(delayed(evaluate_network_real_world_data_parallel)(dill.dumps(model.loss),
                                                                                                    dill.dumps([metric._fn for metric in model.metrics[1:]]),
                                                                                                    test_network_parameters_array, 
                                                                                                    X_train, 
                                                                                                    X_test, 
                                                                                                    dataset_size_list,
                                                                                                    config,
                                                                                                    distribution=distribution_training,
                                                                                                    verbosity=1) for distribution_training in distribution_list_evaluation)      

    del parallel_distrib_evaluation    
    
    end_evaluate_network = time.time()
    print('Evaluate Network: ', format(end_evaluate_network-start_evaluate_network))  
    
    
    model_preds = test_network.predict(X_test).ravel()
    
    soft_binary_crossentropy_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_test.values, model_preds)).numpy()  
    binary_crossentropy_model = log_loss(np.round(y_test.values), model_preds, labels=[0,1])
    accuracy_model = accuracy_score(np.round(y_test.values), np.round(model_preds))
    f1_score_model = f1_score(np.round(y_test.values), np.round(model_preds), average='weighted')  
    try:
        roc_auc_score_model = roc_auc_score(np.round(y_test.values), model_preds, average='weighted')      
    except ValueError:
        roc_auc_score_model = 0

    for i, distribution_training in enumerate(distribution_list_evaluation):

        
        #evaluation_result_dict_distrib, results_list_distrib, test_network_parameters, dt_inet, dt_distilled_list_distrib = evaluate_network_real_world_data(model,
        #                                                                                            test_network, 
        #                                                                                            X_train, 
        #                                                                                            X_test, 
        #                                                                                            dataset_size_list,
        #                                                                                            config,
        #                                                                                            distribution=distribution_training,
        #                                                                                            verbosity=1)
        
        
        evaluation_result_dict_distrib = parallel_results[i][0]
        results_list_distrib = parallel_results[i][1] 
        test_network_parameters = parallel_results[i][2]
        dt_inet = parallel_results[i][3] 
        dt_distilled_list_distrib = parallel_results[i][4]
        
        for result, results_distrib in zip(results_list, results_list_distrib):
            result['dt_scores']['soft_binary_crossentropy_' + str(distribution_training)] = results_distrib['dt_scores']['soft_binary_crossentropy']
            result['dt_scores']['binary_crossentropy_' + str(distribution_training)] = results_distrib['dt_scores']['binary_crossentropy']
            result['dt_scores']['accuracy_' + str(distribution_training)] = results_distrib['dt_scores']['accuracy']
            result['dt_scores']['f1_score_' + str(distribution_training)] = results_distrib['dt_scores']['f1_score']
            result['dt_scores']['roc_auc_score_' + str(distribution_training)] = results_distrib['dt_scores']['roc_auc_score']
            
            result['dt_scores']['soft_binary_crossentropy_data_random_' + str(distribution_training)] = results_distrib['dt_scores']['soft_binary_crossentropy_data_random']
            result['dt_scores']['binary_crossentropy_data_random_' + str(distribution_training)] = results_distrib['dt_scores']['binary_crossentropy_data_random']
            result['dt_scores']['accuracy_data_random_' + str(distribution_training)] = results_distrib['dt_scores']['accuracy_data_random']
            result['dt_scores']['f1_score_data_random_' + str(distribution_training)] = results_distrib['dt_scores']['f1_score_data_random']
            result['dt_scores']['roc_auc_score_data_random_' + str(distribution_training)] = results_distrib['dt_scores']['roc_auc_score_data_random']
            
            result['dt_scores']['runtime_' + str(distribution_training)] = results_distrib['dt_scores']['runtime']


            result['inet_scores']['soft_binary_crossentropy'] = results_distrib['inet_scores']['soft_binary_crossentropy']
            result['inet_scores']['binary_crossentropy'] = results_distrib['inet_scores']['binary_crossentropy']
            result['inet_scores']['accuracy'] = results_distrib['inet_scores']['accuracy']
            result['inet_scores']['f1_score'] = results_distrib['inet_scores']['f1_score']
            result['inet_scores']['roc_auc_score'] = results_distrib['inet_scores']['roc_auc_score']           
            result['inet_scores']['runtime'] = results_distrib['inet_scores']['runtime']          
            
            result['model_scores']['soft_binary_crossentropy'] = soft_binary_crossentropy_model
            result['model_scores']['binary_crossentropy'] = binary_crossentropy_model
            result['model_scores']['accuracy'] = accuracy_model
            result['model_scores']['f1_score'] = f1_score_model
            result['model_scores']['roc_auc_score'] = roc_auc_score_model
            result['model_scores']['runtime'] = runtime_model
            
            
        soft_binary_crossentropy_list.append(evaluation_result_dict_distrib['dt_scores']['soft_binary_crossentropy'])
        binary_crossentropy_list.append(evaluation_result_dict_distrib['dt_scores']['binary_crossentropy'])
        accuracy_list.append(evaluation_result_dict_distrib['dt_scores']['accuracy'])
        f1_score_list.append(evaluation_result_dict_distrib['dt_scores']['f1_score'])
        roc_auc_score_list.append(evaluation_result_dict_distrib['dt_scores']['roc_auc_score'])

        soft_binary_crossentropy_list_data_random.append(evaluation_result_dict_distrib['dt_scores']['soft_binary_crossentropy_data_random'])
        binary_crossentropy_list_data_random.append(evaluation_result_dict_distrib['dt_scores']['binary_crossentropy_data_random'])
        accuracy_list_data_random.append(evaluation_result_dict_distrib['dt_scores']['accuracy_data_random'])
        f1_score_list_data_random.append(evaluation_result_dict_distrib['dt_scores']['f1_score_data_random'])
        roc_auc_score_list_data_random.append(evaluation_result_dict_distrib['dt_scores']['roc_auc_score_data_random'])
        
        runtime_list.append(evaluation_result_dict_distrib['dt_scores']['runtime'])    
        
        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['soft_binary_crossentropy']
        evaluation_result_dict['dt_scores']['binary_crossentropy_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['binary_crossentropy']
        evaluation_result_dict['dt_scores']['accuracy_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['accuracy']
        evaluation_result_dict['dt_scores']['f1_score_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['f1_score']
        evaluation_result_dict['dt_scores']['roc_auc_score_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['roc_auc_score']
        
        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_data_random_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['soft_binary_crossentropy_data_random']
        evaluation_result_dict['dt_scores']['binary_crossentropy_data_random_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['binary_crossentropy_data_random']
        evaluation_result_dict['dt_scores']['accuracy_data_random_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['accuracy_data_random']
        evaluation_result_dict['dt_scores']['f1_score_data_random_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['f1_score_data_random']
        evaluation_result_dict['dt_scores']['roc_auc_score_data_random_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['roc_auc_score_data_random']
        
        evaluation_result_dict['dt_scores']['runtime_' + str(distribution_training)] = evaluation_result_dict_distrib['dt_scores']['runtime']
        
        
        evaluation_result_dict['inet_scores']['soft_binary_crossentropy'] = evaluation_result_dict_distrib['inet_scores']['soft_binary_crossentropy']
        evaluation_result_dict['inet_scores']['binary_crossentropy'] = evaluation_result_dict_distrib['inet_scores']['binary_crossentropy']
        evaluation_result_dict['inet_scores']['accuracy'] = evaluation_result_dict_distrib['inet_scores']['accuracy']
        evaluation_result_dict['inet_scores']['f1_score'] = evaluation_result_dict_distrib['inet_scores']['f1_score']
        evaluation_result_dict['inet_scores']['roc_auc_score'] = evaluation_result_dict_distrib['inet_scores']['roc_auc_score']
        evaluation_result_dict['inet_scores']['runtime'] = evaluation_result_dict_distrib['inet_scores']['runtime']
        
        evaluation_result_dict['model_scores']['soft_binary_crossentropy'] = [soft_binary_crossentropy_model] * len(dataset_size_list)       
        evaluation_result_dict['model_scores']['binary_crossentropy'] = [binary_crossentropy_model] * len(dataset_size_list)       
        evaluation_result_dict['model_scores']['accuracy'] = [accuracy_model] * len(dataset_size_list)       
        evaluation_result_dict['model_scores']['f1_score'] = [f1_score_model] * len(dataset_size_list)      
        evaluation_result_dict['model_scores']['roc_auc_score'] = [roc_auc_score_model] * len(dataset_size_list)        
        evaluation_result_dict['model_scores']['runtime'] = [runtime_model] * len(dataset_size_list)           
        
        dt_distilled_list.append(dt_distilled_list_distrib)
        #results_list.append(results_list_distrib)
        
    for result in results_list:
        soft_binary_crossentropy_list_result = []
        binary_crossentropy_list_result = []
        accuracy_list_result = []
        f1_score_list_result = []
        roc_auc_score_list_result = []

        soft_binary_crossentropy_list_data_random_result = []
        binary_crossentropy_list_data_random_result = []
        accuracy_list_data_random_result = []
        f1_score_list_data_random_result = [] 
        roc_auc_score_list_data_random_result = []     
        
        runtime_list_result = []
        
        for distribution in distribution_list_evaluation:
            soft_binary_crossentropy_list_result.append(result['dt_scores']['soft_binary_crossentropy_' + str(distribution)])
            binary_crossentropy_list_result.append(result['dt_scores']['binary_crossentropy_' + str(distribution)])
            accuracy_list_result.append(result['dt_scores']['accuracy_' + str(distribution)])
            f1_score_list_result.append(result['dt_scores']['f1_score_' + str(distribution)])
            roc_auc_score_list_result.append(result['dt_scores']['roc_auc_score_' + str(distribution)])
            
            
            soft_binary_crossentropy_list_data_random_result.append(result['dt_scores']['soft_binary_crossentropy_data_random_' + str(distribution)])
            binary_crossentropy_list_data_random_result.append(result['dt_scores']['binary_crossentropy_data_random_' + str(distribution)])
            accuracy_list_data_random_result.append(result['dt_scores']['accuracy_data_random_' + str(distribution)])
            f1_score_list_data_random_result.append(result['dt_scores']['f1_score_data_random_' + str(distribution)])
            roc_auc_score_list_data_random_result.append(result['dt_scores']['roc_auc_score_data_random_' + str(distribution)])
                      
            runtime_list_result.append(result['dt_scores']['runtime_' + str(distribution)])
            
        result['dt_scores']['soft_binary_crossentropy'] = np.mean(soft_binary_crossentropy_list_result)
        result['dt_scores']['binary_crossentropy'] = np.mean(binary_crossentropy_list_result)
        result['dt_scores']['accuracy'] = np.mean(accuracy_list_result)
        result['dt_scores']['f1_score'] = np.mean(f1_score_list_result)
        result['dt_scores']['roc_auc_score'] = np.mean(roc_auc_score_list_result)

        result['dt_scores']['soft_binary_crossentropy_data_random'] = np.mean(soft_binary_crossentropy_list_data_random_result)
        result['dt_scores']['binary_crossentropy_data_random'] = np.mean(binary_crossentropy_list_data_random_result)
        result['dt_scores']['accuracy_data_random'] = np.mean(accuracy_list_data_random_result)
        result['dt_scores']['f1_score_data_random'] = np.mean(f1_score_list_data_random_result)
        result['dt_scores']['roc_auc_score_data_random'] = np.mean(roc_auc_score_list_data_random_result)
        
        result['dt_scores']['soft_binary_crossentropy_std'] = np.std(soft_binary_crossentropy_list_result)
        result['dt_scores']['binary_crossentropy_std'] = np.std(binary_crossentropy_list_result)
        result['dt_scores']['accuracy_std'] = np.std(accuracy_list_result)
        result['dt_scores']['f1_score_std'] = np.std(f1_score_list_result)
        result['dt_scores']['roc_auc_score_std'] = np.std(roc_auc_score_list_result)

        result['dt_scores']['soft_binary_crossentropy_data_random_std'] = np.std(soft_binary_crossentropy_list_data_random_result)
        result['dt_scores']['binary_crossentropy_data_random_std'] = np.std(binary_crossentropy_list_data_random_result)
        result['dt_scores']['accuracy_data_random_std'] = np.std(accuracy_list_data_random_result)
        result['dt_scores']['f1_score_data_random_std'] = np.std(f1_score_list_data_random_result)
        result['dt_scores']['roc_auc_score_data_random_std'] = np.std(roc_auc_score_list_data_random_result)
        
        result['dt_scores']['runtime'] = np.mean(runtime_list_result)

    counter = 0
    
    if "TRAINDATA" in dataset_size_list:
        counter += 1
    if "STANDARDUNIFORM" in dataset_size_list:
        counter += 1
    if "STANDARDNORMAL" in dataset_size_list:
        counter += 1   
        
    #print('counter', counter)
    #print('soft_binary_crossentropy_list', soft_binary_crossentropy_list)
    #print('np.mean(soft_binary_crossentropy_list, axis=0)', np.mean(soft_binary_crossentropy_list, axis=0))
    #print('np.mean(soft_binary_crossentropy_list[:-counter], axis=0)', np.mean(soft_binary_crossentropy_list[:,:-counter], axis=0))
        
    if False:
        evaluation_result_dict['dt_scores']['soft_binary_crossentropy'] = np.mean(np.array(soft_binary_crossentropy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy'] = np.mean(np.array(binary_crossentropy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['accuracy'] = np.mean(np.array(accuracy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['f1_score'] = np.mean(np.array(f1_score_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score'] = np.mean(np.array(roc_auc_score_list)[:,:-counter], axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_data_random'] = np.mean(np.array(soft_binary_crossentropy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_data_random'] = np.mean(np.array(binary_crossentropy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['accuracy_data_random'] = np.mean(np.array(accuracy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['f1_score_data_random'] = np.mean(np.array(f1_score_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_data_random'] = np.mean(np.array(roc_auc_score_list_data_random)[:,:-counter], axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_std'] = np.std(np.array(soft_binary_crossentropy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_std'] = np.std(np.array(binary_crossentropy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['accuracy_std'] = np.std(np.array(accuracy_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['f1_score_std'] = np.std(np.array(f1_score_list)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_std'] = np.std(np.array(roc_auc_score_list)[:,:-counter], axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_data_random_std'] = np.std(np.array(soft_binary_crossentropy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_data_random_std'] = np.std(np.array(binary_crossentropy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['accuracy_data_random_std'] = np.std(np.array(accuracy_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['f1_score_data_random_std'] = np.std(np.array(f1_score_list_data_random)[:,:-counter], axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_data_random_std'] = np.std(np.array(roc_auc_score_list_data_random)[:,:-counter], axis=0)    

        evaluation_result_dict['dt_scores']['runtime'] = np.mean(np.array(runtime_list)[:,:-counter], axis=0)    
    else:
        evaluation_result_dict['dt_scores']['soft_binary_crossentropy'] = np.mean(soft_binary_crossentropy_list, axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy'] = np.mean(binary_crossentropy_list, axis=0)
        evaluation_result_dict['dt_scores']['accuracy'] = np.mean(accuracy_list, axis=0)
        evaluation_result_dict['dt_scores']['f1_score'] = np.mean(f1_score_list, axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score'] = np.mean(roc_auc_score_list, axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_data_random'] = np.mean(soft_binary_crossentropy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_data_random'] = np.mean(binary_crossentropy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['accuracy_data_random'] = np.mean(accuracy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['f1_score_data_random'] = np.mean(f1_score_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_data_random'] = np.mean(roc_auc_score_list_data_random, axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_std'] = np.std(soft_binary_crossentropy_list, axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_std'] = np.std(binary_crossentropy_list, axis=0)
        evaluation_result_dict['dt_scores']['accuracy_std'] = np.std(accuracy_list, axis=0)
        evaluation_result_dict['dt_scores']['f1_score_std'] = np.std(f1_score_list, axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_std'] = np.std(roc_auc_score_list, axis=0)

        evaluation_result_dict['dt_scores']['soft_binary_crossentropy_data_random_std'] = np.std(soft_binary_crossentropy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['binary_crossentropy_data_random_std'] = np.std(binary_crossentropy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['accuracy_data_random_std'] = np.std(accuracy_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['f1_score_data_random_std'] = np.std(f1_score_list_data_random, axis=0)
        evaluation_result_dict['dt_scores']['roc_auc_score_data_random_std'] = np.std(roc_auc_score_list_data_random, axis=0)      

        evaluation_result_dict['dt_scores']['runtime'] = np.mean(runtime_list, axis=0)           
    
    distances_dict = calculate_network_distance(mean=mean_train_parameters, 
                                                           std=std_train_parameters, 
                                                           network_parameters=test_network_parameters, 
                                                           lambda_net_parameters_train=lambda_net_parameters_train, 
                                                           config=config)

    
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test,
    }
    
    end_evaluate_network_complete = time.time()
    print('Evaluate Network Complete: ', format(end_evaluate_network_complete-start_evaluate_network_complete))      
        
    return distances_dict, evaluation_result_dict, results_list, dt_inet, dt_distilled_list, data_dict, normalizer_list, test_network


def evaluate_interpretation_net_prediction_single_sample(lambda_net_parameters_array, 
                                                         dt_inet, 
                                                         X_test_lambda, 
                                                         #y_test_lambda,
                                                         config,
                                                         train_data=None,
                                                         distribution=None,
                                                         verbosity=0):

    from utilities.metrics import calculate_function_value_from_decision_tree_parameters_wrapper, calculate_function_value_from_vanilla_decision_tree_parameters_wrapper
    
    if distribution is None:
        distribution = config['evaluation']['random_evaluation_dataset_distribution']

    #ALWAYS PASS TRAIN DATA FOR EVALUATION
    
    lambda_net = network_parameters_to_network(lambda_net_parameters_array, config, base_model=None)  
    
    if train_data is None:    
        try:
            distrib_param_max = config['data']['distrib_param_max']
        except:
            distrib_param_max = 1     
            
        condition = True
        counter = 0
        while condition and counter < 100:
            
            X_data_random = generate_random_data_points_custom(config['data']['x_min'], 
                                                               config['data']['x_max'],
                                                               config['evaluation']['per_network_optimization_dataset_size'], 
                                                               config['data']['number_of_variables'], 
                                                               config['data']['categorical_indices'],
                                                               distrib=distribution,
                                                               random_parameters=config['data']['random_parameters_distribution'],
                                                               distrib_param_max=config['data']['distrib_param_max'],
                                                               seed=config['computation']['RANDOM_SEED'],
                                                               config=config)
            
            
            
            y_data_random_lambda_pred = np.round(np.nan_to_num(lambda_net.predict(X_data_random).ravel()))
            condition = len(np.unique(y_data_random_lambda_pred)) == 1
            
            counter += 1
            
            if not config['evaluation']['optimize_sampling']:
                condition = False ##comment out to ignore loop
        
    else:
        X_data_random = train_data
    
    y_data_random_lambda_pred = np.nan_to_num(lambda_net.predict(X_data_random).ravel())
    y_test_lambda_pred = np.nan_to_num(lambda_net.predict(X_test_lambda).ravel())

    if config['i_net']['nas']:
        dt_inet = dt_inet[:config['function_family']['function_representation_length']]

    y_test_lambda_pred_diff = tf.math.subtract(1.0, y_test_lambda_pred)
    y_test_lambda_pred_softmax = tf.stack([y_test_lambda_pred, y_test_lambda_pred_diff], axis=1)         

    #print(dt_inet.shape)
    #print(dt_inet)

    if config['function_family']['dt_type'] == 'SDT':
        y_test_inet_dt, _  = calculate_function_value_from_decision_tree_parameters_wrapper(X_test_lambda, config)(dt_inet)
        y_test_inet_dt = np.nan_to_num(y_test_inet_dt.numpy())
    elif config['function_family']['dt_type'] == 'vanilla':
        y_test_inet_dt, _  = calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(X_test_lambda, config)(dt_inet)
        y_test_inet_dt = np.nan_to_num(y_test_inet_dt.numpy())

    y_test_inet_dt_diff = tf.math.subtract(1.0, y_test_inet_dt)
    y_test_inet_dt_softmax = tf.stack([y_test_inet_dt, y_test_inet_dt_diff], axis=1)       

    soft_binary_crossentropy_inet_dt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_test_lambda_pred_softmax, y_test_inet_dt_softmax)).numpy()  
    binary_crossentropy_inet_dt = log_loss(np.round(y_test_lambda_pred), y_test_inet_dt, labels=[0,1])
    accuracy_inet_dt = accuracy_score(np.round(y_test_lambda_pred), np.round(y_test_inet_dt))
    f1_score_inet_dt = f1_score(np.round(y_test_lambda_pred), np.round(y_test_inet_dt), average='weighted')        
    try:
        roc_auc_score_inet_dt = roc_auc_score(np.round(y_test_lambda_pred), y_test_inet_dt, average='weighted')      
    except ValueError:
        roc_auc_score_inet_dt = 0    
    

    if config['evaluation']['per_network_optimization_dataset_size'] > 50_000 and config['function_family']['dt_type'] == 'SDT': 

        results =  {
                        #'function_values': {
                        #    'y_test_inet_dt': y_test_inet_dt,
                        #    'y_test_distilled_dt': None,
                        #},
                        'dt_scores': {
                            'soft_binary_crossentropy': np.nan,
                            'soft_binary_crossentropy_data_random': np.nan,                            
                            'binary_crossentropy': np.nan,
                            'binary_crossentropy_data_random': np.nan,
                            'accuracy': np.nan,
                            'accuracy_data_random': np.nan,
                            'f1_score': np.nan,   
                            'f1_score_data_random': np.nan,   
                            'roc_auc_score': np.nan,   
                            'roc_auc_score_data_random': np.nan,                               
                            'runtime': np.nan
                        },
                    'inet_scores': {
                        'soft_binary_crossentropy': np.nan_to_num(soft_binary_crossentropy_inet_dt),
                        'binary_crossentropy': np.nan_to_num(binary_crossentropy_inet_dt),
                        'accuracy': np.nan_to_num(accuracy_inet_dt),
                        'f1_score': np.nan_to_num(f1_score_inet_dt),         
                        'roc_auc_score': np.nan_to_num(roc_auc_score_inet_dt),           
                        #'runtime': inet_runtime
                    },      
                        
                   }



        return results, None        




    start_dt_distilled = time.time() 

    dt_distilled = generate_random_decision_tree(config, config['computation']['RANDOM_SEED'], verbosity=verbosity)
    if config['function_family']['dt_type'] == 'SDT':
        dt_distilled.fit(X_data_random, np.round(y_data_random_lambda_pred).astype(np.int64), epochs=50)

        end_dt_distilled = time.time()     
        dt_distilled_runtime = (end_dt_distilled - start_dt_distilled)        

        y_data_random_distilled_dt = dt_distilled.predict_proba(X_data_random).ravel()
        y_test_distilled_dt = dt_distilled.predict_proba(X_test_lambda).ravel()

        #tf.print('y_data_random_distilled_dt', y_data_random_distilled_dt, summarize=-1)
        #tf.print('y_test_distilled_dt', y_test_distilled_dt, summarize=-1)

    elif config['function_family']['dt_type'] == 'vanilla':
        dt_distilled.fit(X_data_random, np.round(y_data_random_lambda_pred).astype(np.int64))

        end_dt_distilled = time.time()     
        dt_distilled_runtime = (end_dt_distilled - start_dt_distilled)     
        y_data_random_distilled_dt = dt_distilled.predict_proba(X_data_random)
        if y_data_random_distilled_dt.shape[1] > 1:
            y_data_random_distilled_dt = y_data_random_distilled_dt[:,1:].ravel()
        else:
            y_data_random_distilled_dt = dt_distilled.predict(X_data_random)#y_data_random_distilled_dt.ravel()

        y_test_distilled_dt = dt_distilled.predict_proba(X_test_lambda)
        if y_test_distilled_dt.shape[1] > 1:
            y_test_distilled_dt = y_test_distilled_dt[:,1:].ravel()
        else:
            y_test_distilled_dt = dt_distilled.predict(X_test_lambda)# y_test_distilled_dt.ravel()            


    epsilon = 1e-6

    #print(dt_inet)
    #print('y_test_lambda_pred[:10]', y_test_lambda_pred[:10])
    #print('y_test_inet_dt[:10]', y_test_inet_dt[:10])
    #print('y_test_distilled_dt[:10]', y_test_distilled_dt[:10])        


    y_test_distilled_dt_diff = tf.math.subtract(1.0, y_test_distilled_dt)
    y_test_distilled_dt_softmax = tf.stack([y_test_distilled_dt, y_test_distilled_dt_diff], axis=1)

    y_data_random_lambda_pred_diff = tf.math.subtract(1.0, y_data_random_lambda_pred)
    y_data_random_lambda_pred_softmax = tf.stack([y_data_random_lambda_pred, y_data_random_lambda_pred_diff], axis=1)

    y_data_random_distilled_dt_diff = tf.math.subtract(1.0, y_data_random_distilled_dt)
    y_data_random_distilled_dt_softmax = tf.stack([y_data_random_distilled_dt, y_data_random_distilled_dt_diff], axis=1)


    soft_binary_crossentropy_distilled_dt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_test_lambda_pred_softmax, y_test_distilled_dt_softmax)).numpy()    
    binary_crossentropy_distilled_dt = log_loss(np.round(y_test_lambda_pred), y_test_distilled_dt, labels=[0,1])
    accuracy_distilled_dt = accuracy_score(np.round(y_test_lambda_pred), np.round(y_test_distilled_dt))
    f1_score_distilled_dt = f1_score(np.round(y_test_lambda_pred), np.round(y_test_distilled_dt), average='weighted')   
    try:
        roc_auc_score_distilled_dt = roc_auc_score(np.round(y_test_lambda_pred), y_test_distilled_dt, average='weighted')
    except ValueError:
        roc_auc_score_distilled_dt = 0        

    soft_binary_crossentropy_data_random_distilled_dt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_data_random_lambda_pred_softmax, y_data_random_distilled_dt_softmax)).numpy()
    binary_crossentropy_data_random_distilled_dt = log_loss(np.round(y_data_random_lambda_pred), y_data_random_distilled_dt, labels=[0,1])
    accuracy_data_random_distilled_dt = accuracy_score(np.round(y_data_random_lambda_pred), np.round(y_data_random_distilled_dt))
    f1_score_data_random_distilled_dt = f1_score(np.round(y_data_random_lambda_pred), np.round(y_data_random_distilled_dt), average='weighted')
    try:
        roc_auc_score_data_random_distilled_dt = roc_auc_score(np.round(y_data_random_lambda_pred), y_data_random_distilled_dt, average='weighted')    
    except ValueError:
        roc_auc_score_data_random_distilled_dt = 0          
    
    #if train_data is not None:
        #print('np.round(y_test_lambda_pred)',np.round(y_test_lambda_pred))
        #print('accuracy_data_random_distilled_dt, accuracy_distilled_dt', accuracy_data_random_distilled_dt, accuracy_distilled_dt)

    #soft_binary_crossentropy_distilled_dt_median = tfp.stats.percentile(tf.nn.softmax_cross_entropy_with_logits(y_test_lambda_pred_softmax, y_test_distilled_dt_softmax), 50.0, interpolation='midpoint').numpy()    
    #binary_crossentropy_distilled_dt_median = tf.keras.losses.get('binary_crossentropy')(np.round(tf.reshape(y_test_lambda_pred, [-1, 1])), tf.reshape(y_test_distilled_dt, [-1, 1]), labels=[0,1])  

    #soft_binary_crossentropy_data_random_distilled_dt_median = tfp.stats.percentile(tf.nn.softmax_cross_entropy_with_logits(y_data_random_lambda_pred_softmax, y_data_random_distilled_dt_softmax), 50.0, interpolation='midpoint').numpy()
    #binary_crossentropy_data_random_distilled_dt_median = tf.keras.losses.get('binary_crossentropy')(np.round(tf.reshape(y_data_random_lambda_pred, [-1, 1])), tf.reshape(y_data_random_distilled_dt, [-1, 1]), labels=[0,1])   

    #soft_binary_crossentropy_inet_dt_median = tfp.stats.percentile(tf.nn.softmax_cross_entropy_with_logits(y_test_lambda_pred_softmax, y_test_inet_dt_softmax), 50.0, interpolation='midpoint').numpy()  
    #binary_crossentropy_inet_dt_median = tf.keras.losses.get('binary_crossentropy')(np.round(tf.reshape(y_test_lambda_pred, [-1, 1])), tf.reshape(y_test_inet_dt, [-1, 1]), labels=[0,1])     


    results =  {
                    #'function_values': {
                    #    'y_test_inet_dt': y_test_inet_dt,
                    #    'y_test_distilled_dt': y_test_distilled_dt,
                    #},
                    'dt_scores': {
                        'soft_binary_crossentropy': np.nan_to_num(soft_binary_crossentropy_distilled_dt),
                        'soft_binary_crossentropy_data_random': np.nan_to_num(soft_binary_crossentropy_data_random_distilled_dt),                            
                        'binary_crossentropy': np.nan_to_num(binary_crossentropy_distilled_dt),
                        'binary_crossentropy_data_random': np.nan_to_num(binary_crossentropy_data_random_distilled_dt),
                        'accuracy': np.nan_to_num(accuracy_distilled_dt),
                        'accuracy_data_random': np.nan_to_num(accuracy_data_random_distilled_dt),
                        'f1_score': np.nan_to_num(f1_score_distilled_dt),   
                        'f1_score_data_random': np.nan_to_num(f1_score_data_random_distilled_dt),
                        'roc_auc_score': np.nan_to_num(roc_auc_score_distilled_dt),   
                        'roc_auc_score_data_random': np.nan_to_num(roc_auc_score_data_random_distilled_dt),   
                        'runtime': dt_distilled_runtime
                    },
                    'inet_scores': {
                        'soft_binary_crossentropy': np.nan_to_num(soft_binary_crossentropy_inet_dt),
                        'binary_crossentropy': np.nan_to_num(binary_crossentropy_inet_dt),
                        'accuracy': np.nan_to_num(accuracy_inet_dt),
                        'f1_score': np.nan_to_num(f1_score_inet_dt),
                        'roc_auc_score': np.nan_to_num(roc_auc_score_inet_dt),            
                        #'runtime': inet_runtime
                    },                
               }



    return results, dt_distilled
    


def evaluate_interpretation_net_synthetic_data(network_parameters_array,
                                               X_test_lambda_array, 
                                               model,
                                               config, 
                                               identifier, 
                                               mean_train_parameters, 
                                               std_train_parameters,
                                               network_parameters_train_array,
                                               distances_dict={},
                                               verbosity=0):
    
    from utilities.InterpretationNet import autoencode_data, load_encoder_model, restructure_data_cnn_lstm
            
    def print_results_synthetic_evaluation_single(inet_evaluation_result_dict_mean):
        tab = PrettyTable()
        tab.field_names = ['Metric', 'Distilled DT (Train/Random Data)', 'Distilled DT (Test Data)', 'I-Net DT (Test Data)']
        
        max_width = {}   
        for field in tab.field_names:
            if field == 'Metric':
                max_width[field] = 25
            else:
                max_width[field] = 8
        tab._max_width = max_width
    
        tab.add_rows(
            [
                ['Soft Binary Crossentropy (Mean)', np.round(inet_evaluation_result_dict_mean['dt_scores']['soft_binary_crossentropy_data_random'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['soft_binary_crossentropy'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['soft_binary_crossentropy'], 3)],
                ['Binary Crossentropy (Mean)', np.round(inet_evaluation_result_dict_mean['dt_scores']['binary_crossentropy_data_random'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['binary_crossentropy'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['binary_crossentropy'], 3)],
                ['Accuracy (Mean)', np.round(inet_evaluation_result_dict_mean['dt_scores']['accuracy_data_random'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['accuracy'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['accuracy'], 3)],
                ['F1 Score (Mean)', np.round(inet_evaluation_result_dict_mean['dt_scores']['f1_score_data_random'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['f1_score'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['f1_score'], 3)],
                ['ROC AUC Score (Mean)', np.round(inet_evaluation_result_dict_mean['dt_scores']['roc_auc_score_data_random'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['roc_auc_score'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['roc_auc_score'], 3)],                
                ['Runtime (Mean)',  np.round(inet_evaluation_result_dict_mean['dt_scores']['runtime'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['runtime'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['runtime'], 3)],
                ['Soft Binary Crossentropy (Median)', np.round(inet_evaluation_result_dict_mean['dt_scores']['soft_binary_crossentropy_data_random_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['soft_binary_crossentropy_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['soft_binary_crossentropy_median'], 3)],
                ['Binary Crossentropy (Median)', np.round(inet_evaluation_result_dict_mean['dt_scores']['binary_crossentropy_data_random_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['binary_crossentropy_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['binary_crossentropy_median'], 3)],
                ['Accuracy (Median)', np.round(inet_evaluation_result_dict_mean['dt_scores']['accuracy_data_random_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['accuracy_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['accuracy_median'], 3)],
                ['F1 Score (Median)', np.round(inet_evaluation_result_dict_mean['dt_scores']['f1_score_data_random_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['f1_score_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['f1_score_median'], 3)],
                ['ROC AUC Score (Median)', np.round(inet_evaluation_result_dict_mean['dt_scores']['roc_auc_score_data_random_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['roc_auc_score_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['roc_auc_score_median'], 3)],                
                ['Runtime (Median)',  np.round(inet_evaluation_result_dict_mean['dt_scores']['runtime_median'], 3), np.round(inet_evaluation_result_dict_mean['dt_scores']['runtime_median'], 3), np.round(inet_evaluation_result_dict_mean['inet_scores']['runtime_median'], 3)],
            ]    
        )
        print(tab)       

    number = min(min(X_test_lambda_array.shape[0], max(config['i_net']['test_size'], 1)), 50)

    start_inet = time.time() 
    network_parameters = np.array(network_parameters_array[:number])
    if config['i_net']['data_reshape_version'] == 0 or config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2:
        network_parameters, network_parameters_flat = restructure_data_cnn_lstm(network_parameters, config, subsequences=None)
    elif config['i_net']['data_reshape_version'] == 3: #autoencoder
        encoder_model = load_encoder_model(config)
        network_parameters, network_parameters_flat, _ = autoencode_data([network_parameters], config, encoder_model)    
    dt_inet_list = model.predict(network_parameters)  

    end_inet = time.time()     
    inet_runtime = (end_inet - start_inet)    


    parallel_inet_evaluation = Parallel(n_jobs=config['computation']['n_jobs'], verbose=1, backend='loky') #loky #sequential multiprocessing
    inet_evaluation_results_with_dt = parallel_inet_evaluation(delayed(evaluate_interpretation_net_prediction_single_sample)(lambda_net_parameters, 
                                                                                                                   dt_inet,
                                                                                                                   X_test_lambda, 
                                                                                                                   #y_test_lambda,
                                                                                                                   config) for lambda_net_parameters, 
                                                                                                                               dt_inet, 
                                                                                                                               X_test_lambda in zip(network_parameters_array[:number], 
                                                                                                                                                    dt_inet_list, 
                                                                                                                                                    X_test_lambda_array[:number]))      

    del parallel_inet_evaluation

    inet_evaluation_results = [entry[0] for entry in inet_evaluation_results_with_dt]
    dt_distilled_list = [entry[1] for entry in inet_evaluation_results_with_dt]


    inet_evaluation_result_dict = None
    for some_dict in inet_evaluation_results:
        if inet_evaluation_result_dict == None:
            inet_evaluation_result_dict = some_dict
        else:
            inet_evaluation_result_dict = mergeDict(inet_evaluation_result_dict, some_dict)

    inet_evaluation_result_dict['inet_scores']['runtime'] = [inet_runtime/number for _ in range(number)]


    inet_evaluation_result_dict_mean = {}

    for key_l1, values_l1 in inet_evaluation_result_dict.items():
        if key_l1 != 'function_values':
            if isinstance(values_l1, dict):
                inet_evaluation_result_dict_mean[key_l1] = {}
                for key_l2, values_l2 in values_l1.items():
                    inet_evaluation_result_dict_mean[key_l1][key_l2] = np.mean(values_l2)
                    inet_evaluation_result_dict_mean[key_l1][key_l2 + '_median'] = np.median(values_l2)

    if verbosity > 0:
        print_results_synthetic_evaluation_single(inet_evaluation_result_dict_mean)    
    
    distances_dict_list = None

    for network in tqdm(network_parameters_array[:number]):
        distances_dict_single = calculate_network_distance(mean=mean_train_parameters, 
                                                                   std=std_train_parameters, 
                                                                   network_parameters=network, 
                                                                   lambda_net_parameters_train=network_parameters_train_array, 
                                                                   config=config)    

        if distances_dict_list == None:
            distances_dict_list = distances_dict_single
        else:
            distances_dict_list = mergeDict(distances_dict_list, distances_dict_single)     

    distances_dict[identifier] = {}
    for key, value in distances_dict_list.items():
        distances_dict[identifier][key] = np.mean(value)
    
    
    return inet_evaluation_result_dict, inet_evaluation_result_dict_mean, dt_distilled_list, distances_dict

def get_print_network_distances_dataframe(distances_dict): 
    data= np.array(
        [
            [np.round(value['z_score_aggregate'], 3) for value in distances_dict.values()],
            [np.round(value['distance_to_initialization_aggregate'], 3) for value in distances_dict.values()],
            [np.round(value['distance_to_sample_average'], 3) for value in distances_dict.values()],
            [np.round(value['distance_to_sample_min'], 3) for value in distances_dict.values()],
            [np.round(value['max_distance_to_neuron_average'], 3) for value in distances_dict.values()],
            [np.round(value['max_distance_to_neuron_min'], 3) for value in distances_dict.values()],           
        ]    
    ).T
    
    
    columns = ['Average Z-Score (Sample to Train Data)',
             'Average Distance to Initialization',
             'Average Mean Distance to Train Data',
             'Average Distance to closest Train Data Sample',
             'Average Biggest Distance for Single Neuron',
             'Minimum Biggest Distance for Single Neuron'
            ]
    
    index = list(distances_dict.keys())
    
    dataframe = pd.DataFrame(data=data, columns=columns, index=index)
    
    return dataframe

def print_network_distances(distances_dict):
    tab = PrettyTable()
    field_names = ['Measure']
    field_names.extend(list(distances_dict.keys()))
    tab.field_names = field_names
    
    max_width = {}   
    for field in field_names:
        if field == 'Measure':
            max_width[field] = 25
        else:
            max_width[field] = 8
    tab._max_width = max_width    
    tab.add_rows(
        [
            list(flatten_list(['Average Z-Score (Sample to Train Data)', [np.round(value['z_score_aggregate'], 3) for value in distances_dict.values()]])),
            list(flatten_list(['Average Distance to Initialization', [np.round(value['distance_to_initialization_aggregate'], 3) for value in distances_dict.values()]])),
            list(flatten_list(['Average Mean Distance to Train Data', [np.round(value['distance_to_sample_average'], 3) for value in distances_dict.values()]])),
            list(flatten_list(['Average Distance to closest Train Data Sample', [np.round(value['distance_to_sample_min'], 3) for value in distances_dict.values()]])),
            list(flatten_list(['Average Biggest Distance for Single Neuron', [np.round(value['max_distance_to_neuron_average'], 3) for value in distances_dict.values()]])),
            list(flatten_list(['Minimum Biggest Distance for Single Neuron', [np.round(value['max_distance_to_neuron_min'], 3) for value in distances_dict.values()]])),           
        ]    
    )
    print(tab)

    


def get_complete_distribution_evaluation_results_dataframe(inet_evaluation_result_dict_mean_by_distribution):

    identifier_list = list(inet_evaluation_result_dict_mean_by_distribution.keys())
    
    #columns=['Soft BC', 'BC', 'Acc', 'F1 Score', 'Runtime']
    columns=[
             'Acc Distilled Train Data', 
             'Acc Distilled Data Random', 
             'Acc Distilled', 
             'Acc I-Net', 
             'Soft BC Distilled Train Data',
             'Soft BC Distilled Data Random', 
             'Soft BC Distilled', 
             'Soft BC I-Net',     
             'BC Distilled Train Data', 
             'BC Distilled Data Random', 
             'BC Distilled', 
             'BC I-Net', 
             'F1 Score Distilled Train Data', 
             'F1 Score Distilled Data Random', 
             'F1 Score Distilled', 
             'F1 Score I-Net', 
             'ROC AUC Score Distilled Train Data', 
             'ROC AUC Score Distilled Data Random', 
             'ROC AUC Score Distilled', 
             'ROC AUC Score I-Net',         
             'Runtime Distilled Train Data', 
             'Runtime Distilled Data Random', 
             'Runtime Distilled', 
             'Runtime I-Net']
    #index = [] #'Metric'

    #for identifier in identifier_list:
        #index.append('Dist. (Random) ' + identifier)
        #index.append('Dist. ' + identifier)
        #index.append('I-Net ' + identifier)
        
    data = np.array([
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['accuracy_train_data'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['accuracy_data_random'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['accuracy'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['accuracy'], 3)
                        ] for identifier in identifier_list],           
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['soft_binary_crossentropy_train_data'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['soft_binary_crossentropy_data_random'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['soft_binary_crossentropy'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['soft_binary_crossentropy'], 3)
                        ] for identifier in identifier_list],
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['binary_crossentropy_train_data'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['binary_crossentropy_data_random'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['binary_crossentropy'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['binary_crossentropy'], 3)
                        ] for identifier in identifier_list],
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['f1_score_train_data'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['f1_score_data_random'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['f1_score'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['f1_score'], 3)
                        ] for identifier in identifier_list],  
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['roc_auc_score_train_data'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['roc_auc_score_data_random'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['roc_auc_score'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['roc_auc_score'], 3)
                        ] for identifier in identifier_list],            
                      [[
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['runtime'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['runtime'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['dt_scores']['runtime'], 3),
                          np.round(inet_evaluation_result_dict_mean_by_distribution[identifier]['inet_scores']['runtime'], 3)
                        ] for identifier in identifier_list]       
                    ])
    

    data = np.hstack(data)
    #dataframe = pd.DataFrame(data=data, columns=columns, index=index)
    dataframe = pd.DataFrame(data=data, columns=columns, index=identifier_list)
    
    return dataframe
    

      
def get_complete_performance_evaluation_results_dataframe(results_dict, identifier_list, dataset_size_list, dataset_size=10000):

    
    
    columns=[
             'Acc Distilled (Train Data)', 
             'Acc Distilled', 
             'Acc Distilled STD', 
             'Acc I-Net',   
             'Soft BC Distilled (Train Data)', 
             'Soft BC Distilled', 
             'Soft BC Distilled STD', 
             'Soft BC I-Net', 
             'BC Distilled (Train Data)', 
             'BC Distilled', 
             'BC Distilled STD', 
             'BC I-Net', 
             'F1 Score Distilled (Train Data)', 
             'F1 Score Distilled', 
             'F1 Score Distilled STD', 
             'F1 Score I-Net', 
             'ROC AUC Score Distilled (Train Data)', 
             'ROC AUC Score Distilled', 
             'ROC AUC Score Distilled STD', 
             'ROC AUC Score I-Net',         
             'Runtime Distilled (Train Data)', 
             'Runtime Distilled', 
             'Runtime Distilled STD', 
             'Runtime I-Net']    
    #index = [] #'Metric'
    
    indices = [i for i, x in enumerate(dataset_size_list) if x == dataset_size]
    
    if len(indices) > 1:
        data = np.array([
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['accuracy'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['accuracy'] for index in indices]), 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['accuracy_std'] for index in indices]), 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['accuracy'], 3)
                            ] for identifier in identifier_list],          
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['soft_binary_crossentropy'] for index in indices]), 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['soft_binary_crossentropy_std'] for index in indices]), 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['soft_binary_crossentropy'], 3)
                            ] for identifier in identifier_list],
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['binary_crossentropy'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['binary_crossentropy'] for index in indices]), 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['binary_crossentropy_std'] for index in indices]), 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['binary_crossentropy'], 3)
                            ] for identifier in identifier_list], 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['f1_score'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['f1_score'] for index in indices]), 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['f1_score_std'] for index in indices]), 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['f1_score'], 3)
                            ] for identifier in identifier_list],     
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['roc_auc_score'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['roc_auc_score'] for index in indices]), 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['roc_auc_score_std'] for index in indices]), 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['roc_auc_score'], 3)
                            ] for identifier in identifier_list],               
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['runtime'], 3),
                              np.round(np.mean([results_dict[identifier][index]['dt_scores']['runtime'] for index in indices]), 3),
                              np.nan,
                              np.round(np.mean([results_dict[identifier][index]['inet_scores']['runtime'] for index in indices]), 3),                              
                            ] for identifier in identifier_list]       
                        ])       
    
    else:
        data = np.array([
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['accuracy'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_std'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['accuracy'], 3)
                            ] for identifier in identifier_list],          
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['soft_binary_crossentropy'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_std'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['soft_binary_crossentropy'], 3)
                            ] for identifier in identifier_list],
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['binary_crossentropy'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_std'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['binary_crossentropy'], 3)
                            ] for identifier in identifier_list], 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['f1_score'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_std'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['f1_score'], 3)
                            ] for identifier in identifier_list],   
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['roc_auc_score'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_std'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['roc_auc_score'], 3)
                            ] for identifier in identifier_list],                
            
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['runtime'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime'], 3),
                              np.nan,
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['runtime'], 3)
                            ] for identifier in identifier_list]       
                        ])     
        
    data = np.hstack(data)

    #dataframe = pd.DataFrame(data=data, columns=columns, index=index)
    dataframe = pd.DataFrame(data=data, columns=columns, index=identifier_list)
    
    return dataframe



def get_complete_performance_evaluation_results_dataframe_all_distrib(results_dict, 
                                                                      identifier_list, 
                                                                      dataset_size_list, 
                                                                      distribution_list_evaluation, 
                                                                      dataset_size=10000):

    
    columns=flatten_list([
                         'Acc Distilled (Train Data)', 
                         'Acc Distilled (Standard Uniform)',
                         'Acc Distilled (Standard Normal)',
                         ['Acc Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD Acc Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'Acc I-Net',   
                         'Soft BC Distilled (Train Data)', 
                         'Soft BC Distilled (Standard Uniform)',
                         'Soft BC Distilled (Standard Normal)', 
                         ['Soft BC Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD Soft BC Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'Soft BC I-Net', 
                         'BC Distilled (Train Data)', 
                         'BC Distilled (Standard Uniform)',
                         'BC Distilled (Standard Normal)', 
                         ['BC Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD BC Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'BC I-Net', 
                         'F1 Score Distilled (Train Data)', 
                         'F1 Score Distilled (Standard Uniform)',
                         'F1 Score Distilled (Standard Normal)', 
                         ['F1 Score Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD F1 Score Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'F1 Score I-Net', 
                         'ROC AUC Score Distilled (Train Data)', 
                         'ROC AUC Score Distilled (Standard Uniform)',
                         'ROC AUC Score Distilled (Standard Normal)', 
                         ['ROC AUC Score Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD ROC AUC Score Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'ROC AUC Score I-Net',         
        
                         'Runtime Distilled (Train Data)', 
                         'Runtime Distilled (Standard Uniform)',
                         'Runtime Distilled (Standard Normal)', 
                         ['Runtime Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         ['STD Runtime Distilled (' + str(distribution) + ')' for distribution in distribution_list_evaluation], 
                         'Runtime I-Net'])  
    #index = [] #'Metric'
    
    indices = [i for i, x in enumerate(dataset_size_list) if x == dataset_size]
    
    if len(indices) > 1:
        data = np.array([
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['accuracy'], 3),  
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['accuracy'], 3),  
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['accuracy'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['accuracy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['accuracy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['accuracy'], 3)
                            ]) for identifier in identifier_list],          
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['soft_binary_crossentropy'], 3),                              
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['soft_binary_crossentropy'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['soft_binary_crossentropy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['soft_binary_crossentropy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['soft_binary_crossentropy'], 3)
                            ]) for identifier in identifier_list],
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['binary_crossentropy'], 3),                              
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['binary_crossentropy'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['binary_crossentropy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['binary_crossentropy_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['binary_crossentropy'], 3)
                            ]) for identifier in identifier_list], 
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['f1_score'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['f1_score_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['f1_score_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['f1_score'], 3)
                            ]) for identifier in identifier_list],        
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['roc_auc_score'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['roc_auc_score_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['roc_auc_score_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['roc_auc_score'], 3)
                            ]) for identifier in identifier_list],             
            
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['runtime'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['runtime'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['runtime'], 3),                              
                              [np.round(np.mean([results_dict[identifier][index]['dt_scores']['runtime_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              [np.round(np.std([results_dict[identifier][index]['dt_scores']['runtime_' + str(distribution)] for index in indices]), 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['runtime'], 3)
                            ]) for identifier in identifier_list]       
                        ])      
    
    else:
        data = np.array([
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['accuracy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['accuracy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['accuracy'], 3),                              
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_data_random']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['accuracy'], 3)
                            ]) for identifier in identifier_list],          
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['soft_binary_crossentropy'], 3),                                                            
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_data_random']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['soft_binary_crossentropy'], 3)
                            ]) for identifier in identifier_list],
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['binary_crossentropy'], 3),                              
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_data_random']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['binary_crossentropy'], 3)
                            ]) for identifier in identifier_list], 
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['f1_score'], 3),               
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_data_random']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['f1_score'], 3)
                            ]) for identifier in identifier_list],      
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['roc_auc_score'], 3),               
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_data_random']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['roc_auc_score'], 3)
                            ]) for identifier in identifier_list],             
            
                          [flatten_list([
                              np.round(results_dict[identifier][dataset_size_list.index('TRAINDATA')]['dt_scores']['runtime'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDUNIFORM')]['dt_scores']['runtime'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index('STANDARDNORMAL')]['dt_scores']['runtime'], 3),                              
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime']
                              [np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime_' + str(distribution)], 3) for distribution in distribution_list_evaluation],
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['runtime'], 3)
                            ]) for identifier in identifier_list]       
                        ])     

    data = np.hstack(data)

    #dataframe = pd.DataFrame(data=data, columns=columns, index=index)
    dataframe = pd.DataFrame(data=data, columns=columns, index=identifier_list)
    
    return dataframe


    
    
    
    
def print_complete_performance_evaluation_results(results_dict, identifier_list, dataset_size_list, dataset_size=10000):
    print('Dataset Size:\t\t', dataset_size)
    tab = PrettyTable()
    field_names = ['Metric']

    for identifier in identifier_list:
        #field_names.append('Dist. (Random) ' + identifier)
        field_names.append('Dist. ' + identifier)
        field_names.append('I-Net ' + identifier)

    tab.field_names = field_names
    
    max_width = {}   
    for field in field_names:
        if field == 'Metric':
            max_width[field] = 25
        else:
            max_width[field] = 6
    tab._max_width = max_width    
    tab.add_rows(
        [
            flatten_list(['Soft BC', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['soft_binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['soft_binary_crossentropy'], 3)
                            ] for identifier in identifier_list]          
                          ]),
            flatten_list(['BC', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['binary_crossentropy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['binary_crossentropy'], 3)
                            ] for identifier in identifier_list]                               
                          ]),
            flatten_list(['Acc', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['accuracy'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['accuracy'], 3)
                            ] for identifier in identifier_list]                               
                          ]),
            flatten_list(['F1 Score', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['f1_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['f1_score'], 3)
                            ] for identifier in identifier_list]                               
                          ]),
            flatten_list(['ROC AUC Score', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score_data_random']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['roc_auc_score'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['roc_auc_score'], 3)
                            ] for identifier in identifier_list]                               
                          ]),            
            
            flatten_list(['Runtime', 
                          [[
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['runtime_train_data'], 3),
                              #np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime']
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['dt_scores']['runtime'], 3),
                              np.round(results_dict[identifier][dataset_size_list.index(dataset_size)]['inet_scores']['runtime'], 3)
                            ] for identifier in identifier_list]                               
                          ]),
        ]    
    )
    print(tab)
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------')             


def print_results_synthetic_evaluation(inet_evaluation_result_dict_mean_train, inet_evaluation_result_dict_mean_valid, inet_evaluation_result_dict_mean_test, distances_dict):
    
    tab = PrettyTable()
    tab.field_names = ['Metric', 'Train', 'Train ', ' Train ', 'Valid', 'Valid ', ' Valid ', 'Test', 'Test ', ' Test ']
    
    max_width = {}   
    for field in tab.field_names:
        if field == 'Metric':
            max_width[field] = 25
        else:
            max_width[field] = 8
    tab._max_width = max_width
    
    tab.add_rows(
        [
            ['Metric', 
             'Dist. (Random)', 'Dist.', 'I-Net', 
             'Dist. (Random)', 'Dist.', 'I-Net', 
             'Dist. (Random)', 'Dist.', 'I-Net'],
            ['Soft Binary Crossentropy (Mean)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['soft_binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['soft_binary_crossentropy'], 3),
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['soft_binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['soft_binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['soft_binary_crossentropy'], 3),
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['soft_binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['soft_binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['soft_binary_crossentropy'], 3),
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['soft_binary_crossentropy'], 3)],
            ['Binary Crossentropy (Mean)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['binary_crossentropy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['binary_crossentropy'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['binary_crossentropy'], 3)],
            ['Accuracy (Mean)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['accuracy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['accuracy'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['accuracy'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['accuracy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['accuracy'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['accuracy'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['accuracy_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['accuracy'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['accuracy'], 3)],
            ['F1 Score (Mean)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['f1_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['f1_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['f1_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['f1_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['f1_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['f1_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['f1_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['f1_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['f1_score'], 3)],
            ['ROC AUC Score (Mean)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['roc_auc_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['roc_auc_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['roc_auc_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['roc_auc_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['roc_auc_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['roc_auc_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['roc_auc_score_data_random'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['roc_auc_score'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['roc_auc_score'], 3)],            
            ['Runtime (Mean)',  
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['runtime'], 3),  
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['runtime'], 3),  
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['runtime'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['runtime'], 3)],
            ['Soft Binary Crossentropy (Median)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['soft_binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['soft_binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['soft_binary_crossentropy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['soft_binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['soft_binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['soft_binary_crossentropy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['soft_binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['soft_binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['soft_binary_crossentropy_median'], 3)],
            ['Binary Crossentropy (Median)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['binary_crossentropy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['binary_crossentropy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['binary_crossentropy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['binary_crossentropy_median'], 3),
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['binary_crossentropy_median'], 3)],
            ['Accuracy (Median)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['accuracy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['accuracy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['accuracy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['accuracy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['accuracy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['accuracy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['accuracy_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['accuracy_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['accuracy_median'], 3)],
            ['F1 Score (Median)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['f1_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['f1_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['f1_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['f1_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['f1_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['f1_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['f1_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['f1_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['f1_score_median'], 3)],
            ['ROC AUC Score (Median)', 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['roc_auc_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['roc_auc_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['roc_auc_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['roc_auc_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['roc_auc_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['roc_auc_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['roc_auc_score_data_random_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['roc_auc_score_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['roc_auc_score_median'], 3)],            
            
            ['Runtime (Median)',  
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_train['inet_scores']['runtime_median'], 3),  
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_valid['inet_scores']['runtime_median'], 3),  
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['dt_scores']['runtime_median'], 3), 
             np.round(inet_evaluation_result_dict_mean_test['inet_scores']['runtime_median'], 3)],
        ]    
    )
    print(tab)
    
    print_network_distances(distances_dict)



def adjust_data_to_number_of_variables(X_data, y_data, number_of_variables, seed=42):

    if X_data.shape[1] > number_of_variables:
        #X_data = X_data.sample(n=number_of_variables,axis='columns')

        clf_extra = ExtraTreesClassifier(n_estimators=100,
                                          random_state=seed)
        clf_extra = clf_extra.fit(X_data, y_data)

        selector = SelectFromModel(clf_extra, 
                                         prefit=True,
                                         threshold=-np.inf,
                                         max_features=number_of_variables)
        feature_idx = selector.get_support()   
        X_data = X_data.loc[:,feature_idx]
    else:
        for i in range(number_of_variables-X_data.shape[1]):
            column_name = 'zero_dummy_' + str(i+1)
            X_data[column_name] = np.zeros(X_data.shape[0])
    
    return X_data

def normalize_real_world_data(X_data):
    normalizer_list = []
    if isinstance(X_data, pd.DataFrame):
        for column_name in X_data:
            scaler = MinMaxScaler()
            scaler.fit(X_data[column_name].values.reshape(-1, 1))
            X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)
    else:
        for i, column in enumerate(X_data.T):
            scaler = MinMaxScaler()
            scaler.fit(column.reshape(-1, 1))
            X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)
        
    return X_data, normalizer_list

def split_train_test_valid(X_data, y_data, valid_frac=0.05, test_frac=0.10, seed=42, verbose=0):
    data_size = X_data.shape[0]
    test_size = int(data_size*test_frac)
    valid_size = int(data_size*valid_frac)
    
    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_with_valid, y_train_with_valid, test_size=valid_size, random_state=seed)

    if verbose > 0:
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def rebalance_data(X_train, y_train, balance_ratio=0.25, strategy=None, seed=42):#, strategy='SMOTE'
    true_labels = len(y_train[y_train >= 0.5 ]) 
    false_labels = len(y_train[y_train < 0.5 ]) 

    true_ratio = true_labels/(true_labels+false_labels)
    false_ratio = false_labels/(false_labels+true_labels)
    
    min_ratio = min(true_ratio, false_ratio)
    print('True Ratio: ', str(true_labels/(true_labels+false_labels)))    
    if min_ratio <= balance_ratio:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
        from imblearn.combine import SMOTETomek, SMOTEENN
        if strategy == 'SMOTE':
            oversample = SMOTE()
        elif strategy == 'SMOTEN':
            oversample = SMOTEN()                 
        elif strategy == 'BorderlineSMOTE':
            oversample = BorderlineSMOTE()                
        elif strategy == 'KMeansSMOTE':
            oversample = KMeansSMOTE(cluster_balance_threshold=0.1)    
        elif strategy == 'SVMSMOTE':
            oversample = SVMSMOTE()   
        elif strategy == 'SMOTETomek':
            oversample = SMOTETomek()   
        elif strategy == 'SMOTEENN':
            oversample = SMOTEENN()               
        elif strategy == 'ADASYN':
            oversample = ADASYN()
        else:
            oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)

        X_train, y_train = oversample.fit_resample(X_train, y_train)

        true_labels = len(y_train[y_train >= 0.5 ]) 
        false_labels = len(y_train[y_train < 0.5 ]) 

        print('True Ratio: ', str(true_labels/(true_labels+false_labels)))    

    return X_train, y_train

def train_network_real_world_data(X_train, y_train, X_valid, y_valid, config, verbose=1):
    
    from utilities.LambdaNet import generate_lambda_net_from_config

    random.seed(config['computation']['RANDOM_SEED'])
    np.random.seed(config['computation']['RANDOM_SEED'])
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(config['computation']['RANDOM_SEED'])
    else:
        tf.set_random_seed(config['computation']['RANDOM_SEED'])
    test_network = generate_lambda_net_from_config(config, seed=config['computation']['RANDOM_SEED'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      patience=config['lambda_net']['patience_lambda'],#10, 
                                                      min_delta=0.001, 
                                                      verbose=0, 
                                                      mode='min', 
                                                      restore_best_weights=config['lambda_net']['restore_best_weights'],#True
                                                     )
        
    model_history = test_network.fit(X_train,
                                      y_train,
                                      epochs=config['lambda_net']['epochs_lambda'], 
                                      batch_size=config['lambda_net']['batch_lambda'], 
                                      callbacks=[early_stopping], #PlotLossesKerasTF()
                                      validation_data=(X_valid, y_valid),
                                      verbose=0)
 
    
    if verbose > 0:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axes[0].plot(model_history.history['loss'])
        axes[0].plot(model_history.history['val_loss'])      
        axes[0].set_title('model loss')
        axes[0].set_ylabel('loss')
        axes[0].set_xlabel('epoch')
        axes[0].legend(['train loss', 'valid loss'], loc='upper left')  

        axes[1].plot(model_history.history['binary_accuracy'])
        axes[1].plot(model_history.history['val_binary_accuracy'])  
        axes[1].set_title('model accuracy')
        axes[1].set_ylabel('accuracy')
        axes[1].set_xlabel('epoch')
        axes[1].legend(['train acc', 'valid acc'], loc='upper left')  
        plt.show()                    


    return test_network, model_history

                    
    
def make_inet_prediction(model, test_network, config):
    from utilities.InterpretationNet import autoencode_data, load_encoder_model, restructure_data_cnn_lstm
    
    test_network_parameters = shaped_network_parameters_to_array(test_network.get_weights(), config)

    start_inet = time.time() 
    
    network_parameters = np.array([test_network_parameters])
    if config['i_net']['data_reshape_version'] == 0 or config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2:
        network_parameters, network_parameters_flat = restructure_data_cnn_lstm(network_parameters, config, subsequences=None)
    elif config['i_net']['data_reshape_version'] == 3: #autoencoder
        encoder_model = load_encoder_model(config)  
        network_parameters, network_parameters_flat, _ = autoencode_data([network_parameters], config, encoder_model)    
    dt_inet = model.predict(network_parameters)[0]    

    end_inet = time.time()     
    inet_runtime = (end_inet - start_inet)   
    
    test_network_parameters = shaped_network_parameters_to_array(test_network.get_weights(), config)
    return dt_inet, test_network_parameters, inet_runtime

                
    
def print_results_different_data_sizes(results, dataset_size_list_print):
    tab = PrettyTable()
    tab.field_names = flatten_list(['Metric', [['Dist. (Random) ' + str(size), 'Dist. ' + str(size)] for size in dataset_size_list_print], 'I-Net'])
    
    max_width = {}   
    for field in tab.field_names:
        if field == 'Metric':
            max_width[field] = 25
        else:
            max_width[field] = 8
    tab._max_width = max_width
    
    tab.add_rows(
        [
            flatten_list(['Soft Binary Crossentropy', 
                          [[np.round(result_dict['dt_scores']['soft_binary_crossentropy_data_random'], 3), np.round(result_dict['dt_scores']['soft_binary_crossentropy'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['soft_binary_crossentropy'], 3)]),
            flatten_list(['Binary Crossentropy',  
                          [[np.round(result_dict['dt_scores']['binary_crossentropy_data_random'], 3), np.round(result_dict['dt_scores']['binary_crossentropy'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['binary_crossentropy'], 3)]),
            flatten_list(['Accuracy', 
                          [[np.round(result_dict['dt_scores']['accuracy_data_random'], 3), np.round(result_dict['dt_scores']['accuracy'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['accuracy'], 3)]),
            flatten_list(['F1 Score', 
                          [[np.round(result_dict['dt_scores']['f1_score_data_random'], 3), np.round(result_dict['dt_scores']['f1_score'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['f1_score'], 3)]),
            flatten_list(['ROC AUC Score', 
                          [[np.round(result_dict['dt_scores']['roc_auc_score_data_random'], 3), np.round(result_dict['dt_scores']['roc_auc_score'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['roc_auc_score'], 3)]),
            flatten_list(['Runtime',  
                          [[np.round(result_dict['dt_scores']['runtime'], 3), np.round(result_dict['dt_scores']['runtime'], 3)] for result_dict in results],
                          np.round(results[0]['inet_scores']['runtime'], 3)])
        ]    
    )
    print(tab)

    
def plot_decision_tree_from_parameters(dt_parameters, normalizer_list, config):
    if config['function_family']['dt_type'] == 'vanilla':
        image, nodes = anytree_decision_tree_from_parameters(dt_parameters, config=config, normalizer_list=normalizer_list)
    else:
        tree = generate_random_decision_tree(config)
        tree.initialize_from_parameter_array(dt_parameters, reshape=True, config=config)
        image = tree.plot_tree()
    
    return image

def plot_decision_tree_from_model(dt_model, config):
    if config['function_family']['dt_type'] == 'vanilla':
        plt.figure(figsize=(24,12))  # set plot size (denoted in inches)
        plot_tree(dt_model, fontsize=12)
        image = plt.show()
    else:
        image = dt_model.plot_tree()
    
    return image

def evaluate_network_real_world_data_parallel(loss_function, metrics, test_network_parameter_array, X_train, X_test, dataset_size_list, config, verbosity=0, distribution=None):
        
    from utilities.InterpretationNet import load_inet
                
    model = load_inet(loss_function, metrics, config)
    
    test_network = network_parameters_to_network(test_network_parameter_array, config)
        
    dt_inet, test_network_parameters, inet_runtime = make_inet_prediction(model, test_network, config)

    results_list = []
    dt_distilled_list = []
        
    counter_standardnormal = 0
    counter_standarduniform = 0
    counter_random = 0
    
    for i, dataset_size in enumerate(dataset_size_list):

        if dataset_size == 'TRAINDATA': 
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
            
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED']            
            
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config,
                                                                               distribution=distribution,
                                                                               train_data=X_train,
                                                                               verbosity=verbosity)

        elif dataset_size == 'STANDARDUNIFORM': 
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = config['evaluation']['random_evaluation_dataset_size_per_distribution']
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_standarduniform
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution='standarduniform',
                                                                               verbosity=verbosity)     
            counter_standarduniform += 1        
            
        elif dataset_size == 'STANDARDNORMAL': 
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = config['evaluation']['random_evaluation_dataset_size_per_distribution']
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_standardnormal            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution='standardnormal',
                                                                               verbosity=verbosity)   
            counter_standardnormal += 1        
                             
        else:
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = dataset_size
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_random
            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution=distribution,
                                                                               verbosity=verbosity)
            counter_random += 1        

        results['inet_scores']['runtime'] = inet_runtime
        results_list.append(results)
        dt_distilled_list.append(dt_distilled)

        if verbosity > 1:
            print('Dataset Size:\t\t', dataset_size)
            tab = PrettyTable()
            tab.field_names = ['Metric', 'Distilled DT (Train/Random Data)', 'Distilled DT (Test Data)', 'I-Net DT (Test Data)']
            
            max_width = {}   
            for field in tab.field_names:
                if field == 'Metric':
                    max_width[field] = 25
                else:
                    max_width[field] = 10
            tab._max_width = max_width
    
            tab.add_rows(
                [
                    ['Soft Binary Crossentropy', np.round(results['dt_scores']['soft_binary_crossentropy_data_random'], 3), np.round(results['dt_scores']['soft_binary_crossentropy'], 3), np.round(results['inet_scores']['soft_binary_crossentropy'], 3)],
                    ['Binary Crossentropy',  np.round(results['dt_scores']['binary_crossentropy_data_random'], 3), np.round(results['dt_scores']['binary_crossentropy'], 3), np.round(results['inet_scores']['binary_crossentropy'], 3)],
                    ['Accuracy', np.round(results['dt_scores']['accuracy_data_random'], 3), np.round(results['dt_scores']['accuracy'], 3), np.round(results['inet_scores']['accuracy'], 3)],
                    ['F1 Score', np.round(results['dt_scores']['f1_score_data_random'], 3), np.round(results['dt_scores']['f1_score'], 3), np.round(results['inet_scores']['f1_score'], 3)],
                    ['ROC AUC Score', np.round(results['dt_scores']['roc_auc_score_data_random'], 3), np.round(results['dt_scores']['roc_auc_score'], 3), np.round(results['inet_scores']['roc_auc_score'], 3)],                    
                    ['Runtime',  np.round(results['dt_scores']['runtime'], 3), np.round(results['dt_scores']['runtime'], 3), np.round(results['inet_scores']['runtime'], 3)],
                ]    
            )
            print(tab)
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------')             

    evaluation_result_dict = None
    for some_dict in results_list:
        if evaluation_result_dict == None:
            evaluation_result_dict = some_dict
        else:
            evaluation_result_dict = mergeDict(evaluation_result_dict, some_dict)
            
    return evaluation_result_dict, results_list, test_network_parameters, dt_inet, dt_distilled_list



    


def evaluate_network_real_world_data(model, test_network, X_train, X_test, dataset_size_list, config, verbosity=0, distribution=None):
        
    dt_inet, test_network_parameters, inet_runtime = make_inet_prediction(model, test_network, config)

    results_list = []
    dt_distilled_list = []
    
    counter_standardnormal = 0
    counter_standarduniform = 0
    counter_random = 0    
    
    for i, dataset_size in enumerate(dataset_size_list):

        if dataset_size == 'TRAINDATA': 
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values   
                
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED']            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config,
                                                                               distribution=distribution,
                                                                               train_data=X_train,
                                                                               verbosity=verbosity)
        elif dataset_size == 'STANDARDUNIFORM': 
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = config['evaluation']['random_evaluation_dataset_size_per_distribution']
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_standarduniform         
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution='standarduniform',
                                                                               verbosity=verbosity)
            
            counter_standarduniform += 1
        
        elif dataset_size == 'STANDARDNORMAL': 
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = config['evaluation']['random_evaluation_dataset_size_per_distribution']
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_standardnormal            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution='standardnormal',
                                                                               verbosity=verbosity)
            
            counter_standardnormal += 1
        
        else:
            
            config_test = deepcopy(config)
            config_test['evaluation']['per_network_optimization_dataset_size'] = dataset_size
            config_test['computation']['RANDOM_SEED'] = config['computation']['RANDOM_SEED'] + counter_random
            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values            
                
            results, dt_distilled = evaluate_interpretation_net_prediction_single_sample(test_network_parameters, 
                                                                               dt_inet,
                                                                               X_test, 
                                                                               #y_test_lambda,
                                                                               config_test,
                                                                               distribution=distribution,
                                                                               verbosity=verbosity)
            
            counter_random += 1


        results['inet_scores']['runtime'] = inet_runtime
        results_list.append(results)
        dt_distilled_list.append(dt_distilled)

        if verbosity > 1:
            print('Dataset Size:\t\t', dataset_size)
            tab = PrettyTable()
            tab.field_names = ['Metric', 'Distilled DT (Train/Random Data)', 'Distilled DT (Test Data)', 'I-Net DT (Test Data)']
            
            max_width = {}   
            for field in tab.field_names:
                if field == 'Metric':
                    max_width[field] = 25
                else:
                    max_width[field] = 10
            tab._max_width = max_width
    
            tab.add_rows(
                [
                    ['Soft Binary Crossentropy', np.round(results['dt_scores']['soft_binary_crossentropy_data_random'], 3), np.round(results['dt_scores']['soft_binary_crossentropy'], 3), np.round(results['inet_scores']['soft_binary_crossentropy'], 3)],
                    ['Binary Crossentropy',  np.round(results['dt_scores']['binary_crossentropy_data_random'], 3), np.round(results['dt_scores']['binary_crossentropy'], 3), np.round(results['inet_scores']['binary_crossentropy'], 3)],
                    ['Accuracy', np.round(results['dt_scores']['accuracy_data_random'], 3), np.round(results['dt_scores']['accuracy'], 3), np.round(results['inet_scores']['accuracy'], 3)],
                    ['F1 Score', np.round(results['dt_scores']['f1_score_data_random'], 3), np.round(results['dt_scores']['f1_score'], 3), np.round(results['inet_scores']['f1_score'], 3)],
                    ['ROC AUC Score', np.round(results['dt_scores']['roc_auc_score_data_random'], 3), np.round(results['dt_scores']['roc_auc_score'], 3), np.round(results['inet_scores']['roc_auc_score'], 3)],                    
                    ['Runtime',  np.round(results['dt_scores']['runtime'], 3), np.round(results['dt_scores']['runtime'], 3), np.round(results['inet_scores']['runtime'], 3)],
                ]    
            )
            print(tab)
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------')             

    evaluation_result_dict = None
    for some_dict in results_list:
        if evaluation_result_dict == None:
            evaluation_result_dict = some_dict
        else:
            evaluation_result_dict = mergeDict(evaluation_result_dict, some_dict)
            
    return evaluation_result_dict, results_list, test_network_parameters, dt_inet, dt_distilled_list



######################################################################################################################################################################################################################
###########################################################################################  LAMBDA NET UTILITY ################################################################################################ 
######################################################################################################################################################################################################################



#################################################################################################################################################################################### Normalization #################################################################################### ################################################################################################################################################################################################################

def get_order_sum(arrays):
    arrays = np.array(arrays)
    values = [np.sum(arrays[0])]
    order = [0]
    for i in range(1, len(arrays)):
        value = np.sum(arrays[i])
        pos = 0
        while pos<len(values) and value>=values[pos]:
            if value == values[pos]:
                print("!!!!!!!!!!!!!!!!KOLLISION!!!!!!!!!!!!!!!!!!")
                print(value)
                print(arrays[i])
                print(arrays[order[pos]])
            pos += 1
        values.insert(pos, value)
        order.insert(pos, i)
    return order

## source for sort_array: https://www.geeksforgeeks.org/permute-the-elements-of-an-array-following-given-order/

def sort_array(arr, order):
    length = len(order)
    #ordered_arr = np.zeros(length)
    ordered_arr = [None] * length
    for i in range(length):
        ordered_arr[i] = arr[order[i]]
    arr=ordered_arr
    return arr    

def normal_neural_net(model_arr, config):
    for i in range(len(config['lambda_net']['lambda_network_layers'])):
        index = 2*(i)
        dense_arr = np.transpose(model_arr[index])
        order = get_order_sum(dense_arr)
        for j in range(len(model_arr[index])):
            model_arr[index][j] = sort_array(model_arr[index][j], order)
        model_arr[index+1] = np.array(sort_array(model_arr[index+1], order))
        model_arr[index+2] = np.array(sort_array(model_arr[index+2], order))
    return model_arr


#################################################################################################################################################################################### Normalization #################################################################################### 

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
                                                                                                 
def generate_base_model(config, disable_batchnorm = False): #without dropout
    
    output_neurons = 1 if config['data']['num_classes']==2 else config['data']['num_classes']
    output_activation = 'sigmoid' if config['data']['num_classes']==2 else 'softmax'
    
    model = Sequential()
        
    tf.random.set_seed(config['computation']['RANDOM_SEED'])
        
    #kerase defaults: kernel_initializer='glorot_uniform', bias_initializer='zeros'               
    model.add(Dense(config['lambda_net']['lambda_network_layers'][0], activation='relu', input_dim=config['data']['number_of_variables'], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), bias_initializer='zeros'))
    if config['lambda_net']['use_batchnorm_lambda'] and not disable_batchnorm:
        model.add(BatchNormalization())
   
    if config['lambda_net']['dropout_lambda'] > 0:
        model.add(Dropout(config['lambda_net']['dropout_lambda']))

    for neurons in config['lambda_net']['lambda_network_layers'][1:]:
        model.add(Dense(neurons, 
                        activation='relu', 
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                        bias_initializer='zeros'))
        if config['lambda_net']['use_batchnorm_lambda'] and not disable_batchnorm:
            model.add(BatchNormalization())
        
        if config['lambda_net']['dropout_lambda'] > 0:
            model.add(Dropout(config['lambda_net']['dropout_lambda']))   
    
    model.add(Dense(output_neurons, 
                    activation=output_activation, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config['computation']['RANDOM_SEED']), 
                    bias_initializer='zeros'))    
    
    return model




def shape_flat_network_parameters(flat_network_parameters, target_network_parameters):
    
    #from utilities.utility_functions import flatten_list
    
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
    
    #if config['lambda_net']['use_batchnorm_lambda']:
    #    list_indices = [i for i in range(len(shaped_network_parameters))]
    #    list_indices_without_batchnorm = []
    #    for i in range(len(shaped_network_parameters)//6):
    #        list_indices_without_batchnorm.append(i + 6*i)
    #        list_indices_without_batchnorm.append(i+1 + 6*i)
    #    
    #    shaped_network_parameters = [shaped_network_parameters[i] for i in list_indices_without_batchnorm]
    
    
    if config['lambda_net']['use_batchnorm_lambda']:    
        for layer_weights, biases, batchnorm_1, batchnorm_2, batchnorm_3, batchnorm_4 in chunks(shaped_network_parameters[:-2], 6):#pairwise(shaped_network_parameters)
            for neuron in layer_weights:
                for weight in neuron:
                    network_parameter_list.append(weight)
            for bias in biases:
                network_parameter_list.append(bias)
            for batchnorm in batchnorm_1:
                network_parameter_list.append(batchnorm)   
            for batchnorm in batchnorm_2:
                network_parameter_list.append(batchnorm)                      
            for batchnorm in batchnorm_3:
                network_parameter_list.append(batchnorm)      
            for batchnorm in batchnorm_4:
                network_parameter_list.append(batchnorm)     
        for layer_weights, biases in pairwise(shaped_network_parameters[-2:]):    #clf.get_weights()
            for neuron in layer_weights:
                for weight in neuron:
                    network_parameter_list.append(weight)
            for bias in biases:
                network_parameter_list.append(bias)                
    else:
        for layer_weights, biases in pairwise(shaped_network_parameters):    #clf.get_weights()
            for neuron in layer_weights:
                for weight in neuron:
                    network_parameter_list.append(weight)
            for bias in biases:
                network_parameter_list.append(bias)
                
    return np.array(network_parameter_list)



def calculate_network_distance(mean, 
                               std, 
                               network_parameters, 
                               lambda_net_parameters_train, 
                               config):
    
    z_score = (network_parameters-mean)/std
    z_score = z_score[~np.isnan(z_score)]
    z_score = z_score[~np.isinf(z_score)]
    z_score_aggregate = np.sum(np.abs(z_score))
    
    initialization_array = shaped_network_parameters_to_array(generate_base_model(config).get_weights(), config)

    distance_to_initialization = network_parameters - initialization_array
    distance_to_initialization_aggregate = np.sum(np.abs(distance_to_initialization))

    distance_to_sample_aggregate_list = []
    distance_to_sample_max_list = []
    for sample in lambda_net_parameters_train:
        distance_to_sample = network_parameters - sample
        
        distance_to_sample_max = np.max(np.abs(distance_to_sample))
        distance_to_sample_aggregate = np.sum(np.abs(distance_to_sample))
        
        distance_to_sample_max_list.append(distance_to_sample_max)
        distance_to_sample_aggregate_list.append(distance_to_sample_aggregate)
        
    distance_to_sample_average = np.mean(distance_to_sample_aggregate_list)
    distance_to_sample_min = np.min(distance_to_sample_aggregate_list)    
    
    max_distance_to_neuron_average= np.mean(distance_to_sample_max_list) #biggest difference to a single neuron in average    
    max_distance_to_neuron_min = np.min(distance_to_sample_max_list) #biggest difference to a single neuron for closest sample
    
    distances_dict = {
        'z_score_aggregate': z_score_aggregate,
        'distance_to_initialization_aggregate': distance_to_initialization_aggregate,
        'distance_to_sample_average': distance_to_sample_average,
        'distance_to_sample_min': distance_to_sample_min,
        'max_distance_to_neuron_average': max_distance_to_neuron_average,
        'max_distance_to_neuron_min': max_distance_to_neuron_min,        
    }
    
    return distances_dict



def generate_dataset_from_distributions_line(line_distribution_parameters,
                                                number_of_samples_function, 
                                                max_distributions_per_class_function, 
                                                config,
                                                random_parameters_distribution=True,
                                                flip_percentage=0.0,
                                                data_noise=0.0,
                                                seed_function=100_000):
    try:
        if config['data']['max_distributions_per_class'] != 0:
            samples_class_0_list = line_distribution_parameters[:config['data']['number_of_variables']*3].reshape(-1, 3).T[0]
            samples_class_0_list = np.array(samples_class_0_list).astype(np.int64)

            feature_weight_0_list = line_distribution_parameters[:config['data']['number_of_variables']*3].reshape(-1, 3).T[1]
            feature_weight_0_list = np.array(feature_weight_0_list).astype(np.float32)

            seed_shuffeling_list = line_distribution_parameters[:config['data']['number_of_variables']*3].reshape(-1, 3).T[2]
            seed_shuffeling_list = np.array(seed_shuffeling_list).astype(np.int64)

            line_distribution_parameters = line_distribution_parameters[config['data']['number_of_variables']*3:]
        else:
            samples_class_0_list = [np.nan]* config['data']['number_of_variables']
            feature_weight_0_list = [np.nan]* config['data']['number_of_variables']
            seed_shuffeling_list = [np.nan]* config['data']['number_of_variables']
    except:
        samples_class_0_list = [np.nan]* config['data']['number_of_variables']
        feature_weight_0_list = [np.nan]* config['data']['number_of_variables']
        seed_shuffeling_list = [np.nan]* config['data']['number_of_variables']
        
    if config['data']['max_distributions_per_class'] == 0:
        distribution_list = line_distribution_parameters.reshape(-1, 1+max_distributions_per_class_function*2)
    else:
        distribution_list = line_distribution_parameters.reshape(-1, 1+max_distributions_per_class_function*config['data']['num_classes']*2)    

    distribution_dict_list = []
    for i, distribution in enumerate(distribution_list):
        distribution_name = distribution[0][1:]
        distribution_parameters= distribution[1:]

        if config['data']['max_distributions_per_class'] == 0:
            distribution_parameters_0_param_1 = distribution_parameters.reshape(2, -1)[0]
            distribution_parameters_0_param_2 = distribution_parameters.reshape(2, -1)[1]
            distribution_parameters_1_param_1 = distribution_parameters_0_param_1
            distribution_parameters_1_param_2 = distribution_parameters_0_param_2
        else:               
            distribution_parameters_0_param_1 = distribution_parameters.reshape(4, -1)[0]
            distribution_parameters_0_param_2 = distribution_parameters.reshape(4, -1)[1]
            distribution_parameters_1_param_1 = distribution_parameters.reshape(4, -1)[2]
            distribution_parameters_1_param_2 = distribution_parameters.reshape(4, -1)[3]

        distribution_parameters_0_param_1 = distribution_parameters_0_param_1[distribution_parameters_0_param_1 != ' NaN'].astype(np.float64)
        distribution_parameters_0_param_2 = distribution_parameters_0_param_2[distribution_parameters_0_param_2 != ' NaN'].astype(np.float64)
        distribution_parameters_1_param_1 = distribution_parameters_1_param_1[distribution_parameters_1_param_1 != ' NaN'].astype(np.float64)
        distribution_parameters_1_param_2 = distribution_parameters_1_param_2[distribution_parameters_1_param_2 != ' NaN'].astype(np.float64)

        if len(distribution_parameters_0_param_1) == 1:
            distribution_parameters_0_param_1 = distribution_parameters_0_param_1[0]
        if len(distribution_parameters_0_param_2) == 1:
            distribution_parameters_0_param_2 = distribution_parameters_0_param_2[0]
        if len(distribution_parameters_1_param_1) == 1:
            distribution_parameters_1_param_1 = distribution_parameters_1_param_1[0]
        if len(distribution_parameters_1_param_2) == 1:
            distribution_parameters_1_param_2 = distribution_parameters_1_param_2[0]        
        distribution_dict = None

        if distribution_name == 'normal':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'loc': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'loc': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}
        elif distribution_name == 'uniform':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'low': distribution_parameters_0_param_1,
                    'high': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'low': distribution_parameters_1_param_1,
                    'high': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}

        elif distribution_name == 'gamma':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'shape': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'shape': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}
        elif distribution_name == 'exponential':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'scale': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'scale': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}        
        elif distribution_name == 'beta':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'a': distribution_parameters_0_param_1,
                    'b': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'a': distribution_parameters_1_param_1,
                    'b': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}    
        elif distribution_name == 'binomial':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'n': distribution_parameters_0_param_1,
                    'p': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'n': distribution_parameters_1_param_1,
                    'p': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}    
        elif distribution_name == 'poisson':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'lam': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'lam': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}              
        elif distribution_name == 'lognormal':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'mean': distribution_parameters_0_param_1,
                    'mesigmaan': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'mean': distribution_parameters_1_param_1,
                    'sigma': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'f':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'dfnum': distribution_parameters_0_param_1,
                    'dfden': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'dfnum': distribution_parameters_1_param_1,
                    'dfden': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'logistic':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'loc': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'loc': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'weibull':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'a': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'a': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}             
            
            
        distribution_dict_list.append(distribution_dict)



    random_evaluation_dataset, _, _, _ = generate_dataset_from_distributions(distribution_list=distribution_list, 
                                                             number_of_variables=config['data']['number_of_variables'], 
                                                             number_of_samples=number_of_samples_function, 
                                                             distributions_per_class = max_distributions_per_class_function, 
                                                             seed = seed_function, 
                                                             flip_percentage=flip_percentage, 
                                                             data_noise=data_noise,
                                                             random_parameters=random_parameters_distribution,
                                                             distribution_dict_list=distribution_dict_list,
                                                             config=config)   
    
    return random_evaluation_dataset

######################################################################################################################################################################################################################

def unstack_array_to_list(a, axis=0):
    return list(np.moveaxis(a, axis, 0))


class AABB:
    """Axis-aligned bounding box"""
    def __init__(self, n_features):
        self.limits = np.array([[-np.inf, np.inf]] * n_features)

    def split(self, f, v):
        left = AABB(self.limits.shape[0])
        right = AABB(self.limits.shape[0])
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()

        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right


def tree_bounds(tree, n_features=None):
    """Compute final decision rule for each node in tree"""
    if n_features is None:
        n_features = np.max(tree.feature) + 1
    aabbs = [AABB(n_features) for _ in range(tree.node_count)]
    queue = deque([0])
    while queue:
        i = queue.pop()
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != ctree.TREE_LEAF:
            aabbs[l], aabbs[r] = aabbs[i].split(tree.feature[i], tree.threshold[i])
            queue.extend([l, r])
    return aabbs


def decision_areas(tree_classifier, maxrange, x=0, y=1, n_features=None):
    
    """ Extract decision areas.

    tree_classifier: Instance of a sklearn.tree.DecisionTreeClassifier
    maxrange: values to insert for [left, right, top, bottom] if the interval is open (+/-inf) 
    x: index of the feature that goes on the x axis
    y: index of the feature that goes on the y axis
    n_features: override autodetection of number of features
    """
    tree = tree_classifier.tree_
    aabbs = tree_bounds(tree, n_features)

    rectangles = []
    for i in range(len(aabbs)):
        if tree.children_left[i] != ctree.TREE_LEAF:
            continue
        l = aabbs[i].limits
        r = [l[x, 0], l[x, 1], l[y, 0], l[y, 1], np.argmax(tree.value[i])]
        rectangles.append(r)
    rectangles = np.array(rectangles)
    rectangles[:, [0, 2]] = np.maximum(rectangles[:, [0, 2]], maxrange[0::2])
    rectangles[:, [1, 3]] = np.minimum(rectangles[:, [1, 3]], maxrange[1::2])
    return rectangles

def plot_areas(rectangles):
    for rect in rectangles:
        color = ['b', 'r'][int(rect[4])]
        print(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        rp = Rectangle([rect[0], rect[2]], 
                       rect[1] - rect[0], 
                       rect[3] - rect[2], color=color, alpha=0.3)
        plt.gca().add_artist(rp)
        
        
        
        
        
def plot_decision_area_evaluation(X_train, 
                                    y_train, 
                                    X_test, 
                                    y_test,
                                    random_data,
                                    random_data_labels,
                                    network,
                                    tree_train_data,
                                    tree_uniform_data,
                                    tree_normal_data,
                                    tree_random_data,
                                    inet_dt_params,
                                    column_names,
                                    config):

    if X_train.shape[1] > 2:
        if True:
            tree_feature_importance = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])#dt_distilled_list_test[0][index][1]
            tree_feature_importance.fit(X_train, y_train)

            feature_index = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][:2]))
            filler_features = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][2:]))
        else:
            tree_feature_importance = xgb.XGBClassifier(learning_rate=0.1)#RandomForestClassifier()#dt_distilled_list_test[0][index][1]
            tree_feature_importance.fit(X_train, y_train)
            print('Fidelity Feature Importance Model:\t', accuracy_score(np.round(network.predict(X_test)), np.round(tree_feature_importance.predict(X_test)))) 
            print('Feature Importances:\t\t', tree_feature_importance.feature_importances_)

            feature_index = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][:2]))
            filler_features = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][2:]))        
    else:
        feature_index = [0, 1]
        filler_features = []


    filler_feature_values = {}
    filler_feature_ranges = {}
    for feature in filler_features:
        filler_feature_values[feature] = np.median(X_train[:,feature])#value
        filler_feature_ranges[feature] = np.max([np.abs(np.max(X_train[:,feature]) - filler_feature_values[feature]), np.abs(np.min(X_train[:,feature]) - filler_feature_values[feature])])+0.01

        
    if config['function_family']['dt_type'] == 'vanilla':
        tree_inet = parameterDT(inet_dt_params, config)
    else:
        tree_inet = SDT(input_dim=config['data']['number_of_variables'],
                           output_dim=config['data']['num_classes'],
                           depth=config['function_family']['maximum_depth'],
                           beta=config['function_family']['beta'],
                           decision_sparsity=config['function_family']['decision_sparsity'],
                           use_cuda=False,
                           verbosity=0)
        if config['i_net']['function_representation_type'] == 1:
            tree_inet.initialize_from_parameter_array(inet_dt_params)
        else:
            print('RESHAPE0')
            tree_inet.initialize_from_parameter_array(inet_dt_params, reshape=True, config=config)

    preds_network = np.round(network.predict(X_test))
    acc_network = accuracy_score(y_test, preds_network)

    preds_tree_sklearn_train_data = np.round(tree_train_data.predict(X_test))
    acc_inet_sklearn_train_data = accuracy_score(preds_network, preds_tree_sklearn_train_data)

    preds_tree_sklearn_random = np.round(tree_random_data.predict(X_test))
    acc_inet_sklearn_random = accuracy_score(preds_network, preds_tree_sklearn_random)

    preds_tree_inet = np.round(tree_inet.predict(X_test))
    acc_inet_tree = accuracy_score(preds_network, preds_tree_inet)
    
    
    preds_tree_sklearn_uniform_data = np.round(tree_uniform_data.predict(X_test))
    acc_inet_sklearn_uniform_data = accuracy_score(preds_network, preds_tree_sklearn_uniform_data)
    
    preds_tree_sklearn_normal_data = np.round(tree_normal_data.predict(X_test))
    acc_inet_sklearn_normal_data = accuracy_score(preds_network, preds_tree_sklearn_normal_data)
    
    print('Considered Columns:\t\t', '   '.join(list(column_names[feature_index])))
    print('Performance Network:\t\t', acc_network)
    print('Fidelity DT Sklearn Train Data:\t', acc_inet_sklearn_train_data)
    print('Fidelity DT Sklearn Random:\t', acc_inet_sklearn_random)
    print('Fidelity DT Sklearn Uniform Data:\t\t', acc_inet_sklearn_uniform_data)  
    print('Fidelity DT Sklearn Normal Data:\t\t', acc_inet_sklearn_normal_data)  
    print('Fidelity DT I-Net:\t\t', acc_inet_tree)


    #gs = gridspec.GridSpec(2, 2)
    gs = gridspec.GridSpec(1, 6)

    fig = plt.figure(figsize=(40,10))

    labels = ['Neual Network', 'Distilled DT (Train Data)', 'Distilled DT (Random Data)', 'Distilled DT (U(0,1))', 'Distilled DT (N(0,1))', 'Distilled DT (I-Net)'] 
    for i, (clf, lab) in enumerate(zip([network, tree_train_data, tree_random_data, tree_uniform_data, tree_normal_data, tree_inet],
                             labels)):

        #ax = plt.subplot(gs[i//2, i%2])
        ax = plt.subplot(gs[i])
        if lab == 'Distilled DT (Random Data)' and random_data is not None and False:
            fig = plot_decision_regions(X=np.vstack([X_train, random_data]),
                              y=np.hstack([y_train.astype(np.int64), random_data_labels.astype(np.int64)]), 
                              X_highlight=X_train,
                              clf=clf,
                              feature_index=feature_index, #these one will be plotted  
                              filler_feature_values=filler_feature_values,  #these will be ignored (value used for prediction)
                              filler_feature_ranges=filler_feature_ranges, #these will be ignored (value +- feature range used for plotting data)
                              legend=2)
        else:
            fig = plot_decision_regions(X=X_train,
                              y=y_train.astype(np.int64), 
                              clf=clf,
                              feature_index=feature_index, #these one will be plotted  
                              filler_feature_values=filler_feature_values,  #these will be ignored (value used for prediction)
                              filler_feature_ranges=filler_feature_ranges, #these will be ignored (value +- feature range used for plotting data)
                              legend=2)            
        plt.title(lab)
        ax.set_xlabel(feature_index[0])
        ax.set_ylabel(feature_index[1])
        plt.xlim([0, 1])
        plt.ylim([0, 1])


    plt.show()
    
    
    
    
def get_distribution_dict_from_parameters(distribution_name, 
                                          distribution_parameters_0_param_1, 
                                          distribution_parameters_0_param_2, 
                                          distribution_parameters_1_param_1, 
                                          distribution_parameters_1_param_2,
                                          samples_class_0 = None,
                                          feature_weight_0 = 1,
                                          seed_shuffeling = None):
    
        if distribution_name == 'normal':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'loc': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'loc': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,       
            }}
        elif distribution_name == 'uniform':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'low': distribution_parameters_0_param_1,
                    'high': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'low': distribution_parameters_1_param_1,
                    'high': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,       
            }}

        elif distribution_name == 'gamma':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'shape': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'shape': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,         
            }}
        elif distribution_name == 'exponential':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'scale': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'scale': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,          
            }}        
        elif distribution_name == 'beta':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'a': distribution_parameters_0_param_1,
                    'b': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'a': distribution_parameters_1_param_1,
                    'b': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,         
            }}    
        elif distribution_name == 'binomial':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'n': distribution_parameters_0_param_1,
                    'p': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'n': distribution_parameters_1_param_1,
                    'p': distribution_parameters_1_param_2,            
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,         
            }}    
        elif distribution_name == 'poisson':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'lam': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'lam': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0,
                'feature_weight_0': feature_weight_0,
                'seed_shuffeling': seed_shuffeling,         
            }}  
        elif distribution_name == 'lognormal':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'mean': distribution_parameters_0_param_1,
                    'mesigmaan': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'mean': distribution_parameters_1_param_1,
                    'sigma': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'f':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'dfnum': distribution_parameters_0_param_1,
                    'dfden': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'dfnum': distribution_parameters_1_param_1,
                    'dfden': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'logistic':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'loc': distribution_parameters_0_param_1,
                    'scale': distribution_parameters_0_param_2,
                },
                'class_1': {
                    'loc': distribution_parameters_1_param_1,
                    'scale': distribution_parameters_1_param_2,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }} 
        elif distribution_name == 'weibull':
            distribution_dict = {distribution_name: {
                'class_0': {
                    'a': distribution_parameters_0_param_1,
                },
                'class_1': {
                    'a': distribution_parameters_1_param_1,
                },
                'samples_class_0': samples_class_0_list[i],
                'feature_weight_0': feature_weight_0_list[i],
                'seed_shuffeling': seed_shuffeling_list[i],       
            }}              
            
            
        return distribution_dict  
    
    
    
    
    
       
        
def plot_decision_area_evaluation_all_distrib(X_train, 
                                            y_train, 
                                            X_test, 
                                            y_test,
                                            random_data_dict,
                                            random_data_labels_dict,                                              
                                            network,
                                            tree_train_data,
                                            tree_uniform_data,
                                            tree_normal_data,
                                            tree_random_data_list,
                                            inet_dt_params,
                                            column_names,
                                            distrib_list,
                                            config,
                                            identifier_folder = None,
                                            identifier_file = None):

    if X_train.shape[1] > 2:
        if True:
            tree_feature_importance = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])#dt_distilled_list_test[0][index][1]
            tree_feature_importance.fit(X_train, y_train)

            feature_index = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][:2]))
            filler_features = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][2:]))
        else:
            tree_feature_importance = xgb.XGBClassifier(learning_rate=0.1)#RandomForestClassifier()#dt_distilled_list_test[0][index][1]
            tree_feature_importance.fit(X_train, y_train)
            print('Fidelity Feature Importance Model (XGBoost):\t', accuracy_score(np.round(network.predict(X_test)), np.round(tree_feature_importance.predict(X_test)))) 
            #print('Feature Importances:\t\t', tree_feature_importance.feature_importances_)

            feature_index = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][:2]))
            filler_features = list(np.sort(np.argsort(tree_feature_importance.feature_importances_)[::-1][2:]))        
    else:
        feature_index = [0, 1]
        filler_features = []


    filler_feature_values = {}
    filler_feature_ranges = {}
    for feature in filler_features:
        filler_feature_values[feature] = np.median(X_train[:,feature])#value
        filler_feature_ranges[feature] = np.max([np.abs(np.max(X_train[:,feature]) - filler_feature_values[feature]), np.abs(np.min(X_train[:,feature]) - filler_feature_values[feature])])+0.01

        
    if config['function_family']['dt_type'] == 'vanilla':
        tree_inet = parameterDT(inet_dt_params, config)
    else:
        tree_inet = SDT(input_dim=config['data']['number_of_variables'],
                           output_dim=config['data']['num_classes'],
                           depth=config['function_family']['maximum_depth'],
                           beta=config['function_family']['beta'],
                           decision_sparsity=config['function_family']['decision_sparsity'],
                           use_cuda=False,
                           verbosity=0)
        if config['i_net']['function_representation_type'] == 1:
            print('RESHAPE0')
            tree_inet.initialize_from_parameter_array(inet_dt_params)
        else:
            tree_inet.initialize_from_parameter_array(inet_dt_params, reshape=True, config=config)
    print('Considered Columns:\t\t\t\t\t\t\t\t', '   '.join(list(column_names[feature_index])))
    
    preds_network = np.round(network.predict(X_test))
    acc_network = accuracy_score(y_test, preds_network)
    print('Performance Network:\t\t\t\t\t\t\t\t', acc_network)
    
    preds_tree_sklearn_train_data = np.round(tree_train_data.predict(X_test))
    acc_inet_sklearn_train_data = accuracy_score(preds_network, preds_tree_sklearn_train_data)
    print('Fidelity DT Sklearn Train Data:\t\t\t\t\t\t\t', acc_inet_sklearn_train_data)    
    
    for i, distrib in enumerate(distrib_list):
        preds_tree_sklearn_random = np.round(tree_random_data_list[i].predict(X_test))
        acc_inet_sklearn_random = accuracy_score(preds_network, preds_tree_sklearn_random)
        print('Fidelity DT Distilled (' + str(distrib) + '):   \t', acc_inet_sklearn_random)

    preds_tree_sklearn_uniform_data = np.round(tree_uniform_data.predict(X_test))
    acc_inet_sklearn_uniform_data = accuracy_score(preds_network, preds_tree_sklearn_uniform_data)
    print('Fidelity DT Sklearn Uniform Data:\t\t\t\t\t\t', acc_inet_sklearn_uniform_data)    
    
    preds_tree_sklearn_normal_data = np.round(tree_normal_data.predict(X_test))
    acc_inet_sklearn_normal_data = accuracy_score(preds_network, preds_tree_sklearn_normal_data)
    print('Fidelity DT Sklearn Normal Data:\t\t\t\t\t\t', acc_inet_sklearn_normal_data)  
    
    preds_tree_inet = np.round(tree_inet.predict(X_test))
    acc_inet_tree = accuracy_score(preds_network, preds_tree_inet)


    print('Fidelity DT I-Net:\t\t\t\t\t\t\t\t', acc_inet_tree)


    #gs = gridspec.GridSpec(2, 2)
    gs = gridspec.GridSpec(1, 3+len(distrib_list)+2)

    fig = plt.figure(figsize=((3+len(distrib_list)+2)*5,5))

    distrib_list_string = []
    for distrib in distrib_list:
        if '[' in str(distrib):
            new_distrib = 'Random'
            distrib_list_string.append(new_distrib)
        else:
            distrib_list_string.append(str(distrib))    
        
    labels = flatten_list(['Neual Network', 'Distilled DT (Train Data)', ['Distilled DT (' + str(distrib) + ')' for distrib in distrib_list_string], 'Distilled DT (U(0,1))', 'Distilled DT (N(0,1))',  'Distilled DT (I-Net)'])
    counter = 0
    for i, (clf, lab) in enumerate(zip(flatten_list([network, tree_train_data, tree_random_data_list, tree_uniform_data, tree_normal_data, tree_inet]),
                             labels)):
        if 'Distilled DT' in lab and 'Train Data' not in lab and 'I-Net' not in lab and random_data_dict is not None and False:
            #ax = plt.subplot(gs[i//2, i%2])
            ax = plt.subplot(gs[i])
            fig = plot_decision_regions(X=np.vstack([X_train, random_data_dict[distrib_list[counter]]]),
                              y=np.hstack([y_train.astype(np.int64), random_data_labels_dict[distrib_list[counter]].astype(np.int64)]), 
                              X_highlight=X_train,
                              clf=clf,
                              feature_index=feature_index, #these one will be plotted  
                              filler_feature_values=filler_feature_values,  #these will be ignored (value used for prediction)
                              filler_feature_ranges=filler_feature_ranges, #these will be ignored (value +- feature range used for plotting data)
                              legend=2)
            plt.title(lab, fontsize=25)
            ax.set_xlabel(feature_index[0], fontsize=20)
            ax.set_ylabel(feature_index[1], fontsize=20)
            plt.xlim([0, 1])
            plt.ylim([0, 1])   
            counter += 1
            
        else:

            #ax = plt.subplot(gs[i//2, i%2])
            ax = plt.subplot(gs[i])
            fig = plot_decision_regions(X=X_train,
                              y=y_train.astype(np.int64), 
                              clf=clf,
                              feature_index=feature_index, #these one will be plotted  
                              filler_feature_values=filler_feature_values,  #these will be ignored (value used for prediction)
                              filler_feature_ranges=filler_feature_ranges, #these will be ignored (value +- feature range used for plotting data)
                              legend=2)
            plt.title(lab, fontsize=15)
            plt.xticks(fontsize=10) #, rotation=90
            ax.set_xlabel('x' + str(feature_index[0]),  fontsize=12)
            ax.set_ylabel('x' + str(feature_index[1]),  fontsize=12)
            plt.xlim([0, 1])
            plt.ylim([0, 1])


    if identifier_folder is not None and config['data']['number_of_variables'] == 2:
        paths_dict = generate_paths(config, path_type = 'interpretation_net')
    
        plt.savefig('./data/distrib_plots/' + identifier_folder + '/' + identifier_file + '.pdf', bbox_inches = 'tight', pad_inches = 0)
    

    plt.show()
    print('-----------------------------------------------------')
    
    
    
    
def evaluate_network_on_distribution_custom_parameters(distribution_name_feature_0,
                                                       distribution_name_feature_1,
                                                       distribution_parameters_0_param_1_feature_0,
                                                       distribution_parameters_0_param_2_feature_0,
                                                       distribution_parameters_1_param_1_feature_0,
                                                       distribution_parameters_1_param_2_feature_0,
                                                       distribution_parameters_0_param_1_feature_1,
                                                       distribution_parameters_0_param_2_feature_1,
                                                       distribution_parameters_1_param_1_feature_1,
                                                       distribution_parameters_1_param_2_feature_1,
                                                       inet,
                                                       config,
                                                       distribution_list_evaluation):

    if config['data']['number_of_variables']  == 2:

        distribution_1_dict  = get_distribution_dict_from_parameters(distribution_name = distribution_name_feature_0, 
                                                                      distribution_parameters_0_param_1 = distribution_parameters_0_param_1_feature_0, 
                                                                      distribution_parameters_0_param_2 = distribution_parameters_0_param_2_feature_0, 
                                                                      distribution_parameters_1_param_1 = distribution_parameters_1_param_1_feature_0, 
                                                                      distribution_parameters_1_param_2 = distribution_parameters_1_param_2_feature_0)

        distribution_2_dict  = get_distribution_dict_from_parameters(distribution_name = distribution_name_feature_1, 
                                                                      distribution_parameters_0_param_1 = distribution_parameters_0_param_1_feature_1, 
                                                                      distribution_parameters_0_param_2 = distribution_parameters_0_param_2_feature_1, 
                                                                      distribution_parameters_1_param_1 = distribution_parameters_1_param_1_feature_1, 
                                                                      distribution_parameters_1_param_2 = distribution_parameters_1_param_2_feature_1)    

        distribution_dict_list = [distribution_1_dict, distribution_2_dict]

        X_data, y_data, distribution_parameter_list, normalizer_list = generate_dataset_from_distributions(distribution_list = None, 
                                                                                                        number_of_variables = config['data']['number_of_variables'], 
                                                                                                        number_of_samples = config['data']['lambda_dataset_size'], 
                                                                                                        distributions_per_class = config['data']['max_distributions_per_class'], 
                                                                                                        seed = config['computation']['RANDOM_SEED'], 
                                                                                                        flip_percentage = config['data']['noise_injected_level'],  
                                                                                                        data_noise = config['data']['data_noise'],  
                                                                                                        random_parameters = config['data']['random_parameters_distribution'], 
                                                                                                        distribution_dict_list = distribution_dict_list,
                                                                                                        config=config)


        X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_test_valid(X_data, y_data, valid_frac=0.25, test_frac=0.1, seed=config['computation']['RANDOM_SEED'])

        test_network, model_history = train_network_real_world_data(X_train, y_train, X_valid, y_valid, config, verbose=0) 

        if config['function_family']['dt_type'] == 'vanilla':

            tree_train_data = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])

            y_train_network = np.round(test_network.predict(X_train)).astype(np.int64)
            tree_train_data.fit(X_train, y_train_network)

            y_test_pred_tree_train_data = tree_train_data.predict(X_test)
            y_test_pred_network = np.round(test_network.predict(X_test)).astype(np.int64)  

            ##print('Accuracy  NN:\t\t', accuracy_score(y_test, y_test_pred_network))
            ##print('Fidelity Train Data DT:\t', accuracy_score(y_test_pred_network, y_test_pred_tree_train_data))

            tree_random_data_dict = {}
            accuracy_tree_random_data_dict = {}
            random_data_dict = {}
            random_data_labels_dict = {}
            
            for distribution in distribution_list_evaluation:

                random_data = generate_random_data_points_custom(config['data']['x_min'], 
                                                                 config['data']['x_max'],
                                                                 config['evaluation']['per_network_optimization_dataset_size'], 
                                                                 config['data']['number_of_variables'], 
                                                                 config['data']['categorical_indices'],
                                                                 distrib=distribution,
                                                                 random_parameters=config['data']['random_parameters_distribution'],
                                                                 distrib_param_max=config['data']['distrib_param_max'],
                                                                 seed=config['computation']['RANDOM_SEED'],
                                                                 config=config)

                for i, column in enumerate(random_data.T):
                    scaler = MinMaxScaler()
                    scaler.fit(column.reshape(-1, 1))
                    random_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()

                tree_random_data = DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
                random_data_labels = np.round(test_network.predict(random_data)).astype(np.int64)
                tree_random_data.fit(random_data, random_data_labels)

                y_test_pred_tree_random_data = tree_random_data.predict(X_test)
                y_test_pred_network = np.round(test_network.predict(X_test)).astype(np.int64)
                accuracy_tree_random_data = accuracy_score(y_test_pred_network, y_test_pred_tree_random_data)

                ##print('Fidelity Random Data DT (' + distribution + '):\t', accuracy_tree_random_data)

                accuracy_tree_random_data_dict[str(distribution)] = accuracy_tree_random_data 
                tree_random_data_dict[str(distribution)] = tree_random_data
                
                random_data_list[str(distribution)] = random_data 
                random_data_labels_list[str(distribution)] = random_data_labels

        elif config['function_family']['dt_type'] == 'SDT':

            tree_train_data = SDT(input_dim=X_train.shape[1],#X_train.shape[1], 
                                   output_dim=2,#int(max(y_train))+1, 
                                   depth=config['function_family']['maximum_depth'],
                                   #beta=0,
                                   decision_sparsity=config['function_family']['decision_sparsity'],#-1,
                                   random_seed=config['computation']['RANDOM_SEED'],
                                   use_cuda=False,
                                   verbosity=0)

            y_train_network = np.round(test_network.predict(X_train)).astype(np.int64)
            tree_train_data.fit(X_train, y_train_network, epochs=50)         

            y_test_pred_tree_train_data = tree_train_data.predict(X_test)
            y_test_pred_network = np.round(test_network.predict(X_test)).astype(np.int64)  

            ##print('Accuracy  NN:\t\t', accuracy_score(y_test, y_test_pred_network))
            ##print('Fidelity Train Data DT:\t', accuracy_score(y_test_pred_network, y_test_pred_tree_train_data))

            tree_random_data_dict = {}
            accuracy_tree_random_data_dict = {}
            random_data_dict = {}
            random_data_labels_dict = {}
            
            for distribution in distribution_list_evaluation:

                random_data = generate_random_data_points_custom(config['data']['x_min'], 
                                                                 config['data']['x_max'],
                                                                 config['evaluation']['per_network_optimization_dataset_size'], 
                                                                 config['data']['number_of_variables'], 
                                                                 config['data']['categorical_indices'],
                                                                 distrib=distribution,
                                                                 random_parameters=config['data']['random_parameters_distribution'],
                                                                 distrib_param_max=config['data']['distrib_param_max'],
                                                                 seed=config['computation']['RANDOM_SEED'],
                                                                 config=config)

                for i, column in enumerate(random_data.T):
                    scaler = MinMaxScaler()
                    scaler.fit(column.reshape(-1, 1))
                    random_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()

                tree_random_data = SDT(input_dim=random_data.shape[1],#X_train.shape[1], 
                                       output_dim=2,#int(max(y_train))+1, 
                                       depth=config['function_family']['maximum_depth'],
                                       #beta=0,
                                       decision_sparsity=config['function_family']['decision_sparsity'],#-1,
                                       random_seed=config['computation']['RANDOM_SEED'],
                                       use_cuda=False,
                                       verbosity=0)            

                random_data_labels = np.round(test_network.predict(random_data)).astype(np.int64)
                tree_random_data.fit(random_data, random_data_labels, epochs=50)

                y_test_pred_tree_random_data = tree_random_data.predict(X_test)
                y_test_pred_network = np.round(test_network.predict(X_test)).astype(np.int64)
                accuracy_tree_random_data = accuracy_score(y_test_pred_network, y_test_pred_tree_random_data)

                ##print('Fidelity Random Data DT (' + distribution + '):\t', accuracy_tree_random_data)

                accuracy_tree_random_data_dict[str(distribution)] = accuracy_tree_random_data 
                tree_random_data_dict[str(distribution)] = tree_random_data
                
                random_data_list[str(distribution)] = random_data 
                random_data_labels_list[str(distribution)] = random_data_labels
                
                
        if False:
            key_best_distribution = list(tree_random_data_dict.keys())[np.argmax(tree_random_data_dict.values())]
            print('Best Distribution\t', key_best_distribution)

            test_network_parameters_flat = shaped_network_parameters_to_array(test_network.get_weights(), config)

            dt_inet = inet.predict(np.array([test_network_parameters_flat]))[0]

            plot_decision_area_evaluation(X_train, 
                                        y_train,
                                        X_test, 
                                        y_test, 
                                        [random_data_list[str(distribution)] for distribution in distribution_list_evaluation],
                                        [random_data_labels_list[str(distribution)] for distribution in distribution_list_evaluation],
                                        test_network,
                                        tree_train_data,
                                        tree_random_data_dict[key_best_distribution],
                                        dt_inet,
                                        np.array([str(i) for i in range(X_train.shape[1])]),
                                        config
                                       )
        else:
            #key_best_distribution = list(tree_random_data_dict.keys())[np.argmax(tree_random_data_dict.values())]
            #print('Best Distribution', key_best_distribution)

            test_network_parameters_flat = shaped_network_parameters_to_array(test_network.get_weights(), config)

            dt_inet = inet.predict(np.array([test_network_parameters_flat]))[0]

            plot_decision_area_evaluation_all_distrib(X_train, 
                                                        y_train,
                                                        X_test, 
                                                        y_test, 
                                                        [random_data_list[str(distribution)] for distribution in distribution_list_evaluation],
                                                        [random_data_labels_list[str(distribution)] for distribution in distribution_list_evaluation],
                                                        test_network,
                                                        tree_train_data,
                                                        [tree_random_data_dict[str(distribution)] for distribution in distribution_list_evaluation],
                                                        dt_inet,
                                                        np.array([str(i) for i in range(X_train.shape[1])]),
                                                        distribution_list_evaluation,
                                                        config
                                                       )
            
            
def plot_class_distrib_by_feature(model,
                                  index,
                                  test_network,
                                  distribution_training,
                                  distribution_dict,
                                  X_test,
                                  config):

    evaluation_result_dict, results_list, test_network_parameters, dt_inet, dt_distilled_list = evaluate_network_real_world_data(model,
                                                                                                test_network, 
                                                                                                _, 
                                                                                                X_test, 
                                                                                                dataset_size_list=[10000],
                                                                                                config=config,
                                                                                                distribution=distribution_training)


    results_list_extended = results_list[0]

    results_list = results_list_extended
    evaluation_result_dict = results_list

    test_network_parameters = test_network_parameters#[:1]
    dt_inet = dt_inet#[:1]
    dt_distilled_list = dt_distilled_list#[:1]

    print(evaluation_result_dict)
    
    print('Class 0: ', test_network.predict(X_test[np.where(test_network.predict(X_test).ravel()<0.5)]).shape[0])
    print('Class 1: ', test_network.predict(X_test[np.where(test_network.predict(X_test).ravel()>=0.5)]).shape[0])    
    
    ############################################################################################################################################
    ############################################################################################################################################
    
    colors_list = ['green','blue','yellow','cyan','magenta','pink']

    if config['data']['number_of_variables'] > 4:
        fig,ax = plt.subplots(nrows=np.ceil(config['data']['number_of_variables']*2/4).astype(int), ncols=4,figsize=(20,15))
    else:
        fig,ax = plt.subplots(nrows=np.ceil(config['data']['number_of_variables']*2/2).astype(int), ncols=2,figsize=(20,15))

    for axis_1 in ax:
        for axis_2 in axis_1:
            axis_2.set_xlim([0, 1])                          
    
    plot_index = 0

    for i in range(X_test.shape[1]):
        try:
            distribution_parameter = distribution_dict[i]
            print(distribution_parameter)
        except:
            pass
        colors = colors_list[i%6]

        x = X_test[:,i][np.where(test_network.predict(X_test).ravel()<=0.5)]
        plt.subplot(np.ceil(config['data']['number_of_variables']*2/4).astype(int), 4,plot_index+1)
        plt.hist(x,bins=[i/10 for i in range(11)],color=colors)
        try:
            plt.title(list(distribution_parameter.keys())[0] + ' Class 0' )
        except:
            pass            
        plot_index += 1

        x = X_test[:,i][np.where(test_network.predict(X_test).ravel()>0.5)]
        plt.subplot(np.ceil(config['data']['number_of_variables']*2/4).astype(int),4,plot_index+1)
        plt.hist(x,bins=[i/10 for i in range(11)],color=colors)
        try:
            plt.title(list(distribution_parameter.keys())[0] + ' Class 1' )
        except:
            pass        
        plot_index += 1

    fig.subplots_adjust(hspace=0.4,wspace=.3) 
    plt.suptitle('Sampling from Various Distributions',fontsize=20)
    plt.show()