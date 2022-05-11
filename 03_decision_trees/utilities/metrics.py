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

#from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
#from similaritymeasures import frechet_dist, area_between_two_curves, dtw
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


import tensorflow as tf
#import keras
import random 
#import tensorflow_addons as tfa

#udf import
from utilities.LambdaNet import *
#from utilities.metrics import *
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
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_metrics_config_from_curent_notebook(config):
    try:
        globals().update(config['data'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['lambda_net'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['i_net'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['evaluation'])
    except KeyError:
        print(KeyError)
        
    try:
        globals().update(config['computation'])
    except KeyError:
        print(KeyError)
        
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if int(tf.__version__[0]) >= 2:
        tf.random.set_seed(RANDOM_SEED)
    else:
        tf.set_random_seed(RANDOM_SEED)
        
                    
        
#######################################################################################################################################################
######################Manual TF Loss function for comparison with lambda-net prediction based (predictions made in loss function)######################
#######################################################################################################################################################

def compute_loss_single_tree_wrapper(config):

    def compute_loss_single_tree(input_list):

        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth']  

        (splits_features_true, splits_values_true, leaf_probabilities_true, splits_features_pred, splits_values_pred, leaf_probabilities_pred) = input_list

        if True:
            loss_internal_feature = []
            true_features = []
            for internal_node_true, internal_node_pred in zip(tf.split(splits_features_true, internal_node_num_), tf.split(splits_features_pred, internal_node_num_)):
                loss_internal_feature.append(tf.cast(tf.equal(tf.argmax(tf.squeeze(internal_node_true)), tf.argmax(tf.squeeze(internal_node_pred))), tf.int64))

                true_features.append(tf.argmax(tf.squeeze(internal_node_true)))

            #loss_internal_complete = 0
            loss_internal_complete_list = []
            for internal_node_true, internal_node_pred, correct_feature_identifier, true_feature_index in zip(tf.split(splits_values_true, internal_node_num_), tf.split(splits_values_pred, internal_node_num_), loss_internal_feature, true_features):                    
                split_value_true = tf.gather(tf.squeeze(internal_node_true), true_feature_index)
                split_value_pred = tf.gather(tf.squeeze(internal_node_pred), true_feature_index)

                loss_internal = tf.reduce_max([(1.0-tf.cast(correct_feature_identifier, tf.float32)), tf.keras.metrics.mean_absolute_error([split_value_true], [split_value_pred])]) #loss = 1 if wrong feature, else split_distance
                loss_internal_complete_list.append(loss_internal)
                #loss_internal_complete += loss_internal        


            #loss_leaf_complete = 0   
            loss_leaf_complete_list = []
            for leaf_node_true, leaf_node_pred in zip(tf.split(leaf_probabilities_true, leaf_node_num_), tf.split(leaf_probabilities_pred, leaf_node_num_)):
                loss_leaf = tf.keras.metrics.binary_crossentropy(leaf_node_true, leaf_node_pred)
                loss_leaf_complete_list.append(loss_leaf)
                #loss_leaf_complete += loss_leaf

            loss_internal_complete = tf.reduce_mean(loss_internal_complete_list)
            loss_leaf_complete = tf.reduce_mean(loss_leaf_complete_list)

            loss_complete = loss_internal_complete + loss_leaf_complete * 0.5
        else:
            #pass
            splits_true = splits_features_true #* splits_values_true
            splits_pred = splits_features_pred #* tfa.seq2seq.hardmax(splits_values_pred)

            error_splits = tf.reduce_mean(tf.keras.metrics.mean_squared_error(splits_true, splits_pred))
            error_leaf = tf.keras.metrics.mean_squared_error(leaf_probabilities_true, leaf_probabilities_pred)
            #tf.print('splits_true', splits_true.shape, splits_true)
            #tf.print('splits_pred', splits_pred.shape, splits_pred)
            #tf.print('error_splits', error_splits.shape, error_splits)
            #tf.print('error_leaf', error_leaf.shape, error_leaf)        
            loss_complete = tf.reduce_mean([error_splits, error_leaf])
            

        return loss_complete 

    return compute_loss_single_tree



def inet_decision_function_fv_loss_wrapper_parameters(config):   
            
    def inet_decision_function_fv_loss_parameters(function_true_with_network_parameters, function_pred):     

        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth'] 

        config['function_family']['basic_function_representation_length'] = internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes']
        
        decision_tree_parameters = function_true_with_network_parameters[:, : config['function_family']['basic_function_representation_length']]   
           
        decision_tree_parameters = tf.dtypes.cast(tf.convert_to_tensor(decision_tree_parameters), tf.float32)

        splits_features_true = tf.reshape(decision_tree_parameters[:,:internal_node_num_ * config['data']['number_of_variables']], shape=(-1, internal_node_num_, config['data']['number_of_variables']))
        splits_values_true = tf.reshape(decision_tree_parameters[:,internal_node_num_* config['data']['number_of_variables']:internal_node_num_ * config['data']['number_of_variables'] * 2], shape=(-1, internal_node_num_, config['data']['number_of_variables']))
        leaf_probabilities_true = tf.reshape(decision_tree_parameters[:,internal_node_num_ * config['data']['number_of_variables'] * 2:], shape=(-1, leaf_node_num_, config['data']['num_classes']))[:,:,0]

        splits_features_pred = tf.reshape(function_pred[:,:internal_node_num_ * config['data']['number_of_variables']], shape=(-1, internal_node_num_, config['data']['number_of_variables']))
        splits_values_pred = tf.reshape(function_pred[:,internal_node_num_* config['data']['number_of_variables']:internal_node_num_ * config['data']['number_of_variables'] * 2], shape=(-1, internal_node_num_, config['data']['number_of_variables']))

        leaf_probabilities_pred = tf.reshape(function_pred[:,internal_node_num_ * config['data']['number_of_variables'] * 2:internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_], shape=(-1, leaf_node_num_))
        
        if True:
            loss_complete_list = tf.vectorized_map(compute_loss_single_tree_wrapper(config), (splits_features_true, 
                                                                                        splits_values_true, 
                                                                                        leaf_probabilities_true, 
                                                                                        splits_features_pred, 
                                                                                        splits_values_pred, 
                                                                                        leaf_probabilities_pred))
        else:
            loss_complete_list = tf.map_fn(compute_loss_single_tree_wrapper(config), (splits_features_true, 
                                                                                        splits_values_true, 
                                                                                        leaf_probabilities_true, 
                                                                                        splits_features_pred, 
                                                                                        splits_values_pred, 
                                                                                        leaf_probabilities_pred), fn_output_signature=(tf.float32))            
        
        ####tf.print('FINAL loss_complete_list', loss_complete_list)

        loss_complete = tf.reduce_mean(loss_complete_list)


        def mae_wrapper(input_list):
            return tf.keras.metrics.mean_absolute_error(input_list[0], input_list[1])
        
        if False:
            loss_coeff = tf.reduce_mean(tf.vectorized_map(mae_wrapper, (splits_features_true, splits_features_pred)))
            loss_leaf = tf.reduce_mean(tf.vectorized_map(mae_wrapper, (leaf_probabilities_true, leaf_probabilities_pred)))
            #loss_coeff = tf.keras.metrics.mean_absolute_error(splits_features_true, splits_features_pred)
            #loss_leaf = tf.keras.metrics.mean_absolute_error(leaf_probabilities_true, leaf_probabilities_pred)
            loss_complete = loss_coeff + loss_leaf
        
        ####tf.print('FINAL loss_complete', loss_complete)
        
        return loss_complete #*tf.random.uniform(0,1, 1) #tf.reduce_mean(function_pred)#
    

    inet_decision_function_fv_loss_parameters.__name__ = config['i_net']['loss'] + '_' + inet_decision_function_fv_loss_parameters.__name__        


    return inet_decision_function_fv_loss_parameters




def inet_decision_function_fv_loss_wrapper(model_lambda_placeholder, network_parameters_structure, config, use_distribution_list):   
                 
    def inet_decision_function_fv_loss(function_true_with_network_parameters, function_pred):      
        
        if not config['i_net']['function_value_loss']:
            internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
            leaf_node_num_ = 2 ** config['function_family']['maximum_depth'] 
            
            config['function_family']['basic_function_representation_length'] = internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes']
        
        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth']    
        
        
        #tf.print("internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes']", internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes'])
        #tf.print("config['function_family']['basic_function_representation_length']", config['function_family']['basic_function_representation_length'])
        
        network_parameters = function_true_with_network_parameters[:,config['function_family']['basic_function_representation_length']: config['function_family']['basic_function_representation_length'] + config['lambda_net']['number_of_lambda_weights']]         
        function_true = function_true_with_network_parameters[:,:config['function_family']['basic_function_representation_length']]
        distribution_line_array = function_true_with_network_parameters[:, config['function_family']['basic_function_representation_length'] + config['lambda_net']['number_of_lambda_weights']:]
        
        if config['i_net']['nas']:
            function_pred = function_pred[:,:config['function_family']['function_representation_length']]
            
        network_parameters = tf.dtypes.cast(tf.convert_to_tensor(network_parameters), tf.float32)
        function_true = tf.dtypes.cast(tf.convert_to_tensor(function_true), tf.float32)
        function_pred = tf.dtypes.cast(tf.convert_to_tensor(function_pred), tf.float32)
                
        assert network_parameters.shape[1] == config['lambda_net']['number_of_lambda_weights'], 'Shape of Network Parameters: ' + str(network_parameters.shape)  
        assert function_true.shape[1] == config['function_family']['basic_function_representation_length'], 'Shape of True Function: ' + str(function_true.shape)      
        assert function_pred.shape[1] == config['function_family']['function_representation_length'], 'Shape of Pred Function: ' + str(function_pred.shape)   
        
        #tf.print('GO function_values_array_function_true')
        
        #function_values_array_function_true = tf.map_fn(calculate_function_value_from_lambda_net_parameters_wrapper(random_evaluation_dataset, network_parameters_structure, model_lambda_placeholder), network_parameters, fn_output_signature=tf.float32)  
        
        #tf.print('function_values_array_function_true', function_values_array_function_true)
        if use_distribution_list:
            function_values_array_function_true, function_values_array_function_pred, penalties = tf.map_fn(calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list), (network_parameters, function_pred, distribution_line_array), fn_output_signature=(tf.float32, tf.float32, tf.float32))       
        else:
            function_values_array_function_true, function_values_array_function_pred, penalties = tf.map_fn(calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list), (network_parameters, function_pred), fn_output_signature=(tf.float32, tf.float32, tf.float32))        
                
                
        def loss_function_wrapper(loss_function_name):
            
            def loss_function(input_list):                    
                nonlocal loss_function_name
                function_values_true = input_list[0]
                function_values_pred = input_list[1]
                
                penalized = False
                if '_penalized' in loss_function_name:
                    penalized = True
                    loss_function_name = loss_function_name.replace('_penalized','')
                    
                if loss_function_name == 'soft_binary_crossentropy':
                    function_values_true_diff = tf.math.subtract(1.0, function_values_true)
                    function_values_true_softmax = tf.stack([function_values_true, function_values_true_diff], axis=1)
                    
                    function_values_pred_diff = tf.math.subtract(1.0, function_values_pred)
                    function_values_pred_softmax = tf.stack([function_values_pred, function_values_pred_diff], axis=1)
                    
                    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(function_values_true_softmax, function_values_pred_softmax))
                else:                    
                    if 'soft_' not in loss_function_name:
                        function_values_true = tf.math.round(function_values_true)  
                        loss_function_name = loss_function_name.replace('soft_','')
                    
                    loss = tf.keras.losses.get(loss_function_name)
                    loss_value = loss(function_values_true, function_values_pred)
                if penalized:
                    #tf.print('loss_value', loss_value.shape, loss_value)
                    #tf.print('penalties', penalties.shape, penalties)
                    loss_value = loss_value * penalties
                
                return loss_value
                
            return loss_function
        
        #tf.print('function_values_array_function_true', function_values_array_function_true, summarize=10)
        #tf.print('function_values_array_function_pred', function_values_array_function_pred, summarize=10)
        loss_per_sample = tf.vectorized_map(loss_function_wrapper(config['i_net']['loss']), (function_values_array_function_true, function_values_array_function_pred))
        #tf.print('loss_per_sample', loss_per_sample)
        loss_value = tf.math.reduce_mean(loss_per_sample)
        #tf.print(loss_value)
    
        #loss_value = tf.math.reduce_mean(function_true - function_pred)
        #tf.print(loss_value)
        
        return loss_value
    
    inet_decision_function_fv_loss.__name__ = config['i_net']['loss'] + '_' + inet_decision_function_fv_loss.__name__        


    return inet_decision_function_fv_loss



def calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list):
    
    
    def calculate_function_values_loss_decision(input_list):  
             
        network_parameters = input_list[0]
        function_array = input_list[1]
        if use_distribution_list:
            distribution_line = input_list[2]
        #tf.print('distribution_line', distribution_line, summarize=10)
        if not use_distribution_list:
            random_evaluation_dataset = generate_random_data_points_custom(config['data']['x_min'], config['data']['x_max'], config['evaluation']['random_evaluation_dataset_size'], config['data']['number_of_variables'], categorical_indices=None, distrib=config['evaluation']['random_evaluation_dataset_distribution'])
            
        else:
            random_evaluation_dataset = tf.reshape(tensor=distribution_line, shape=(-1,config['data']['number_of_variables']))

        random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)   
        
        function_values_true = calculate_function_value_from_lambda_net_parameters_wrapper(random_evaluation_dataset, network_parameters_structure, model_lambda_placeholder, config)(network_parameters)
        #tf.print('function_values_true', function_values_true[:50], summarize=50)
        
        function_values_pred = tf.zeros_like(function_values_true)
        if config['function_family']['dt_type'] == 'SDT':
            function_values_pred, penalty = calculate_function_value_from_decision_tree_parameters_wrapper(random_evaluation_dataset, config)(function_array)
        elif config['function_family']['dt_type'] == 'vanilla':
            function_values_pred, penalty = calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(random_evaluation_dataset, config)(function_array)
        #tf.print('function_values_pred', function_values_pred[:50], summarize=50)
            
      
        function_values_true_ones_rounded = tf.math.reduce_sum(tf.cast(tf.equal(tf.round(function_values_true), 1), tf.float32))
        function_values_pred_ones_rounded = tf.math.reduce_sum(tf.cast(tf.equal(tf.round(function_values_pred), 1), tf.float32))
        
        ## tf.print('function_values_true_ones_rounded', function_values_true_ones_rounded, len(function_values_true)-function_values_true_ones_rounded, 'function_values_pred_ones_rounded', function_values_pred_ones_rounded, len(function_values_pred)-function_values_pred_ones_rounded)
        threshold = 5
        penalty_value = 2.0
        
        if False:
            if tf.less(function_values_pred_ones_rounded, config['evaluation']['random_evaluation_dataset_size']/threshold) and tf.greater(function_values_true_ones_rounded, config['evaluation']['random_evaluation_dataset_size']/threshold/2):
                penalty = 1 + penalty_value
            elif tf.greater(function_values_pred_ones_rounded, config['evaluation']['random_evaluation_dataset_size']-config['evaluation']['random_evaluation_dataset_size']/threshold) and tf.less(function_values_true_ones_rounded, config['evaluation']['random_evaluation_dataset_size']-config['evaluation']['random_evaluation_dataset_size']/threshold/2):
                penalty = 1 + penalty_value
            else:
                penalty = 1.0            
        else:
            fraction = tf.reduce_max([function_values_true_ones_rounded/function_values_pred_ones_rounded, function_values_pred_ones_rounded/function_values_true_ones_rounded])  
            if tf.greater(fraction, tf.cast(threshold, tf.float32)):
                penalty = tf.reduce_min([20, 1.0 + fraction])#**(1.5)
                #tf.print(penalty)
            else: 
                penalty = 1.0
                
        return function_values_true, function_values_pred, penalty
    
    return calculate_function_values_loss_decision



def calculate_function_value_from_lambda_net_parameters_wrapper(random_evaluation_dataset, network_parameters_structure, model_lambda_placeholder, config):
    
    random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)    
            
    #@tf.function(jit_compile=True)
    def calculate_function_value_from_lambda_net_parameters(network_parameters):
        i = 0
        index = 0
        if config['lambda_net']['use_batchnorm_lambda']:
            start = 0
            for i in range((len(network_parameters_structure)-2)//6):
                # set weights of layer
                index = i*6
                size = np.product(network_parameters_structure[index])
                weights_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2].weights[0].assign(weights_tf_true)
                start += size

                # set biases of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                biases_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2].weights[1].assign(biases_tf_true)
                start += size    
                
                # set batchnorm of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                batchnorm_1_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2+1].weights[0].assign(batchnorm_1_tf_true)
                start += size       
                
                # set batchnorm of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                batchnorm_2_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2+1].weights[1].assign(batchnorm_2_tf_true)
                start += size   
                
                # set batchnorm of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                batchnorm_3_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2+1].weights[2].assign(batchnorm_3_tf_true)
                start += size    
                
                # set batchnorm of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                batchnorm_4_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i*2+1].weights[3].assign(batchnorm_4_tf_true)
                start += size    
                
        
            index += 1
            size = np.product(network_parameters_structure[index])
            weights_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
            model_lambda_placeholder.layers[i*2+1+1].weights[0].assign(weights_tf_true)
            start += size

            # set biases of layer
            index += 1
            size = np.product(network_parameters_structure[index])
            biases_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
            model_lambda_placeholder.layers[i*2+1+1].weights[1].assign(biases_tf_true)
            start += size        

        else:
            #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
            # build models
            start = 0
            for i in range(len(network_parameters_structure)//2):
                # set weights of layer
                index = i*2
                size = np.product(network_parameters_structure[index])
                weights_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
                start += size

                # set biases of layer
                index += 1
                size = np.product(network_parameters_structure[index])
                biases_tf_true = tf.reshape(network_parameters[start:start+size], network_parameters_structure[index])
                model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
                start += size

        lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(random_evaluation_dataset))
        #tf.print('lambda_fv ones', tf.math.count_nonzero(tf.math.round(lambda_fv)), 'lambda_fv zeros', len(lambda_fv)-tf.math.count_nonzero(tf.math.round(lambda_fv)))
        
        return lambda_fv
    return calculate_function_value_from_lambda_net_parameters

def calculate_function_value_from_decision_tree_parameters_wrapper(random_evaluation_dataset, config):
        
    random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)        
    
    
    maximum_depth = config['function_family']['maximum_depth']
    leaf_node_num_ = 2 ** maximum_depth
    
    #@tf.function(jit_compile=True)
    def calculate_function_value_from_decision_tree_parameters(function_array):    
        
        from utilities.utility_functions import get_shaped_parameters_for_decision_tree      
        #tf.print('function_array', function_array, summarize=-1)
        
        weights, biases, leaf_probabilities = get_shaped_parameters_for_decision_tree(function_array, config)
        
        #tf.print('weights', weights, summarize=-1)
        #tf.print('biases', biases, summarize=-1)
        #tf.print('leaf_probabilities', leaf_probabilities, summarize=-1)
        
        function_values_sdt = tf.vectorized_map(calculate_function_value_from_decision_tree_parameter_single_sample_wrapper(weights, biases, leaf_probabilities, leaf_node_num_, maximum_depth), random_evaluation_dataset)
        
        #penalty = tf.cast((tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 0)) or tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 1))), tf.float32) * 1.25
        
        #penalty = tf.math.maximum(tf.cast((tf.math.reduce_all(tf.equal(tf.round(function_values_sdt), 0)) or tf.math.reduce_all(tf.equal(tf.round(function_values_sdt), 1))), tf.float32) * 1.25, 1)        
        
        return function_values_sdt, tf.constant(1.0, dtype=tf.float32)#penalty
    return calculate_function_value_from_decision_tree_parameters



def calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(random_evaluation_dataset, config):
                
    random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)       
    
    maximum_depth = config['function_family']['maximum_depth']
    leaf_node_num_ = 2 ** maximum_depth
    internal_node_num_ = 2 ** maximum_depth - 1

    #@tf.function(jit_compile=True)
    def calculate_function_value_from_vanilla_decision_tree_parameters(function_array):
                            
        from utilities.utility_functions import get_shaped_parameters_for_decision_tree
            
        #tf.print('function_array', function_array)
        weights, leaf_probabilities = get_shaped_parameters_for_decision_tree(function_array, config)
        #tf.print('weights', weights)
        #tf.print('leaf_probabilities', leaf_probabilities)
        
        function_values_vanilla_dt = tf.vectorized_map(calculate_function_value_from_vanilla_decision_tree_parameter_single_sample_wrapper(weights, leaf_probabilities, leaf_node_num_, internal_node_num_, maximum_depth, config['data']['number_of_variables']), random_evaluation_dataset)
        #tf.print('function_values_vanilla_dt', function_values_vanilla_dt, summarize=-1)
        

        
        #penalty = tf.math.maximum(tf.cast((tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 0)) or tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 1))), tf.float32) * 1.25, 1)

        return function_values_vanilla_dt, tf.constant(1.0, dtype=tf.float32)#penalty
    return calculate_function_value_from_vanilla_decision_tree_parameters




def calculate_function_value_from_decision_tree_parameter_single_sample_wrapper(weights, biases, leaf_probabilities, leaf_node_num_, maximum_depth):
    
    weights = tf.cast(weights, tf.float32)
    biases = tf.cast(biases, tf.float32)
    leaf_probabilities = tf.cast(leaf_probabilities, tf.float32)   
    
    #@tf.function(jit_compile=True)
    def calculate_function_value_from_decision_tree_parameter_single_sample(evaluation_entry):
        
        evaluation_entry = tf.cast(evaluation_entry, tf.float32)
        
        path_prob = tf.expand_dims(tf.sigmoid(tf.add(tf.reduce_sum(tf.multiply(weights, evaluation_entry), axis=1), biases)), axis=0)
        #tf.print(path_prob)
        path_prob = tf.expand_dims(path_prob, axis=2)
        #tf.print(path_prob)
        path_prob = tf.concat((path_prob, 1 - path_prob), axis=2)
        #tf.print(path_prob)        

        begin_idx = 0
        end_idx = 1 

        _mu = tf.fill((1,1,1), 1.0)

        for layer_idx in range(0, maximum_depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _mu =  tf.repeat(tf.reshape(_mu, (1,-1,1)), 2, axis=2)
            #tf.print('_mu', _mu)
            _mu = _mu * _path_prob
            #tf.print('_mu', _mu)
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)    

        _mu = tf.reshape(_mu, (1, leaf_node_num_))
        #tf.print(_mu)   

        cond = tf.equal(_mu, tf.reduce_max(_mu))
        _mu = tf.where(cond, _mu, tf.zeros_like(_mu))
        #tf.print(_mu)

        y_pred = tf.reduce_sum(_mu * leaf_probabilities, axis=1)
        #tf.print(y_pred)
        y_pred = tf.nn.softmax(y_pred)[1]
        
        return y_pred
    
    return calculate_function_value_from_decision_tree_parameter_single_sample


def calculate_function_value_from_vanilla_decision_tree_parameter_single_sample_wrapper(weights, leaf_probabilities, leaf_node_num_, internal_node_num_, maximum_depth, number_of_variables):
        
    weights = tf.cast(weights, tf.float32)
    leaf_probabilities = tf.cast(leaf_probabilities, tf.float32)   
    
    #@tf.function(jit_compile=True)
    def calculate_function_value_from_vanilla_decision_tree_parameter_single_sample(evaluation_entry):
         
        evaluation_entry = tf.cast(evaluation_entry, tf.float32)
        
        weights_split = tf.split(weights, internal_node_num_)
        weights_split_new = [[] for _ in range(maximum_depth)]
        for i, tensor in enumerate(weights_split):
            current_depth = np.ceil(np.log2((i+1)+1)).astype(np.int32)

            weights_split_new[current_depth-1].append(tf.squeeze(tensor, axis=0))
            
        weights_split = weights_split_new
        
        #TDOD if multiclass, take index of min and max of leaf_proba to generate classes
        #leaf_probabilities_split = tf.split(leaf_probabilities, leaf_node_num_)
        #leaf_classes_list = []
        #for leaf_probability in leaf_probabilities_split:
        #    leaf_classes = tf.stack([tf.argmax(leaf_probability), tf.argmin(leaf_probability)])
        #    leaf_classes_list.append(leaf_classes)
        #leaf_classes = tf.keras.backend.flatten(tf.stack(leaf_classes_list))
        
        split_value_list = []

        for i in range(maximum_depth):
            #print('LOOP 1 ', i)
            current_depth = i+1#np.ceil(np.log2((i+1)+1)).astype(np.int32)
            num_nodes_current_layer = 2**current_depth - 1 - (2**(current_depth-1) - 1)
            #print('current_depth', current_depth, 'num_nodes_current_layer', num_nodes_current_layer)
            split_value_list_per_depth = []
            for j in range(num_nodes_current_layer):
                #tf.print('weights_split[i][j]', weights_split[i][j])
                #print('LOOP 2 ', j)
                zero_identifier = tf.not_equal(weights_split[i][j], tf.zeros_like(weights_split[i][j]))
                #tf.print('zero_identifier', zero_identifier)
                split_complete = tf.greater(evaluation_entry, weights_split[i][j])
                #tf.print('split_complete', split_complete, 'evaluation_entry', evaluation_entry, 'weights_split[i][j]', weights_split[i][j])
                split_value = tf.reduce_any(tf.logical_and(zero_identifier, split_complete))
                #tf.print('split_value', split_value)
                split_value_filled = tf.fill( [2**(maximum_depth-current_depth)] , split_value)
                split_value_neg_filled = tf.fill( [2**(maximum_depth-current_depth)], tf.logical_not(split_value))
                #tf.print('tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled]))', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
                #print('LOOP 2 OUTPUT', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
                split_value_list_per_depth.append(tf.keras.backend.flatten(tf.stack([split_value_neg_filled, split_value_filled])))        
                #tf.print('tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled]))', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
            #print('LOOP 1 OUTPUT', tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)))
            split_value_list.append(tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)))
            #tf.print('DT SPLITS ENCODED', tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)), summarize=-1)
                #node_index_in_layer += 1        
        #tf.print(split_value_list)
        #tf.print(tf.stack(split_value_list))
        #tf.print('split_value_list', split_value_list, summarize=-1)
        #tf.print('tf.stack(split_value_list)\n', tf.stack(split_value_list), summarize=-1)
        split_values = tf.cast(tf.reduce_all(tf.stack(split_value_list), axis=0), tf.float32)    
        #tf.print('split_values', split_values, summarize=-1)
        leaf_classes = tf.cast(leaf_probabilities, tf.float32)
        #tf.print('leaf_classes', leaf_classes, summarize=-1)
        final_class_probability = 1-tf.reduce_max(tf.multiply(leaf_classes, split_values))                                                                                                                                            
        return final_class_probability#y_pred
    
    return calculate_function_value_from_vanilla_decision_tree_parameter_single_sample
        




#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################



def inet_decision_function_fv_metric_wrapper(model_lambda_placeholder, network_parameters_structure, config, metric, use_distribution_list):
    
    
    def inet_decision_function_fv_metric(function_true_with_network_parameters, function_pred):    
        
        #random_evaluation_dataset = generate_random_data_points_custom(config['data']['x_min'], config['data']['x_max'], config['evaluation']['random_evaluation_dataset_size'], config['data']['number_of_variables'], categorical_indices=None, distrib=config['evaluation']['random_evaluation_dataset_distribution'])            
        #random_evaluation_dataset =  np.random.uniform(low=config['data']['x_min'], high=config['data']['x_max'], size=(config['evaluation']['random_evaluation_dataset_size'], config['data']['number_of_variables']))
        #random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)
        if not config['i_net']['function_value_loss']:
            internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
            leaf_node_num_ = 2 ** config['function_family']['maximum_depth'] 
            
            config['function_family']['basic_function_representation_length'] = internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes']            
            
        
        network_parameters = function_true_with_network_parameters[:,config['function_family']['basic_function_representation_length']: config['function_family']['basic_function_representation_length'] + config['lambda_net']['number_of_lambda_weights']]
        function_true = function_true_with_network_parameters[:,:config['function_family']['basic_function_representation_length']]
        distribution_line_array = function_true_with_network_parameters[:, config['function_family']['basic_function_representation_length'] + config['lambda_net']['number_of_lambda_weights']:]
        
        if config['i_net']['nas']:
            function_pred = function_pred[:,:config['function_family']['function_representation_length']]
            
        network_parameters = tf.dtypes.cast(tf.convert_to_tensor(network_parameters), tf.float32)
        function_true = tf.dtypes.cast(tf.convert_to_tensor(function_true), tf.float32)
        function_pred = tf.dtypes.cast(tf.convert_to_tensor(function_pred), tf.float32)
        
        assert network_parameters.shape[1] == config['lambda_net']['number_of_lambda_weights'], 'Shape of Network Parameters: ' + str(network_parameters.shape)            
        assert function_true.shape[1] == config['function_family']['basic_function_representation_length'], 'Shape of True Function: ' + str(function_true.shape)      
        assert function_pred.shape[1] == config['function_family']['function_representation_length'], 'Shape of Pred Function: ' + str(function_pred.shape)   
        
        if use_distribution_list:
            function_values_array_function_true, function_values_array_function_pred, penalties = tf.map_fn(calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list), (network_parameters, function_pred, distribution_line_array), fn_output_signature=(tf.float32, tf.float32, tf.float32))              
        else:        
            function_values_array_function_true, function_values_array_function_pred, penalties = tf.map_fn(calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list), (network_parameters, function_pred), fn_output_signature=(tf.float32, tf.float32, tf.float32))  
            
        def loss_function_wrapper(metric_name):
            def loss_function(input_list):                    
                nonlocal metric_name
                function_values_true = input_list[0]
                function_values_pred = input_list[1]
                
                penalized = False
                if '_penalized' in metric_name:
                    penalized = True
                    metric_name = metric_name.replace('_penalized','')
                    
                if metric_name == 'soft_binary_crossentropy':
                    function_values_true_diff = tf.math.subtract(1.0, function_values_true)
                    function_values_true_softmax = tf.stack([function_values_true, function_values_true_diff], axis=1)
                    
                    function_values_pred_diff = tf.math.subtract(1.0, function_values_pred)
                    function_values_pred_softmax = tf.stack([function_values_pred, function_values_pred_diff], axis=1)
                    
                    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(function_values_true_softmax, function_values_pred_softmax))
                else:
                    if 'soft_' not in metric_name:
                        function_values_true = tf.math.round(function_values_true)  
                        metric_name = metric_name.replace('soft_','')
                    
                    loss = tf.keras.metrics.get(metric_name)
                    loss_value = loss(function_values_true, function_values_pred)
                    
                if penalized:
                    loss_value = loss_value * penalties      
                    
                return loss_value
                
            return loss_function
        
        loss_per_sample = tf.vectorized_map(loss_function_wrapper(metric), (function_values_array_function_true, function_values_array_function_pred))
        #tf.print('loss_per_sample', loss_per_sample, summarize=-1)
        loss_value = tf.math.reduce_mean(loss_per_sample)
        #tf.print('loss_value', loss_value, summarize=-1)
    
        #loss_value = tf.math.reduce_mean(function_true - function_pred)
        #tf.print(loss_value)
        
        return loss_value
    
    inet_decision_function_fv_metric.__name__ = metric + '_' + inet_decision_function_fv_metric.__name__        


    return inet_decision_function_fv_metric

#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################







def r2_keras_loss(y_true, y_pred, epsilon=tf.keras.backend.epsilon()):

    SS_res =  tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)) 
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))) 
    return  - ( 1 - SS_res/(SS_tot + epsilon) )


#######################################################################################################################################################
######################################################Basic Keras/TF Loss functions####################################################################
#######################################################################################################################################################


def root_mean_squared_error(y_true, y_pred):   
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
        
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred)           
            
    return tf.math.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))) 

def accuracy_multilabel(y_true, y_pred, a_step=0.1):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))      
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.reduce_all(tf.keras.backend.equal(y_true, y_pred), axis=1), tf.int32), tf.float32))        

def accuracy_single(y_true, y_pred, a_step=0.1):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))
        
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.keras.backend.equal(y_true, y_pred), tf.int32), tf.float32))

def mean_absolute_percentage_error_keras(y_true, y_pred, epsilon=10e-3): 
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred)        
    epsilon = return_float_tensor_representation(epsilon)
        
    return tf.reduce_mean(tf.abs(tf.divide(tf.subtract(y_pred, y_true),(y_true + epsilon))))

def huber_loss_delta_set(y_true, y_pred):
    return keras.losses.huber_loss(y_true, y_pred, delta=0.3)



#######################################################################################################################################################
##########################################################Standard Metrics (no TF!)####################################################################
#######################################################################################################################################################


def mean_absolute_error_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.mean(np.abs(true_values-pred_values)))
    
    return np.mean(np.array(result_list))  

def mean_absolute_error_function_values_return_multi_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        #if np.isnan(true_values).all() or np.isnan(pred_values).all():
        #    continue
        #true_values = np.nan_to_num(true_values)
        #pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.mean(np.abs(true_values-pred_values)))
    
    return np.array(result_list) 

def mean_std_function_values_difference(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)
        
        result_list.append(np.std(true_values-pred_values))
    
    return np.mean(np.array(result_list))  

def root_mean_squared_error_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)        
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.sqrt(np.mean((true_values-pred_values)**2)))
    
    return np.mean(np.array(result_list)) 

def mean_absolute_percentage_error_function_values(y_true, y_pred, epsilon=10e-3):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.mean(np.abs(((true_values-pred_values)/(true_values+epsilon)))))

    return np.mean(np.array(result_list))

def r2_score_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(r2_score(true_values, pred_values))
    
    return np.mean(np.array(result_list))

def r2_score_function_values_return_multi_values(y_true, y_pred, epsilon=1e-07):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        #if np.isnan(true_values).all() or np.isnan(pred_values).all():
        #    continue
        #true_values = np.nan_to_num(true_values)
        #pred_values = np.nan_to_num(pred_values)        
        
        SS_res = np.sum(np.square(true_values - pred_values)) 
        SS_tot = np.sum(np.square(true_values - np.mean(true_values))) 
            
            
        result_list.append(( 1 - SS_res/(SS_tot + epsilon) )   )
    
    return np.array(result_list)

def relative_absolute_average_error_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.sum(np.abs(true_values-pred_values))/(true_values.shape[0]*np.std(true_values)))
    
    return np.mean(np.array(result_list))

def relative_maximum_average_error_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(np.max(true_values-pred_values)/np.std(true_values))
    
    return np.mean(np.array(result_list))

def mean_area_between_two_curves_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    assert number_of_variables==1
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)        
        
        result_list.append(area_between_two_curves(true_values, pred_values))
 
    return np.mean(np.array(result_list))

def mean_dtw_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)    

    result_list_single = []
    result_list_array = []
    
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)
        
        result_single_value, result_single_array = dtw(true_values, pred_values)
        result_list_single.append(result_single_value)
        result_list_array.append(result_single_array)
    
    return np.mean(np.array(result_list_single)), np.mean(np.array(result_list_array), axis=1)

def mean_frechet_dist_function_values(y_true, y_pred):
    
    from utilities.utility_functions import return_numpy_representation
    
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)   
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        if np.isnan(true_values).all() or np.isnan(pred_values).all():
            continue
        true_values = np.nan_to_num(true_values)
        pred_values = np.nan_to_num(pred_values)
        
        result_list.append(frechet_dist(true_values, pred_values))
    
    return np.mean(np.array(result_list))



#######################################################################################################################################################
#######################################################LAMBDA-NET METRICS##################################################################
#######################################################################################################################################################

def calcualate_function_value_with_X_data_entry(coefficient_list, X_data_entry):
    
    global list_of_monomial_identifiers
     
    result = 0    
    for coefficient_value, coefficient_multipliers in zip(coefficient_list, list_of_monomial_identifiers):
        partial_results = [X_data_value**coefficient_multiplier for coefficient_multiplier, X_data_value in zip(coefficient_multipliers, X_data_entry)]
        
        result += coefficient_value * reduce(lambda x, y: x*y, partial_results)
        
    return result, np.append(X_data_entry, result)


def calculate_function_values_from_polynomial(X_data, polynomial):
    function_value_list = []
    for entry in X_data:
        function_value, _ = calcualate_function_value_with_X_data_entry(polynomial, entry)
        function_value_list.append(function_value)
    function_value_array = np.array(function_value_list).reshape(len(function_value_list), 1)     

    return function_value_array

def generate_term_matric_for_lstsq(X_data, polynomial_indices):
    
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)    
    
    term_list_all = []
    y = 0
    for term in list(polynomial_indices):
        term_list = [int(value_mult) for value_mult in term]
        term_list_all.append(term_list)
    terms_matrix = []
    for unknowns in X_data:
        terms = []
        for term_multipliers in term_list_all:
            term_value = prod([unknown**multiplier for unknown, multiplier in zip(unknowns, term_multipliers)])
            terms.append(term_value)
        terms_matrix.append(np.array(terms))
    terms_matrix = np.array(terms_matrix)
    
    return terms_matrix

def root_mean_squared_error(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
        
    if tf.is_tensor(y_true):
        y_true = tf.dtypes.cast(y_true, tf.float32) 
    else:
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.dtypes.cast(y_true, tf.float32) 
    if tf.is_tensor(y_pred):
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
    else:
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
            
            
    return tf.math.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))) 

def accuracy_multilabel(y_true, y_pred, a_step=0.1):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    if 'float' in str(y_true[0].dtype):        
        if tf.is_tensor(y_true):
            y_true = tf.dtypes.cast(y_true, tf.float32) 
        else:
            y_true = y_true.astype('float32')
        if tf.is_tensor(y_pred):
            y_pred = tf.dtypes.cast(y_pred, tf.float32)
        else:
            y_pred = y_pred.astype('float32')
            
        n_digits = int(-np.log10(a_step))
        
        y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
        y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.reduce_all(tf.keras.backend.equal(y_true, y_pred), axis=1), tf.int32), tf.float32))

def accuracy_single(y_true, y_pred, a_step=0.1):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    if 'float' in str(y_true[0].dtype):        
        if tf.is_tensor(y_true):
            y_true = tf.dtypes.cast(y_true, tf.float32) 
        else:
            y_true = y_true.astype('float32')
        if tf.is_tensor(y_pred):
            y_pred = tf.dtypes.cast(y_pred, tf.float32)
        else:
            y_pred = y_pred.astype('float32')
            
        n_digits = int(-np.log10(a_step))
        
        y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
        y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.keras.backend.equal(y_true, y_pred), tf.int32), tf.float32))      

def mean_absolute_percentage_error_keras(y_true, y_pred, epsilon=10e-3): 
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values    
        
    if tf.is_tensor(y_true):
        y_true = tf.dtypes.cast(y_true, tf.float32) 
    else:
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.dtypes.cast(y_true, tf.float32) 
    if tf.is_tensor(y_pred):
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
    else:
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
        
    epsilon = tf.convert_to_tensor(epsilon)
    epsilon = tf.dtypes.cast(epsilon, tf.float32)
        
    return tf.reduce_mean(tf.abs(tf.divide(tf.subtract(y_pred, y_true),(y_true + epsilon))))

def huber_loss_delta_set(y_true, y_pred):
    return keras.losses.huber_loss(y_true, y_pred, delta=0.3)

def relative_absolute_average_error(y_true, y_pred):
    
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
       
    #error value calculation    
    result = np.sum(np.abs(y_true-y_pred))/(y_true.shape[0]*np.std(y_true)) #correct STD?
    
    return result

def relative_maximum_average_error(y_true, y_pred):
    
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    
    #error value calculation    
    result = np.max(y_true-y_pred)/np.std(y_true) #correct STD?
    
    return result


#######################################################################################################################################################
#########################################################I-NET EVALUATION FUNCTIONs####################################################################
#######################################################################################################################################################



def evaluate_interpretation_net(function_1_coefficients, 
                                function_2_coefficients, 
                                function_1_fv, 
                                function_2_fv):
    
    from utilities.utility_functions import return_numpy_representation
    #global list_of_monomial_identifiers
    
    if type(function_1_coefficients) != type(None) and type(function_2_coefficients) != type(None):
        function_1_coefficients = return_numpy_representation(function_1_coefficients)
        function_2_coefficients = return_numpy_representation(function_2_coefficients)     
        
        assert function_1_coefficients.shape[1] == sparsity or function_1_coefficients.shape[1] == interpretation_net_output_shape or function_1_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers), 'Coefficients Function 1 not in shape ' + str(function_1_coefficients.shape)
        assert function_2_coefficients.shape[1] == sparsity or function_2_coefficients.shape[1] == interpretation_net_output_shape or function_2_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers), 'Coefficients Function 2 not in shape ' + str(function_2_coefficients.shape)
       
        
        if function_1_coefficients.shape[1] != function_2_coefficients.shape[1]:
                
                
            if function_1_coefficients.shape[1] == interpretation_net_output_shape or function_1_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers):
                if function_1_coefficients.shape[1] == interpretation_net_output_shape:
                    coefficient_indices_array = function_1_coefficients[:,interpretation_net_output_monomials:]
                    function_1_coefficients_reduced = function_1_coefficients[:,:interpretation_net_output_monomials]

                    assert coefficient_indices_array.shape[1] == interpretation_net_output_monomials*sparsity or coefficient_indices_array.shape[1] == interpretation_net_output_monomials*(d+1)*n, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                    coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials, axis=1)

                    assert len(coefficient_indices_list) == function_1_coefficients_reduced.shape[1] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                    coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))
                elif function_1_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers):
                    coefficient_indices_array = function_1_coefficients[:,interpretation_net_output_monomials+1:]
                    function_1_coefficients_reduced = function_1_coefficients[:,:interpretation_net_output_monomials+1]

                    assert coefficient_indices_array.shape[1] == (interpretation_net_output_monomials+1)*sparsity, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                    coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials+1, axis=1)

                    assert len(coefficient_indices_list) == function_1_coefficients_reduced.shape[1] == interpretation_net_output_monomials+1, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                    coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))         


                function_2_coefficients_reduced = []
                for function_2_coefficients_entry, coefficient_indices_entry in zip(function_2_coefficients, coefficient_indices):
                    function_2_coefficients_reduced.append(function_2_coefficients_entry[[coefficient_indices_entry]])
                function_2_coefficients_reduced = np.array(function_2_coefficients_reduced)

                function_1_coefficients = function_1_coefficients_reduced
                function_2_coefficients = function_2_coefficients_reduced
            
            if function_2_coefficients.shape[1] == interpretation_net_output_shape or function_2_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers):
                if function_2_coefficients.shape[1] == interpretation_net_output_shape:
                    coefficient_indices_array = function_2_coefficients[:,interpretation_net_output_monomials:]
                    function_2_coefficients_reduced = function_2_coefficients[:,:interpretation_net_output_monomials]

                    assert coefficient_indices_array.shape[1] == interpretation_net_output_monomials*sparsity or coefficient_indices_array.shape[1] == interpretation_net_output_monomials*(d+1)*n, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                    coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials, axis=1)

                    assert len(coefficient_indices_list) == function_2_coefficients_reduced.shape[1] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                    coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))
                elif function_2_coefficients.shape[1] == interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers):
                    coefficient_indices_array = function_2_coefficients[:,interpretation_net_output_monomials+1:]
                    function_2_coefficients_reduced = function_2_coefficients[:,:interpretation_net_output_monomials+1]

                    assert coefficient_indices_array.shape[1] == (interpretation_net_output_monomials+1)*sparsity, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                    coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials+1, axis=1)

                    assert len(coefficient_indices_list) == function_2_coefficients_reduced.shape[1] == interpretation_net_output_monomials+1, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                    coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))                    
                

                function_1_coefficients_reduced = []
                for function_1_coefficients_entry, coefficient_indices_entry in zip(function_1_coefficients, coefficient_indices):
                    function_1_coefficients_reduced.append(function_1_coefficients_entry[[coefficient_indices_entry]])
                function_1_coefficients_reduced = np.array(function_1_coefficients_reduced)

                function_2_coefficients = function_2_coefficients_reduced
                function_1_coefficients = function_1_coefficients_reduced   
                
            if not ((function_2_coefficients.shape[1] != interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) or function_2_coefficients.shape[1] == interpretation_net_output_shape) and (function_1_coefficients.shape[1] != interpretation_net_output_shape + 1 + len(list_of_monomial_identifiers) or function_1_coefficients.shape[1] == interpretation_net_output_shape)):
                print(function_1_coefficients.shape)
                print(function_2_coefficients.shape)
                raise SystemExit('Shapes Inconsistent') 

                
                
        
        mae_coeff = np.round(mean_absolute_error(function_1_coefficients, function_2_coefficients), 4)
        rmse_coeff = np.round(root_mean_squared_error(function_1_coefficients, function_2_coefficients), 4)
        mape_coeff = np.round(mean_absolute_percentage_error_keras(function_1_coefficients, function_2_coefficients), 4)
        accuracy_coeff = np.round(accuracy_single(function_1_coefficients, function_2_coefficients), 4)
        accuracy_multi_coeff = np.round(accuracy_multilabel(function_1_coefficients, function_2_coefficients), 4)
    else:
        mae_coeff = np.nan
        rmse_coeff = np.nan
        mape_coeff = np.nan
        accuracy_coeff = np.nan
        accuracy_multi_coeff = np.nan
        

    try:    
        function_1_fv = return_numpy_representation(function_1_fv)
        function_2_fv = return_numpy_representation(function_2_fv)
    except Exception as e:
        
        print(function_1_fv)
        print(function_2_fv)   
        
        raise SystemExit(e)
        
    #print(function_1_fv)
    #print(function_2_fv)    
    
    assert function_1_fv.shape == function_2_fv.shape, 'Shape of Function 1 FVs: ' + str(function_1_fv.shape) + str(function_1_fv[:10])  + 'Shape of Functio 2 FVs' + str(function_2_fv.shape) + str(function_2_fv[:10])
        
    mae_fv = np.round(mean_absolute_error_function_values(function_1_fv, function_2_fv), 4)
    rmse_fv = np.round(root_mean_squared_error_function_values(function_1_fv, function_2_fv), 4)
    mape_fv = np.round(mean_absolute_percentage_error_function_values(function_1_fv, function_2_fv), 4)

    
    #print(function_1_fv[:10])
    #print(function_2_fv[:10])
    
    #function_1_fv = function_1_fv.astype('float32')
    #print(np.isnan(function_1_fv).any())
    #print(np.isinf(function_1_fv).any())
    #print(np.max(function_1_fv))
    #print(np.min(function_1_fv))
    
    #function_2_fv = function_2_fv.astype('float32')
    #print(np.isnan(function_2_fv).any())
    #print(np.isinf(function_2_fv).any())
    #print(np.max(function_2_fv))
    #print(np.min(function_2_fv))
    
    
    r2_fv = np.round(r2_score_function_values(function_1_fv, function_2_fv), 4)
    raae_fv = np.round(relative_absolute_average_error_function_values(function_1_fv, function_2_fv), 4)
    rmae_fv = np.round(relative_maximum_average_error_function_values(function_1_fv, function_2_fv), 4) 
    
    std_fv_diff = np.round(mean_std_function_values_difference(function_1_fv, function_2_fv), 4)
    mean_fv_1 = np.mean(function_1_fv)
    mean_fv_2 = np.mean(function_2_fv)
    std_fv_1 = np.std(function_1_fv)
    std_fv_2 = np.std(function_2_fv)

    mae_distribution = mean_absolute_error_function_values_return_multi_values(function_1_fv, function_2_fv)
    r2_distribution = r2_score_function_values_return_multi_values(function_1_fv, function_2_fv)

    return pd.Series(data=[mae_coeff,
                          rmse_coeff,
                          mape_coeff,
                          accuracy_coeff,
                          accuracy_multi_coeff,
                          
                          mae_fv,
                          rmse_fv,
                          mape_fv,
                          r2_fv,
                          raae_fv,
                          rmae_fv,
                          
                          std_fv_diff,
                           
                          mean_fv_1,
                          mean_fv_2,
                          std_fv_1,
                          std_fv_2],
                     index=['MAE',
                           'RMSE',
                           'MAPE',
                           'Accuracy',
                           'Accuracy Multilabel',
                           
                           'MAE FV',
                           'RMSE FV',
                           'MAPE FV',
                           'R2 FV',
                           'RAAE FV',
                           'RMAE FV',
                            
                           'MEAN STD FV DIFF',
                           'MEAN FV1',
                           'MEAN FV2',
                           'STD FV1',
                           'STD FV2']), {'MAE': pd.Series(data=mae_distribution, 
                                                  index=['L-' + str(i) for i in range(function_1_fv.shape[0])]),
                                        'R2': pd.Series(data=r2_distribution, 
                                                  index=['L-' + str(i) for i in range(function_1_fv.shape[0])])}