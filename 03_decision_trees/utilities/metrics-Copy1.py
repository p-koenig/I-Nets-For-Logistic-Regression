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




def inet_decision_function_fv_loss_wrapper(model_lambda_placeholder, network_parameters_structure, config, use_distribution_list):   
                 
    def inet_decision_function_fv_loss(function_true_with_network_parameters, function_pred):      
        
        
        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth']    
        
        
        #tf.print("internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes']", internal_node_num_ * config['data']['number_of_variables'] * 2 + leaf_node_num_ * config['data']['num_classes'])
        #tf.print("config['function_family']['basic_function_representation_length']", config['function_family']['basic_function_representation_length'])
        
        network_parameters = function_true_with_network_parameters[:,config['function_family']['basic_function_representation_length']: config['function_family']['basic_function_representation_length'] + config['lambda_net']['number_of_lambda_weights']]         
        function_true = function_true_with_network_parameters[:,:config['function_family']['basic_function_representation_length']]

    
            
        network_parameters = tf.dtypes.cast(tf.convert_to_tensor(network_parameters), tf.float32)
        function_true = tf.dtypes.cast(tf.convert_to_tensor(function_true), tf.float32)
        function_pred = tf.dtypes.cast(tf.convert_to_tensor(function_pred), tf.float32)
                
        assert network_parameters.shape[1] == config['lambda_net']['number_of_lambda_weights'], 'Shape of Network Parameters: ' + str(network_parameters.shape)  
        assert function_true.shape[1] == config['function_family']['basic_function_representation_length'], 'Shape of True Function: ' + str(function_true.shape)      
        assert function_pred.shape[1] == config['function_family']['function_representation_length'], 'Shape of Pred Function: ' + str(function_pred.shape)   
        
        #tf.print('GO function_values_array_function_true')
        
        #function_values_array_function_true = tf.map_fn(calculate_function_value_from_lambda_net_parameters_wrapper(random_evaluation_dataset, network_parameters_structure, model_lambda_placeholder), network_parameters, fn_output_signature=tf.float32)  
        
        #tf.print('function_values_array_function_true', function_values_array_function_true)

        function_values_array_function_true, function_values_array_function_pred, penalties = tf.map_fn(calculate_function_values_loss_decision_wrapper(network_parameters_structure, model_lambda_placeholder, config, use_distribution_list), (network_parameters, function_pred), fn_output_signature=(tf.float32, tf.float32, tf.float32))        
                
                
        def loss_function_wrapper(loss_function_name):
            
            def loss_function(input_list):                    
                nonlocal loss_function_name
                function_values_true = input_list[0]
                function_values_pred = input_list[1]
                
                function_values_true = tf.math.round(function_values_true)  
                    
                loss = tf.keras.losses.get(loss_function_name)
                loss_value = loss(function_values_true, function_values_pred)

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
        #tf.print('distribution_line', distribution_line, summarize=10)
        random_evaluation_dataset = generate_random_data_points_custom(config['data']['x_min'], config['data']['x_max'], config['evaluation']['random_evaluation_dataset_size'], config['data']['number_of_variables'], categorical_indices=None, distrib=config['evaluation']['random_evaluation_dataset_distribution'])
            
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
        
