#######################################################################################################################################################
#######################################################################Parameters######################################################################
#######################################################################################################################################################

RANDOM_SEED =42
d = 3  
n = 4

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
from similaritymeasures import frechet_dist, area_between_two_curves, dtw


import tensorflow as tf
import random 
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if int(tf.__version__[0]) >= 2:
    tf.random.set_seed(RANDOM_SEED)
else:
    tf.set_random_seed(RANDOM_SEED)

#udf import
from utilities.LambdaNet import *
#from utilities.metrics import *
from utilities.utility_functions import *

sparsity = nCr(n+d, d)

#######################################################################################################################################################
######################Manual TF Loss function for comparison with lambda-net prediction based (predictions made in loss function)######################
#######################################################################################################################################################
def mean_absolute_error_tf_fv_lambda_extended_wrapper(evaluation_dataset, list_of_monomial_identifiers, base_model):
    
    evaluation_dataset = return_float_tensor_representation(evaluation_dataset)
    list_of_monomial_identifiers = return_float_tensor_representation(list_of_monomial_identifiers)    
    
    model_lambda_placeholder = keras.models.clone_model(base_model)  
    
    weights_structure = base_model.get_weights()
    dims = [np_arrays.shape for np_arrays in weights_structure]
    
    def mean_absolute_error_tf_fv_lambda_extended(polynomial_true_with_lambda_fv, polynomial_pred):

        if seed_in_inet_training:
            network_parameters = polynomial_true_with_lambda_fv[:,sparsity+1:]
            polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]
        else:
            network_parameters = polynomial_true_with_lambda_fv[:,sparsity:]
            polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]
            
        network_parameters = return_float_tensor_representation(network_parameters)
        polynomial_true = return_float_tensor_representation(polynomial_true)
        polynomial_pred = return_float_tensor_representation(polynomial_pred)
        
        assert polynomial_true.shape[1] == sparsity
        assert polynomial_pred.shape[1] == sparsity
        assert network_parameters.shape[1] == number_of_lambda_weights 
        
        return tf.math.reduce_mean(tf.map_fn(calculate_mae_fv_lambda_wrapper(evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder), (polynomial_pred, network_parameters), fn_output_signature=tf.float32))
    return mean_absolute_error_tf_fv_lambda_extended

    def calculate_mae_fv_lambda_wrapper(evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder):

        def calculate_mae_fv_lambda(input_list):

            #single polynomials
            #polynomial_true = input_list[0]
            polynomial_pred = input_list[0]
            network_parameters = input_list[1]

            polynomial_pred_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))

            #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
            # build models
            start = 0
            layers = []
            for i in range(len(dims)//2):

                # set weights of layer
                index = i*2
                size = np.product(dims[index])
                weights_tf_true = tf.reshape(network_parameters[start:start+size], dims[index])
                model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
                start += size

                # set biases of layer
                index += 1
                size = np.product(dims[index])
                biases_tf_true = tf.reshape(network_parameters[start:start+size], dims[index])
                model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
                start += size


            lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(evaluation_dataset))

            return tf.math.reduce_mean(tf.vectorized_map(calculate_mae_single_input, (lambda_fv, polynomial_pred_fv_list)))

        return calculate_mae_fv_lambda



    #Manual TF Loss function for fv comparison of real and predicted polynomial

    def mean_absolute_error_tf_fv_poly_extended_wrapper(evaluation_dataset, list_of_monomial_identifiers):

        evaluation_dataset = return_float_tensor_representation(evaluation_dataset)
        list_of_monomial_identifiers = return_float_tensor_representation(list_of_monomial_identifiers)        

        def mean_absolute_error_tf_fv_poly_extended(polynomial_true, polynomial_pred):

            polynomial_true = return_float_tensor_representation(polynomial_true)
            polynomial_pred = return_float_tensor_representation(polynomial_pred)

            assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape)
            assert polynomial_pred.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_pred.shape)       

            return tf.math.reduce_mean(tf.map_fn(calculate_mae_fv_poly_wrapper(evaluation_dataset, list_of_monomial_identifiers), (polynomial_true, polynomial_pred), fn_output_signature=tf.float32))
        return mean_absolute_error_tf_fv_poly_extended

    def calculate_mae_fv_poly_wrapper(evaluation_dataset, list_of_monomial_identifiers):

        def calculate_mae_fv_poly(input_list):

            #single polynomials
            polynomial_true = input_list[0]
            polynomial_pred = input_list[1]

            polynomial_true_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_true), (evaluation_dataset))
            polynomial_pred_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))

            return tf.math.reduce_mean(tf.vectorized_map(calculate_mae_single_input, (polynomial_true_fv_list, polynomial_pred_fv_list)))

        return calculate_mae_fv_poly


    #GENERAL LOSS UTILITY FUNCTIONS
    def calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_pred):


        def calculate_fv_from_data(evaluation_entry):


            value_without_coefficient = tf.vectorized_map(calculate_value_without_coefficient_wrapper(evaluation_entry), (list_of_monomial_identifiers))
            polynomial_pred_value_per_term = tf.vectorized_map(lambda x: x[0]*x[1], (value_without_coefficient, polynomial_pred))

            polynomial_pred_fv = tf.reduce_sum(polynomial_pred_value_per_term)     

            return polynomial_pred_fv
        return calculate_fv_from_data


    #calculate intermediate term (without coefficient multiplication)
    def calculate_value_without_coefficient_wrapper(evaluation_entry):
        def calculate_value_without_coefficient(coefficient_multiplier_term):      

            return tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**x[1], (evaluation_entry, coefficient_multiplier_term)))
        return calculate_value_without_coefficient

    #calculate MAE at the end ---> general:REPLACE FUNCTION WITH LOSS CALL OR LAMBDA
    def calculate_mae_single_input(input_list):
        true_fv = input_list[0]
        pred_fv = input_list[1]

        return tf.math.abs(tf.math.subtract(true_fv, pred_fv))




#BASIC COEFFICIENT-BASED LOSS IF X_DATA IS APPENDED
def mean_absolute_error_extended(polynomial_true_with_lambda_fv, polynomial_pred): 
    
    if seed_in_inet_training:
        assert polynomial_true_with_lambda_fv.shape[1] == sparsity+number_of_lambda_weights+1
    else:
        assert polynomial_true_with_lambda_fv.shape[1] == sparsity+number_of_lambda_weights
    
    polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]    
    
    assert polynomial_true.shape[1] == sparsity
    assert polynomial_pred.shape[1] == sparsity
    
    return tf.keras.losses.MAE(polynomial_true, polynomial_pred)


def r2_tf_fv_lambda_extended_wrapper(evaluation_dataset, list_of_monomial_identifiers, base_model):
    
    evaluation_dataset = return_float_tensor_representation(evaluation_dataset)
    list_of_monomial_identifiers = return_float_tensor_representation(list_of_monomial_identifiers)    
    
    model_lambda_placeholder = keras.models.clone_model(base_model)  
    
    weights_structure = base_model.get_weights()
    dims = [np_arrays.shape for np_arrays in weights_structure]
    
    def r2_tf_fv_lambda_extended(polynomial_true_with_lambda_fv, polynomial_pred):

        if seed_in_inet_training:
            network_parameters = polynomial_true_with_lambda_fv[:,sparsity+1:]
            polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]
        else:
            network_parameters = polynomial_true_with_lambda_fv[:,sparsity:]
            polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]
            
        network_parameters = return_float_tensor_representation(network_parameters)
        polynomial_true = return_float_tensor_representation(polynomial_true)
        polynomial_pred = return_float_tensor_representation(polynomial_pred)
        
        assert polynomial_true.shape[1] == sparsity
        assert polynomial_pred.shape[1] == sparsity
        assert network_parameters.shape[1] == number_of_lambda_weights
        
        return tf.math.reduce_mean(tf.map_fn(calculate_r2_fv_lambda_wrapper(evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder), (polynomial_pred, network_parameters), fn_output_signature=tf.float32))
    return r2_tf_fv_lambda_extended

    def calculate_r2_fv_lambda_wrapper(evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder):

        def calculate_r2_fv_lambda(input_list):

            #single polynomials
            #polynomial_true = input_list[0]
            polynomial_pred = input_list[0]
            network_parameters = input_list[1]

            polynomial_pred_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))

            #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
            # build models
            start = 0
            layers = []
            for i in range(len(dims)//2):

                # set weights of layer
                index = i*2
                size = np.product(dims[index])
                weights_tf_true = tf.reshape(network_parameters[start:start+size], dims[index])
                model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
                start += size

                # set biases of layer
                index += 1
                size = np.product(dims[index])
                biases_tf_true = tf.reshape(network_parameters[start:start+size], dims[index])
                model_lambda_placeholder.layers[i].weights[1].assign(biases_tf_true)
                start += size


            lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(evaluation_dataset))

            return r2_keras_loss(lambda_fv, polynomial_pred_fv_list)

        return calculate_r2_fv_lambda



    #Manual TF Loss function for fv comparison of real and predicted polynomial

    def r2_tf_fv_poly_extended_wrapper(evaluation_dataset, list_of_monomial_identifiers):

        evaluation_dataset = return_float_tensor_representation(evaluation_dataset)
        list_of_monomial_identifiers = return_float_tensor_representation(list_of_monomial_identifiers)        

        @tf.function()
        def r2_tf_fv_poly_extended(polynomial_true, polynomial_pred):

            polynomial_true = return_float_tensor_representation(polynomial_true)
            polynomial_pred = return_float_tensor_representation(polynomial_pred)

            assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape)
            assert polynomial_pred.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_pred.shape)     

            return tf.math.reduce_mean(tf.map_fn(calculate_r2_fv_poly_wrapper(evaluation_dataset, list_of_monomial_identifiers), (polynomial_true, polynomial_pred), fn_output_signature=tf.float32))
        return r2_tf_fv_poly_extended

    def calculate_r2_fv_poly_wrapper(evaluation_dataset, list_of_monomial_identifiers):

        def calculate_r2_fv_poly(input_list):

            #single polynomials
            polynomial_true = input_list[0]
            polynomial_pred = input_list[1]

            polynomial_true_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_true), (evaluation_dataset))
            polynomial_pred_fv_list = tf.vectorized_map(calculate_fv_from_data_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))

            return r2_keras_loss(polynomial_true_fv_list, polynomial_pred_fv_list)

        return calculate_r2_fv_poly



    #calculate MAE at the end ---> general:REPLACE FUNCTION WITH LOSS CALL OR LAMBDA
    def calculate_r2_single_input(input_list):
        true_fv = input_list[0]
        pred_fv = input_list[1]

        return r2_keras(true_fv, pred_fv)




    #BASIC COEFFICIENT-BASED LOSS IF X_DATA IS APPENDED
    def r2_extended(polynomial_true_with_lambda_fv, polynomial_pred): 

        if seed_in_inet_training:
            assert polynomial_true_with_lambda_fv.shape[1] == sparsity+number_of_lambda_weights+1
        else:
            assert polynomial_true_with_lambda_fv.shape[1] == sparsity+number_of_lambda_weights

        polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]    

        assert polynomial_true.shape[1] == sparsity
        assert polynomial_pred.shape[1] == sparsity

        return r2_keras(polynomial_true, polynomial_pred)

def r2_keras(y_true, y_pred, epsilon=tf.keras.backend.epsilon()):

    y_true_cleared = tf.boolean_mask(y_true, tf.not_equal(return_float_tensor_representation(0), y_true))
    y_pred_cleared = tf.boolean_mask(y_pred, tf.not_equal(return_float_tensor_representation(0), y_true))

    epsilon = 1e-5
    SS_res =  K.sum(K.square(y_true_cleared - y_pred_cleared)) 
    SS_tot = K.sum(K.square(y_true_cleared - K.mean(y_true_cleared))) 
    return ( 1 - SS_res/(SS_tot + epsilon) )

def r2_keras_loss(y_true, y_pred, epsilon=tf.keras.backend.epsilon()):

    #y_true = tf.boolean_mask(y_true, tf.not_equal(return_float_tensor_representation(0), y_true))
    #y_pred = tf.boolean_mask(y_pred, tf.not_equal(return_float_tensor_representation(0), y_true))

    #epsilon = 1e-5
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return  - ( 1 - SS_res/(SS_tot + epsilon) )


#######################################################################################################################################################
######################################################Basic Keras/TF Loss functions####################################################################
#######################################################################################################################################################


def root_mean_squared_error(y_true, y_pred):   
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
        
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred)           
            
    return tf.math.sqrt(K.mean(K.square(y_pred - y_true))) 

def accuracy_multilabel(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))      
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return K.mean(tf.dtypes.cast(tf.dtypes.cast(tf.reduce_all(K.equal(y_true, y_pred), axis=1), tf.int32), tf.float32))#tf.reduce_all(K.equal(K.equal(y_true, y_pred), True), axis=1)#K.all(K.equal(y_true, y_pred)) #K.equal(y_true, y_pred)                        

def accuracy_single(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))
        
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return K.mean(tf.dtypes.cast(tf.dtypes.cast(K.equal(y_true, y_pred), tf.int32), tf.float32))#tf.reduce_all(K.equal(K.equal(y_true, y_pred), True), axis=1)#K.all(K.equal(y_true, y_pred)) #K.equal(y_true, y_pred)                        

def mean_absolute_percentage_error_keras(y_true, y_pred, epsilon=10e-3): 
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
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.mean(np.abs(true_values-pred_values)))
    
    return np.mean(np.array(result_list))  

def mean_absolute_error_function_values_return_multi_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.mean(np.abs(true_values-pred_values)))
    
    return np.array(result_list) 

def mean_std_function_values_difference(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)      
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.std(true_values-pred_values))
    
    return np.mean(np.array(result_list))  

def root_mean_squared_error_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)         
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.sqrt(np.mean((true_values-pred_values)**2)))
    
    return np.mean(np.array(result_list)) 

def mean_absolute_percentage_error_function_values(y_true, y_pred, epsilon=10e-3):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred) 
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.mean(np.abs(((true_values-pred_values)/(true_values+epsilon)))))

    return np.mean(np.array(result_list))

def r2_score_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(r2_score(true_values, pred_values))
    
    return np.mean(np.array(result_list))

def r2_score_function_values_return_multi_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(r2_score(true_values, pred_values))
    
    return np.array(result_list)

def relative_absolute_average_error_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    result_list = []
    
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.sum(np.abs(true_values-pred_values))/(true_values.shape[0]*np.std(true_values)))
    
    return np.mean(np.array(result_list))

def relative_maximum_average_error_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(np.max(true_values-pred_values)/np.std(true_values))
    
    return np.mean(np.array(result_list))

def mean_area_between_two_curves_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
      
    assert number_of_variables==1
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(area_between_two_curves(true_values, pred_values))
 
    return np.mean(np.array(result_list))

def mean_dtw_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)

    result_list_single = []
    result_list_array = []
    
    for true_values, pred_values in zip(y_true, y_pred):
        result_single_value, result_single_array = dtw(true_values, pred_values)
        result_list_single.append(result_single_value)
        result_list_array.append(result_single_array)
    
    return np.mean(np.array(result_list_single)), np.mean(np.array(result_list_array), axis=1)

def mean_frechet_dist_function_values(y_true, y_pred):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    result_list = []
    for true_values, pred_values in zip(y_true, y_pred):
        result_list.append(frechet_dist(true_values, pred_values))
    
    return np.mean(np.array(result_list))
