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
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


import tensorflow as tf
import keras
import random 

#udf import
from utilities.LambdaNet import *
#from utilities.metrics import *
from utilities.utility_functions import *

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

#GENERAL LOSS UTILITY FUNCTIONS
def calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial, force_complete_poly_representation=False, set_config=False, config=None):
    
    if set_config:
        #globals().update(dict(config))
        globals().update(config)

    def calculate_poly_fv_tf(evaluation_entry):        
        
        
        monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
        
        if interpretation_net_output_monomials == None or force_complete_poly_representation:
            monomial_values = tf.vectorized_map(lambda x: x[0]*x[1], (monomials_without_coefficient, polynomial))
        else: 
            coefficients = polynomial[:interpretation_net_output_monomials]
            index_array = polynomial[interpretation_net_output_monomials:]
            
            assert index_array.shape[0] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices : ' + str(index_array.shape)
            
            index_list = tf.split(index_array, interpretation_net_output_monomials)
            
            assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))
            
            indices = tf.argmax(index_list, axis=1) 

            monomial_values = tf.vectorized_map(lambda x: tf.gather(monomials_without_coefficient, x[0])*x[1], (indices, coefficients)) 
            
            
        polynomial_fv = tf.reduce_sum(monomial_values)     

        return polynomial_fv
    return calculate_poly_fv_tf

#calculate intermediate term (without coefficient multiplication)
def calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry):
    def calculate_monomial_without_coefficient_tf(coefficient_multiplier_term) :      

        return tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**x[1], (evaluation_entry, coefficient_multiplier_term)))
    return calculate_monomial_without_coefficient_tf



def inet_lambda_fv_loss_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, base_model, set_config=False, config=None, dims=None):        
        
        
    if set_config:
        globals().update(config)
        
    global nas
        
    evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(evaluation_dataset), tf.float32)#return_float_tensor_representation(evaluation_dataset)
    list_of_monomial_identifiers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers), tf.float32)#return_float_tensor_representation(list_of_monomial_identifiers)    
    
    model_lambda_placeholder = base_model#keras.models.clone_model(base_model)  
    if not set_config:
        weights_structure = base_model.get_weights()
        dims = [np_arrays.shape for np_arrays in weights_structure]
    
    def inet_lambda_fv_loss(polynomial_true_with_lambda_fv, polynomial_pred):     
        
        
        def calculate_lambda_fv_error_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder):

            def calculate_lambda_fv_error(input_list):

                #single polynomials
                #polynomial_true = input_list[0]
                polynomial_pred = input_list[0]
                network_parameters = input_list[1]
                
                polynomial_pred_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))
                
                #CALCULATE LAMBDA FV HERE FOR EVALUATION DATASET
                # build models
                start = 0
                layers = []
                if not set_config:
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
                else:
                    #tf.print(dims)
                    for i in range(dims.shape[0]//2):
                        # set weights of layer
                        
                        index = i*2
                        dim = dims[index]
                        if tf.math.equal(dim[1], 0):                            
                            size = tf.math.reduce_prod((dim[0],))
                            #tf.print(network_parameters[start:start+size].shape)
                            #tf.print((dim[0],))
                            weights_tf_true = tf.reshape(network_parameters[start:start+size], (dim[0],))
                            #tf.print(weights_tf_true)
                            
                            model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
                            start += size                        
                        else:
                            size = tf.math.reduce_prod((dim[0],dim[1]))
                            #tf.print(network_parameters[start:start+size].shape)
                            #tf.print((dim[0],dim[1]))
                            weights_tf_true = tf.reshape(network_parameters[start:start+size], (dim[0],dim[1]))
                            #tf.print(weights_tf_true)
                            model_lambda_placeholder.layers[i].weights[0].assign(weights_tf_true)
                            start += size                                
                            
                        # set biases of layer
                        index += 1
                        dim = dims[index]
                        if tf.math.equal(dim[1], 0):                            
                            size = tf.math.reduce_prod((dim[0],))
                            #tf.print(network_parameters[start:start+size].shape)
                            #tf.print((dim[0],))
                            weights_tf_true = tf.reshape(network_parameters[start:start+size], (dim[0],))
                            #tf.print(weights_tf_true)
                            
                            model_lambda_placeholder.layers[i].weights[1].assign(weights_tf_true)
                            start += size                        
                        else:
                            size = tf.math.reduce_prod((dim[0],dim[1]))
                            #tf.print(network_parameters[start:start+size].shape)
                            #tf.print((dim[0],dim[1]))
                            weights_tf_true = tf.reshape(network_parameters[start:start+size], (dim[0],dim[1]))
                            #tf.print(weights_tf_true)
                            model_lambda_placeholder.layers[i].weights[1].assign(weights_tf_true)
                            start += size                      


                lambda_fv = tf.keras.backend.flatten(model_lambda_placeholder(evaluation_dataset))
                #lambda_fv = tf.zeros_like(polynomial_pred_fv_list)
                
                ##### REMOVE NAN VALUES IN BOTH TENSORS #####
                #lambda_fv = tf.boolean_mask(lambda_fv, tf.math.logical_and(tf.math.logical_not(tf.math.is_nan(lambda_fv)), tf.math.logical_not(tf.math.is_nan(polynomial_pred_fv_list))))
                #polynomial_pred_fv_list = tf.boolean_mask(polynomial_pred_fv_list, tf.math.logical_and(tf.math.logical_not(tf.math.is_nan(polynomial_pred_fv_list)), tf.math.logical_not(tf.math.is_nan(lambda_fv))))
                
                
                error = None
                if loss == 'mae':
                    error = tf.keras.losses.MAE(lambda_fv, polynomial_pred_fv_list)
                elif loss == 'r2':
                    error = r2_keras_loss(lambda_fv, polynomial_pred_fv_list)  
                else:
                    error = tf.keras.losses.MAE(lambda_fv, polynomial_pred_fv_list)
                    #raise SystemExit('Unknown I-Net Metric: ' + loss)                

                error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)
                #error = np.inf*tf.cast(tf.ones_like(error), tf.float64)
                    
                return error#tf.math.reduce_mean(tf.vectorized_map(error_function, (lambda_fv, polynomial_pred_fv_list)))

            return calculate_lambda_fv_error            
        
        
        network_parameters = polynomial_true_with_lambda_fv[:,sparsity:] #sparsity here because true poly is always maximal, just prediction is reduced
        polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]
        if False:
            tf.print(network_parameters.shape)
            tf.print(polynomial_true.shape)
            tf.print(polynomial_pred.shape)
            tf.print(polynomial_true_with_lambda_fv.shape)
            tf.print(evaluation_dataset.shape)
            tf.print(list_of_monomial_identifiers.shape)            
            tf.print(network_parameters.dtype)
            tf.print(polynomial_true.dtype)
            tf.print(polynomial_pred.dtype)
            tf.print(polynomial_true_with_lambda_fv.dtype) 
            tf.print(evaluation_dataset.dtype)
            tf.print(list_of_monomial_identifiers.dtype)                
        if nas:
            if polynomial_pred.shape[1] != interpretation_net_output_shape:
                polynomial_pred = polynomial_pred[:,:interpretation_net_output_shape]
            
        network_parameters = tf.dtypes.cast(tf.convert_to_tensor(network_parameters), tf.float32)#return_float_tensor_representation(network_parameters)
        polynomial_true = tf.dtypes.cast(tf.convert_to_tensor(polynomial_true), tf.float32)#return_float_tensor_representation(polynomial_true)
        polynomial_pred = tf.dtypes.cast(tf.convert_to_tensor(polynomial_pred), tf.float32)#return_float_tensor_representation(polynomial_pred)
        
        assert network_parameters.shape[1] == number_of_lambda_weights, 'Shape of Network Parameters: ' + str(network_parameters.shape)            
        assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape)      
        assert polynomial_pred.shape[1] == interpretation_net_output_shape, 'Shape of Pred Polynomial: ' + str(polynomial_pred.shape)   
        
        return tf.math.reduce_mean(tf.map_fn(calculate_lambda_fv_error_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder), (polynomial_pred, network_parameters), fn_output_signature=tf.float32))
    
    if not set_config:
        inet_lambda_fv_loss.__name__ = loss + '_' + inet_lambda_fv_loss.__name__
    return inet_lambda_fv_loss



   

        
def inet_poly_fv_loss_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, base_model):   
    
    evaluation_dataset = return_float_tensor_representation(evaluation_dataset)
    list_of_monomial_identifiers = return_float_tensor_representation(list_of_monomial_identifiers)            
    
    def inet_poly_fv_loss(polynomial_true, polynomial_pred):
        
        def calculate_poly_fv_error_wrapper(evaluation_dataset, list_of_monomial_identifiers):

            def calculate_poly_fv_error(input_list):

                #single polynomials
                polynomial_true = input_list[0]
                polynomial_pred = input_list[1]

                polynomial_true_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial_true, force_complete_poly_representation=True), (evaluation_dataset))
                polynomial_pred_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial_pred), (evaluation_dataset))

                error = None
                if loss == 'mae':
                    error = tf.keras.losses.MAE(polynomial_true_fv_list, polynomial_pred_fv_list)
                elif loss == 'r2':
                    error = r2_keras_loss(polynomial_true_fv_list, polynomial_pred_fv_list)  
                else:
                    raise SystemExit('Unknown I-Net Metric: ' + loss)

                error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)
                    
                return error#tf.math.reduce_mean(tf.vectorized_map(calculate_mae_single_input, (polynomial_true_fv_list, polynomial_pred_fv_list)))

            return calculate_poly_fv_error     
        

        polynomial_true = return_float_tensor_representation(polynomial_true)
        polynomial_pred = return_float_tensor_representation(polynomial_pred)
        
        if nas:
            if polynomial_pred.shape[1] != interpretation_net_output_shape:
                polynomial_pred = polynomial_pred[:,:interpretation_net_output_shape]
            
        assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape)      
        assert polynomial_pred.shape[1] == interpretation_net_output_shape, 'Shape of Pred Polynomial: ' + str(polynomial_pred.shape) 
   

        return tf.math.reduce_mean(tf.map_fn(calculate_mae_fv_poly_wrapper(evaluation_dataset, list_of_monomial_identifiers), (polynomial_true, polynomial_pred), fn_output_signature=tf.float32))    
    
    inet_poly_fv_loss.__name__ = loss + '_' + inet_poly_fv_loss.__name__
    return inet_poly_fv_loss    





def inet_coefficient_loss_wrapper(loss):

    def inet_coefficient_loss(polynomial_true_with_lambda_fv, polynomial_pred): 

        assert polynomial_true_with_lambda_fv.shape[1] == sparsity+number_of_lambda_weights, 'Shape of Polynomial True with Lambda: ' + str(polynomial_true_with_lambda_fv.shape) 
        
        polynomial_true = polynomial_true_with_lambda_fv[:,:sparsity]    
        
        if nas:
            if polynomial_pred.shape[1] != interpretation_net_output_shape:
                polynomial_pred = polynomial_pred[:,:interpretation_net_output_shape] 

        assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape) 
        assert polynomial_pred.shape[1] == interpretation_net_output_shape, 'Shape of Pred Polynomial: ' + str(polynomial_pred.shape) 
        
        if interpretation_net_output_monomials != None:
            #get relevant indices and compare just coefficients for those indices with predicted coefficients --> no good metric, just for consistency included
            #TODO add 0 for all other indices?
            coefficient_indices_array = tf.map_fn(fn=lambda x: x[interpretation_net_output_monomials:], elems=polynomial_pred, fn_output_signature=tf.float32)
            polynomial_pred = tf.map_fn(fn=lambda x: x[:interpretation_net_output_monomials], elems=polynomial_pred, fn_output_signature=tf.float32)      
            
            assert coefficient_indices_array.shape[1] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 
            
            coefficient_indices_list = tf.split(coefficient_indices_array, interpretation_net_output_monomials, axis=1)
       
            assert len(coefficient_indices_list) == polynomial_pred.shape[1] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 
            
            coefficient_indices = tf.transpose(tf.argmax(coefficient_indices_list, axis=2))
                       
            polynomial_true = tf.map_fn(fn=lambda x: tf.gather(x[0], x[1]), elems=(polynomial_true, coefficient_indices), fn_output_signature=tf.float32)
            
        error = None
        if loss == 'mae':
            error = tf.keras.losses.MAE(polynomial_true, polynomial_pred)
        elif loss == 'r2':
            error = r2_keras_loss(polynomial_true, polynomial_pred)  
        else:
            raise SystemExit('Unknown I-Net Metric: ' + loss)
            
        error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)
                
        return error#tf.keras.losses.MAE(polynomial_true, polynomial_pred)

    inet_coefficient_loss.__name__ = loss + '_' + inet_coefficient_loss.__name__
    return inet_coefficient_loss   

def r2_keras_loss(y_true, y_pred, epsilon=tf.keras.backend.epsilon()):

    SS_res =  tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)) 
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))) 
    return  - ( 1 - SS_res/(SS_tot + epsilon) )


#######################################################################################################################################################
######################################################Basic Keras/TF Loss functions####################################################################
#######################################################################################################################################################


def root_mean_squared_error(y_true, y_pred):   
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
        
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred)           
            
    return tf.math.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))) 

def accuracy_multilabel(y_true, y_pred, a_step=0.1):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))      
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.reduce_all(tf.keras.backend.equal(y_true, y_pred), axis=1), tf.int32), tf.float32))        

def accuracy_single(y_true, y_pred, a_step=0.1):
    y_true = return_numpy_representation(y_true)
    y_pred = return_numpy_representation(y_pred)
    
    y_true =  return_float_tensor_representation(y_true)
    y_pred =  return_float_tensor_representation(y_pred) 
            
    n_digits = int(-np.log10(a_step))
        
    y_true = tf.math.round(y_true * 10**n_digits) / (10**n_digits) 
    y_pred = tf.math.round(y_pred * 10**n_digits) / (10**n_digits) 
        
    return tf.keras.backend.mean(tf.dtypes.cast(tf.dtypes.cast(tf.keras.backend.equal(y_true, y_pred), tf.int32), tf.float32))

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


#######################################################################################################################################################
#######################################################LAMBDA-NET EVALUATION FUNCTION##################################################################
#######################################################################################################################################################

def evaluate_lambda_net(identifier, 
                        X_data_real_lambda, 
                        y_data_real_lambda, 
                        y_data_pred_lambda, 
                        y_data_pred_lambda_poly_lstsq, 
                        y_data_real_lambda_poly_lstsq):
    
    mae_real_VS_predLambda = np.round(mean_absolute_error(y_data_real_lambda, y_data_pred_lambda), 4)
    mae_real_VS_predPolyLstsq = np.round(mean_absolute_error(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)
    mae_predLambda_VS_predPolyLstsq = np.round(mean_absolute_error(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)
    mae_real_VS_realPolyLstsq = np.round(mean_absolute_error(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)    

    rmse_real_VS_predLambda = np.round(root_mean_squared_error(y_data_real_lambda, y_data_pred_lambda), 4)    
    rmse_real_VS_predPolyLstsq = np.round(root_mean_squared_error(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)    
    rmse_predLambda_VS_predPolyLstsq = np.round(root_mean_squared_error(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)    
    rmse_real_VS_realPolyLstsq = np.round(root_mean_squared_error(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)    

    mape_real_VS_predLambda = np.round(mean_absolute_percentage_error_keras(y_data_real_lambda, y_data_pred_lambda), 4)    
    mape_real_VS_predPolyLstsq = np.round(mean_absolute_percentage_error_keras(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)    
    mape_predLambda_VS_predPolyLstsq = np.round(mean_absolute_percentage_error_keras(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)    
    mape_real_VS_realPolyLstsq = np.round(mean_absolute_percentage_error_keras(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)            

    r2_real_VS_predLambda = np.round(r2_score(y_data_real_lambda, y_data_pred_lambda), 4)
    r2_real_VS_predPolyLstsq = np.round(r2_score(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)
    r2_predLambda_VS_predPolyLstsq = np.round(r2_score(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)
    r2_real_VS_realPolyLstsq = np.round(r2_score(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)

    raae_real_VS_predLambda = np.round(relative_absolute_average_error(y_data_real_lambda, y_data_pred_lambda), 4)
    raae_real_VS_predPolyLstsq = np.round(relative_absolute_average_error(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)
    raae_predLambda_VS_predPolyLstsq = np.round(relative_absolute_average_error(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)
    raae_real_VS_realPolyLstsq = np.round(relative_absolute_average_error(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)

    rmae_real_VS_predLambda = np.round(relative_maximum_average_error(y_data_real_lambda, y_data_pred_lambda), 4)
    rmae_real_VS_predPolyLstsq = np.round(relative_maximum_average_error(y_data_real_lambda, y_data_pred_lambda_poly_lstsq), 4)
    rmae_predLambda_VS_predPolyLstsq = np.round(relative_maximum_average_error(y_data_pred_lambda_poly_lstsq, y_data_pred_lambda), 4)
    rmae_real_VS_realPolyLstsq = np.round(relative_maximum_average_error(y_data_real_lambda, y_data_real_lambda_poly_lstsq), 4)
        
    std_data_real_lambda = np.round(np.std(y_data_real_lambda), 4) 
    std_data_pred_lambda = np.round(np.std(y_data_pred_lambda), 4) 
    std_data_pred_lambda_poly_lstsq = np.round(np.std(y_data_pred_lambda_poly_lstsq), 4) 
    std_data_real_lambda_poly_lstsq = np.round(np.std(y_data_real_lambda_poly_lstsq), 4) 

    mean_data_real_lambda = np.round(np.mean(y_data_real_lambda), 4) 
    mean_data_pred_lambda = np.round(np.mean(y_data_pred_lambda), 4) 
    mean_data_pred_lambda_poly_lstsq = np.round(np.mean(y_data_pred_lambda_poly_lstsq), 4) 
    mean_data_real_lambda_poly_lstsq = np.round(np.mean(y_data_real_lambda_poly_lstsq), 4) 

    return [{
             'MAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': mae_real_VS_predLambda,
             'MAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': mae_real_VS_predPolyLstsq,
             'MAE FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': mae_predLambda_VS_predPolyLstsq,
             'MAE FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': mae_real_VS_realPolyLstsq,
             'RMSE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': rmse_real_VS_predLambda,
             'RMSE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': rmse_real_VS_predPolyLstsq,
             'RMSE FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': rmse_predLambda_VS_predPolyLstsq,
             'RMSE FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': rmse_real_VS_realPolyLstsq,
             'MAPE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': mape_real_VS_predLambda,
             'MAPE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': mape_real_VS_predPolyLstsq,
             'MAPE FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': mape_predLambda_VS_predPolyLstsq,
             'MAPE FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': mape_real_VS_realPolyLstsq,
             'R2 FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': r2_real_VS_predLambda,
             'R2 FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': r2_real_VS_predPolyLstsq,
             'R2 FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': r2_predLambda_VS_predPolyLstsq,
             'R2 FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': r2_real_VS_realPolyLstsq,
             'RAAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': raae_real_VS_predLambda,
             'RAAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': raae_real_VS_predPolyLstsq,
             'RAAE FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': raae_predLambda_VS_predPolyLstsq,
             'RAAE FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': raae_real_VS_realPolyLstsq,
             'RMAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA': rmae_real_VS_predLambda,
             'RMAE FV ' + identifier + ' REAL LAMBDA VS PRED LAMBDA POLY LSTSQ': rmae_real_VS_predPolyLstsq,
             'RMAE FV ' + identifier + ' PRED LAMBDA VS PRED LAMBDA POLY LSTSQ': rmae_predLambda_VS_predPolyLstsq,
             'RMAE FV ' + identifier + ' REAL LAMBDA VS REAL LAMBDA POLY LSTSQ': rmae_real_VS_realPolyLstsq,
            },
            {
             'STD FV ' + identifier + ' REAL LAMBDA': std_data_real_lambda,
             'STD FV ' + identifier + ' PRED LAMBDA': std_data_pred_lambda, 
             'STD FV ' + identifier + ' PRED LAMBDA POLY LSTSQ': std_data_pred_lambda_poly_lstsq, 
             'STD FV ' + identifier + ' REAL LAMBDA POLY LSTSQ': std_data_real_lambda_poly_lstsq, 
            },
            {
             'MEAN FV ' + identifier + ' REAL LAMBDA': mean_data_real_lambda,
             'MEAN FV ' + identifier + ' PRED LAMBDA': mean_data_pred_lambda, 
             'MEAN FV ' + identifier + ' PRED LAMBDA POLY LSTSQ': mean_data_pred_lambda_poly_lstsq, 
             'MEAN FV ' + identifier + ' REAL LAMBDA POLY LSTSQ': mean_data_real_lambda_poly_lstsq, 
            }]
    
        


#######################################################################################################################################################
#######################################################LAMBDA-NET METRICS##################################################################
#######################################################################################################################################################

def calcualate_function_value_with_X_data_entry(coefficient_list, X_data_entry):
    
    global list_of_monomial_identifiers
     
    result = 0    
    for coefficient_value, coefficient_multipliers in zip(coefficient_list, list_of_monomial_identifiers):
        partial_results = [X_data_value**int(coefficient_multiplier) for coefficient_multiplier, X_data_value in zip(coefficient_multipliers, X_data_entry)]
        
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



def evaluate_interpretation_net(y_data_real, 
                                y_data_pred, 
                                polynomial_true_fv, 
                                polynomial_pred_inet_fv):
    
    if type(y_data_real) != type(None) and type(y_data_pred) != type(None):
        y_data_real = return_numpy_representation(y_data_real)
        y_data_pred = return_numpy_representation(y_data_pred)     
        
        assert y_data_real.shape[1] == sparsity or y_data_real.shape[1] == interpretation_net_output_shape, 'Polynomial Real not in shape ' + str(y_data_real.shape)
        assert y_data_pred.shape[1] == sparsity or y_data_pred.shape[1] == interpretation_net_output_shape, 'Polynomial Pred not in shape ' + str(y_data_pred.shape)
       
        
        if y_data_real.shape[1] != y_data_pred.shape[1]:
            
            if y_data_real.shape[1] == interpretation_net_output_shape:
            
                coefficient_indices_array = y_data_real[:,interpretation_net_output_monomials:]
                y_data_real_reduced = y_data_real[:,:interpretation_net_output_monomials]

                assert coefficient_indices_array.shape[1] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials, axis=1)

                assert len(coefficient_indices_list) == y_data_real_reduced.shape[1] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))

                y_data_pred_reduced = []
                for y_data_pred_entry, coefficient_indices_entry in zip(y_data_pred, coefficient_indices):
                    y_data_pred_reduced.append(y_data_pred_entry[[coefficient_indices_entry]])
                y_data_pred_reduced = np.array(y_data_pred_reduced)

                y_data_real = y_data_real_reduced
                y_data_pred = y_data_pred_reduced
            
            elif y_data_pred.shape[1] == interpretation_net_output_shape:
                
                coefficient_indices_array = y_data_pred[:,interpretation_net_output_monomials:]
                y_data_pred_reduced = y_data_pred[:,:interpretation_net_output_monomials]

                assert coefficient_indices_array.shape[1] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices: ' + str(coefficient_indices_array.shape) 

                coefficient_indices_list = np.split(coefficient_indices_array, interpretation_net_output_monomials, axis=1)

                assert len(coefficient_indices_list) == y_data_pred_reduced.shape[1] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(coefficient_indices_list)) 

                coefficient_indices = np.transpose(np.argmax(coefficient_indices_list, axis=2))

                y_data_real_reduced = []
                for y_data_real_entry, coefficient_indices_entry in zip(y_data_real, coefficient_indices):
                    y_data_real_reduced.append(y_data_real_entry[[coefficient_indices_entry]])
                y_data_real_reduced = np.array(y_data_real_reduced)

                y_data_pred = y_data_pred_reduced
                y_data_real = y_data_real_reduced            
            
            else:
                raise SystemExit('Shapes Inconsistent') 
                
                
        
        mae_coeff = np.round(mean_absolute_error(y_data_real, y_data_pred), 4)
        rmse_coeff = np.round(root_mean_squared_error(y_data_real, y_data_pred), 4)
        mape_coeff = np.round(mean_absolute_percentage_error_keras(y_data_real, y_data_pred), 4)
        accuracy_coeff = np.round(accuracy_single(y_data_real, y_data_pred), 4)
        accuracy_multi_coeff = np.round(accuracy_multilabel(y_data_real, y_data_pred), 4)
    else:
        mae_coeff = np.nan
        rmse_coeff = np.nan
        mape_coeff = np.nan
        accuracy_coeff = np.nan
        accuracy_multi_coeff = np.nan
        
    polynomial_true_fv = return_numpy_representation(polynomial_true_fv)
    polynomial_pred_inet_fv = return_numpy_representation(polynomial_pred_inet_fv)
    
    assert polynomial_true_fv.shape == polynomial_pred_inet_fv.shape, 'Shape of True Polynomial FVs: ' + str(polynomial_true_fv.shape) 
        
    mae_fv = np.round(mean_absolute_error_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    rmse_fv = np.round(root_mean_squared_error_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    mape_fv = np.round(mean_absolute_percentage_error_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    r2_fv = np.round(r2_score_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    raae_fv = np.round(relative_absolute_average_error_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    rmae_fv = np.round(relative_maximum_average_error_function_values(polynomial_true_fv, polynomial_pred_inet_fv), 4) 
    
    std_fv_diff = np.round(mean_std_function_values_difference(polynomial_true_fv, polynomial_pred_inet_fv), 4)
    mean_fv_1 = np.mean(polynomial_true_fv)
    mean_fv_2 = np.mean(polynomial_pred_inet_fv)
    std_fv_1 = np.std(polynomial_true_fv)
    std_fv_2 = np.std(polynomial_pred_inet_fv)

    mae_distribution = mean_absolute_error_function_values_return_multi_values(polynomial_true_fv, polynomial_pred_inet_fv)
    r2_distribution = r2_score_function_values_return_multi_values(polynomial_true_fv, polynomial_pred_inet_fv)

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
                                                  index=['L-' + str(i) for i in range(polynomial_true_fv.shape[0])]),
                                        'R2': pd.Series(data=r2_distribution, 
                                                  index=['L-' + str(i) for i in range(polynomial_true_fv.shape[0])])}