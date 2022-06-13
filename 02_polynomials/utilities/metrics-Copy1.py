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
        
    global list_of_monomial_identifiers
    from utilities.utility_functions import flatten, rec_gen, gen_monomial_identifier_list

    list_of_monomial_identifiers_extended = []

    if laurent:
        variable_sets = [list(flatten([[_d for _d in range(d+1)], [-_d for _d in range(1, neg_d+1)]])) for _ in range(n)]
        list_of_monomial_identifiers_extended = rec_gen(variable_sets)    

        if len(list_of_monomial_identifiers_extended) < 500:
            print(list_of_monomial_identifiers_extended)     

        list_of_monomial_identifiers = []
        for monomial_identifier in tqdm(list_of_monomial_identifiers_extended):
            if np.sum(monomial_identifier) <= d:
                if monomial_vars == None or len(list(filter(lambda x: x != 0, monomial_identifier))) <= monomial_vars:
                    list_of_monomial_identifiers.append(monomial_identifier)        
    else:
        variable_list = ['x'+ str(i) for i in range(n)]
        list_of_monomial_identifiers = gen_monomial_identifier_list(variable_list, d, n)
                    
        
#######################################################################################################################################################
######################Manual TF Loss function for comparison with lambda-net prediction based (predictions made in loss function)######################
#######################################################################################################################################################

#GENERAL LOSS UTILITY FUNCTIONS 
def calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial, current_monomial_degree, force_complete_poly_representation=False, config=None):

    if config != None:
        globals().update(config)

    #@tf.function(jit_compile=True)
    @tf.function(jit_compile=True)
    def calculate_poly_fv_tf(evaluation_entry):  
                
        def limit_monomial_to_degree_wrapper(monomial_degrees_by_variable, current_monomial_degree):
            #global current_monomial_degree
            current_monomial_degree.assign(0) #= tf.Variable(0, dtype=tf.int64)

            def limit_monomial_to_degree(index):
                #global current_monomial_degree   
                #nonlocal current_monomial_degree   
                additional_degree = tf.gather(monomial_degrees_by_variable, index)
                
                if tf.math.greater(current_monomial_degree + additional_degree, d):
                    adjusted_additional_degree = tf.cast(0, tf.int64)
                else:
                    adjusted_additional_degree = additional_degree
                #tf.print('current_monomial_degree', current_monomial_degree, summarize=-1)
                current_monomial_degree.assign(current_monomial_degree + adjusted_additional_degree)
                
                #tf.print('monomial_degrees_by_variable', monomial_degrees_by_variable, summarize=-1)
                #tf.print('index', index, summarize=-1)
                #tf.print('adjusted_additional_degree', adjusted_additional_degree, summarize=-1)
                
                return tf.stack([index, adjusted_additional_degree])


            return limit_monomial_to_degree            
        
        def calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry):
            def calculate_monomial_with_coefficient_degree_by_var(input_list):                     
                degree_by_var_per_monomial = input_list[0]
                coefficient = input_list[1]
                
                #degree_by_var_per_monomial = gewählter degree für jede variable in monomial
                monomial_value_without_coefficient = tf.math.reduce_prod(tf.vectorized_map(lambda x: x[0]**tf.dtypes.cast(x[1], tf.float32), (evaluation_entry, degree_by_var_per_monomial)))                
                return coefficient*monomial_value_without_coefficient
            return calculate_monomial_with_coefficient_degree_by_var
        
        
        if interpretation_net_output_monomials == None or force_complete_poly_representation:
            monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))      
            monomial_values = tf.vectorized_map(lambda x: x[0]*x[1], (monomials_without_coefficient, polynomial))
        else: 
            if sparse_poly_representation_version == 1:
                monomials_without_coefficient = tf.vectorized_map(calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry), (list_of_monomial_identifiers))              
                
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]

                assert index_array.shape[0] == interpretation_net_output_monomials*sparsity, 'Shape of Coefficient Indices : ' + str(index_array.shape)

                index_list = tf.split(index_array, interpretation_net_output_monomials)

                assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials, 'Shape of Coefficient Indices Split: ' + str(len(index_list))

                indices = tf.argmax(index_list, axis=1) 

                monomial_values = tf.vectorized_map(lambda x: tf.gather(monomials_without_coefficient, x[0])*x[1], (indices, coefficients)) 

            elif sparse_poly_representation_version == 2:
                coefficients = polynomial[:interpretation_net_output_monomials]
                index_array = polynomial[interpretation_net_output_monomials:]
                #tf.print('index_array.shape', index_array)
                
                assert index_array.shape[0] == interpretation_net_output_monomials*n*(d+1), 'Shape of Coefficient Indices : ' + str(index_array.shape)                  
                    
                index_list_by_monomial = tf.transpose(tf.split(index_array, interpretation_net_output_monomials))

                index_list_by_monomial_by_var = tf.split(index_list_by_monomial, n, axis=0)
                index_list_by_monomial_by_var_new = []
                for tensor in index_list_by_monomial_by_var:
                    index_list_by_monomial_by_var_new.append(tf.transpose(tensor))
                index_list_by_monomial_by_var = index_list_by_monomial_by_var_new   
                degree_by_var_per_monomial_list = tf.transpose(tf.argmax(index_list_by_monomial_by_var, axis=2))        

                maximum_by_var_per_monomial_list = tf.transpose(tf.reduce_max(index_list_by_monomial_by_var, axis=2))
                maximum_by_var_per_monomial_list_sort_index = tf.cast(tf.argsort(maximum_by_var_per_monomial_list, direction='DESCENDING'), tf.int64)


                #degree_by_var_per_monomial_list_adjusted_with_index = tf.map_fn(fn=lambda x: tf.map_fn(fn=limit_monomial_to_degree_wrapper(x[0], current_monomial_degree),  elems=x[1], fn_output_signature=tf.TensorSpec([2], dtype=tf.int64)),  elems=(degree_by_var_per_monomial_list, maximum_by_var_per_monomial_list_sort_index), fn_output_signature=tf.TensorSpec([maximum_by_var_per_monomial_list_sort_index.shape[1], 2], dtype=tf.int64))
                degree_by_var_per_monomial_list_adjusted_with_index = tf.vectorized_map(fn=lambda x: tf.vectorized_map(fn=limit_monomial_to_degree_wrapper(x[0], current_monomial_degree),  elems=x[1]),  elems=(degree_by_var_per_monomial_list, maximum_by_var_per_monomial_list_sort_index))

                #degree_by_var_per_monomial_list_adjusted = tf.map_fn(fn=lambda x: tf.keras.backend.flatten(tf.gather(x[:,1:], tf.argsort(tf.keras.backend.flatten(x[:,:1])))), elems=degree_by_var_per_monomial_list_adjusted_with_index, fn_output_signature=tf.TensorSpec([degree_by_var_per_monomial_list_adjusted_with_index.shape[1]], dtype=tf.int64))
                degree_by_var_per_monomial_list_adjusted = tf.vectorized_map(fn=lambda x: tf.keras.backend.flatten(tf.gather(x[:,1:], tf.argsort(tf.keras.backend.flatten(x[:,:1])))), elems=degree_by_var_per_monomial_list_adjusted_with_index)


                monomial_values = tf.vectorized_map(calculate_monomial_with_coefficient_degree_by_var_wrapper(evaluation_entry), (degree_by_var_per_monomial_list, coefficients))
                    
        
        polynomial_fv = tf.reduce_sum(monomial_values)         
        return polynomial_fv
    return calculate_poly_fv_tf

#calculate intermediate term (without coefficient multiplication)
def calculate_monomial_without_coefficient_tf_wrapper(evaluation_entry):
    def calculate_monomial_without_coefficient_tf(coefficient_multiplier_term):    
        monomial_without_coefficient_list = tf.vectorized_map(lambda x: x[0]**x[1], (evaluation_entry, coefficient_multiplier_term))
        monomial_without_coefficient = tf.math.reduce_prod(monomial_without_coefficient_list)
        return monomial_without_coefficient
    return calculate_monomial_without_coefficient_tf



def inet_lambda_fv_loss_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, current_monomial_degree, base_model, weights_structure, dims):        
        
            
    evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(evaluation_dataset), tf.float32)#return_float_tensor_representation(evaluation_dataset)
    list_of_monomial_identifiers = tf.dtypes.cast(tf.convert_to_tensor(list_of_monomial_identifiers), tf.float32)#return_float_tensor_representation(list_of_monomial_identifiers)    
    
    model_lambda_placeholder = base_model#keras.models.clone_model(base_model)  

    #weights_structure = base_model.get_weights()
    #dims = [np_arrays.shape for np_arrays in weights_structure]
    #@tf.function
    def inet_lambda_fv_loss(polynomial_true_with_lambda_fv, polynomial_pred): 

    
        def calculate_lambda_fv_error_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder):
                                        
            def calculate_lambda_fv_error(input_list):

                #single polynomials
                #polynomial_true = input_list[0]
                polynomial_pred = input_list[0]
                network_parameters = input_list[1]
                                
                polynomial_pred_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers, polynomial_pred, current_monomial_degree), (evaluation_dataset))
                
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
          
            
        network_parameters = tf.dtypes.cast(tf.convert_to_tensor(network_parameters), tf.float32)#return_float_tensor_representation(network_parameters)
        polynomial_true = tf.dtypes.cast(tf.convert_to_tensor(polynomial_true), tf.float32)#return_float_tensor_representation(polynomial_true)
        polynomial_pred = tf.dtypes.cast(tf.convert_to_tensor(polynomial_pred), tf.float32)#return_float_tensor_representation(polynomial_pred)
        
        assert network_parameters.shape[1] == number_of_lambda_weights, 'Shape of Network Parameters: ' + str(network_parameters.shape)            
        assert polynomial_true.shape[1] == sparsity, 'Shape of True Polynomial: ' + str(polynomial_true.shape)      
        assert polynomial_pred.shape[1] == interpretation_net_output_shape, 'Shape of Pred Polynomial: ' + str(polynomial_pred.shape)   
        
        loss_value = tf.math.reduce_mean(tf.map_fn(calculate_lambda_fv_error_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder), (polynomial_pred, network_parameters), fn_output_signature=tf.float32))
        #loss_value = tf.math.reduce_mean(tf.vectorized_map(calculate_lambda_fv_error_wrapper(loss, evaluation_dataset, list_of_monomial_identifiers, dims, model_lambda_placeholder), (polynomial_pred, network_parameters)))  #NOT WORKING VDECTORIZED     
        
        return loss_value
    
    inet_lambda_fv_loss.__name__ = loss + '_' + inet_lambda_fv_loss.__name__
    
    return inet_lambda_fv_loss



   


