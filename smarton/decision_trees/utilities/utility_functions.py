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

#######################################################################################################################################################
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_utility_functions_config_from_curent_notebook(config):
       
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
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=min(50, epochs//10), verbose=0, min_delta=0, mode='min') #epsilon
        callbacks.append(reduce_lr_loss)
    if 'early_stopping' in callback_string_list:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=min(50, epochs//10), min_delta=0, verbose=0, mode='min', restore_best_weights=True)
        callbacks.append(earlyStopping)        
    #if not multi_epoch_analysis and samples_list == None: 
        #callbacks.append(TQDMNotebookCallback())        
    return callbacks


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


#######################################################################################################################################################
###########################Manual calculations for comparison of polynomials based on function values (no TF!)#########################################
#######################################################################################################################################################


    
def generate_paths(path_type='interpretation_net'):
    
    noise_path = noise  
    RANDOM_SEED_path = RANDOM_SEED
    
    if path_type=='interpretation_net_no_noise':
        lambda_nets_total_path = 50000
        noise_path = 0
        RANDOM_SEED_path = 42
    
    paths_dict = {}
    
        
    training_string = '_sameX' if same_training_all_lambda_nets else '_diffX'
        
    laurent_str = '_laurent' if laurent else ''
    monomial_vars_str = '_monvars_' + str(monomial_vars) if monomial_vars != None else ''
    neg_d_str = '_negd_' + str(neg_d) + '_prob_' + str(neg_d_prob) if neg_d != None else ''

        
    dataset_description_string = ('_var_' + str(n) + 
                                  '_d_' + str(d) + 
                                   laurent_str + 
                                   monomial_vars_str + 
                                   neg_d_str + 
                                   '_spars_' + str(sample_sparsity) + 
                                   '_amin_' + str(a_min) + 
                                   '_amax_' + str(a_max) + 
                                   #'_xmin_' + str(x_min) + 
                                   #'_xmax_' + str(x_max) + 
                                   '_xdist_' + str(x_distrib) + 
                                   '_noise_' + str(noise_distrib) + '_' + str(noise_path))
        
        
    adjusted_dataset_string = ('bmin' + str(border_min) +
                                'bmax' + str(border_max) +
                                'lowd' + str(lower_degree_prob) +
                                'azero' + str(a_zero_prob) +
                                'arand' + str(a_random_prob))
        
        

    if path_type == 'data_creation' or path_type == 'lambda_net': #Data Generation
  
        path_identifier_polynomial_data = ('poly_' + str(polynomial_data_size) + 
                                           '_train_' + str(lambda_dataset_size) + 
                                           dataset_description_string + 
                                           adjusted_dataset_string +
                                           training_string)            

        paths_dict['path_identifier_polynomial_data'] = path_identifier_polynomial_data
    
    if path_type == 'lambda_net' or path_type == 'interpretation_net' or path_type == 'interpretation_net_no_noise': #Lambda-Net
        
        if path_type == 'lambda_net' or path_type == 'interpretation_net':
            lambda_nets_total_path = lambda_nets_total
        
        if fixed_seed_lambda_training and fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeedInit'
        elif fixed_seed_lambda_training and not fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeed'
        elif not fixed_seed_lambda_training and fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixInit'
        elif not fixed_seed_lambda_training and not fixed_initialization_lambda_training:            
            seed_init_string = '_NoFixSeedInit'

            
        early_stopping_string = '_ES' + str(early_stopping_min_delta_lambda) + '_' if early_stopping_lambda else ''
            
        lambda_layer_str = ''.join([str(neurons) + '-' for neurons in lambda_network_layers])
        lambda_net_identifier = '_' + lambda_layer_str + str(epochs_lambda) + 'e' + early_stopping_string + str(batch_lambda) + 'b' + '_' + optimizer_lambda + '_' + loss_lambda

        path_identifier_lambda_net_data = ('lnets_' + str(lambda_nets_total_path) +
                                           lambda_net_identifier + 
                                           '_train_' + str(lambda_dataset_size) + 
                                           training_string + 
                                           seed_init_string + '_' + str(RANDOM_SEED_path) +
                                           '/' +
                                           dataset_description_string[1:] + 
                                           adjusted_dataset_string)        

        paths_dict['path_identifier_lambda_net_data'] = path_identifier_lambda_net_data
    
    
    if path_type == 'interpretation_net' or path_type == 'interpretation_net_no_noise': #Interpretation-Net   
            
        interpretation_network_layers_string = 'dense' + ''.join([str(neurons) + '-' for neurons in dense_layers])
        
        if convolution_layers != None:
            interpretation_network_layers_string += 'conv' + str(convolution_layers)
        if lstm_layers != None:
            interpretation_network_layers_string += 'lstm' + str(lstm_layers)

        interpretation_net_identifier = '_' + interpretation_network_layers_string + 'output_' + str(interpretation_net_output_shape) + '_drop' + str(dropout) + 'e' + str(epochs) + 'b' + str(batch_size) + '_' + optimizer
        
        path_identifier_interpretation_net_data = ('inet' + interpretation_net_identifier +
                                                   '/lnets_' + str(interpretation_dataset_size) +
                                                   lambda_net_identifier + 
                                                   '_train_' + str(lambda_dataset_size) + 
                                                   training_string + 
                                                   seed_init_string + '_' + str(RANDOM_SEED_path) +
                                                   '/' +
                                                   dataset_description_string[1:] + 
                                                   adjusted_dataset_string)       
        
        
        paths_dict['path_identifier_interpretation_net_data'] = path_identifier_interpretation_net_data
        
    return paths_dict
    
def create_folders_inet():
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    try:
        # Create target Directory
        os.makedirs('./data/plotting/' + paths_dict['path_identifier_interpretation_net_data'] + '/')
        os.makedirs('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/')
    except FileExistsError:
        pass
    

def generate_directory_structure():
    
    directory_names = ['parameters', 'plotting', 'saved_polynomial_lists', 'results', 'saved_models', 'weights', 'weights_training']
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
        text_file = open('./data/.gitignore', 'w')
        text_file.write('*')
        text_file.close()  
        
    for directory_name in directory_names:
        path = './data/' + directory_name
        if not os.path.exists(path):
            os.makedirs(path)
            
            
def generate_lambda_net_directory():
    
    paths_dict = generate_paths(path_type = 'lambda_net')
    
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

