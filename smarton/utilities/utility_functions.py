#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

#from itertools import product       # forms cartesian products
#from tqdm import tqdm_notebook as tqdm
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
#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw
import time

import tensorflow as tf
import random 

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from IPython.display import display, Math, Latex, clear_output

import os
import shutil
import pickle
    
#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
#from utilities.utility_functions import *

from scipy.optimize import minimize
import sympy as sym
from sympy import Symbol, sympify, lambdify


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
        
    global list_of_monomial_identifiers
        
    list_of_monomial_identifiers_extended = []
    for i in range((d+1)**n):    
        monomial_identifier = dec_to_base(i, base = (d+1)).zfill(n) 
        list_of_monomial_identifiers_extended.append(monomial_identifier)


    list_of_monomial_identifiers = []
    for monomial_identifier in list_of_monomial_identifiers_extended:
        monomial_identifier_values = list(map(int, list(monomial_identifier)))
        if sum(monomial_identifier_values) <= d:
            list_of_monomial_identifiers.append(monomial_identifier)
                                    
#######################################################################################################################################################
#############################################################General Utility Functions#################################################################
#######################################################################################################################################################

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

def encode (n):
    try:
        return ALPHABET [n]
    except IndexError:
        raise Exception ("cannot encode: %s" % n)
        
def dec_to_base (dec = 0, base = 16):
    if dec < base:
        return encode (dec)
    else:
        return dec_to_base (dec // base, base) + encode (dec % base)

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


def return_numpy_representation(some_representation):
    if isinstance(some_representation, pd.DataFrame):
        some_representation = some_representation.values
        
    if isinstance(some_representation, list):
        some_representation = np.array(some_representation)
    
    if not isinstance(some_representation, np.ndarray):
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
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=epochs/10, verbose=0, min_delta=0, mode='min') #epsilon
        callbacks.append(reduce_lr_loss)
    if 'early_stopping' in callback_string_list:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0, verbose=0, mode='min', restore_best_weights=True)
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
            
            
def print_polynomial_from_coefficients(coefficient_array, force_complete_poly_representation=False, round_digits=None):
    return display(get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=force_complete_poly_representation, round_digits=round_digits))


def get_polynomial_string_from_coefficients(coefficients, force_complete_poly_representation=False, round_digits=None):

    global list_of_monomial_identifiers
    global interpretation_net_output_monomials
        
    string = ''
    
    try: #catch if this is lambda-net training
        interpretation_net_output_monomials == None
    except NameError:
        interpretation_net_output_monomials = None
        
    if interpretation_net_output_monomials == None or force_complete_poly_representation:
        for identifier, coefficient in zip(list_of_monomial_identifiers, coefficients):
            if round_digits != None:
                string += str(np.round(coefficient, round_digits))
            else:
                string += str(coefficient)
            for index, variable_identifier in enumerate(identifier):  
                if int(variable_identifier) == 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[index]
                elif int(variable_identifier) > 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[index] + '^' + str(variable_identifier)
            string += ' + '
    else:
        # Convert output array to monomial identifier index and corresponding coefficient
        assert coefficient_array.shape[0] == interpretation_net_output_shape
        
        coefficients = coefficient_array[:interpretation_net_output_monomials]
        index_array = coefficient_array[interpretation_net_output_monomials:]
        
        
        assert index_array.shape[0] == interpretation_net_output_monomials*sparsity
        index_list = np.split(index_array, interpretation_net_output_monomials)
        
        assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
        indices = np.argmax(index_list, axis=1)        
        
        
        for monomial_index, monomial_coefficient in zip(indices, coefficients):
            if round_digits != None:
                string += str(np.round(monomial_coefficient, round_digits))
            else:
                string += str(monomial_coefficient)
            #REPLACE NAN            
            for i, monomial_exponent in enumerate(list_of_monomial_identifiers[monomial_index]):
                if int(monomial_exponent) == 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[i]
                elif int(monomial_exponent) > 1:
                    #string += '*'
                    string += 'abcdefghijklmnopqrstuvwxyz'[i] + '^' + str(monomial_exponent)                  
            string += ' + '   
            
    return string[:-3]

def get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=False, round_digits=None):
    
    global list_of_monomial_identifiers
    global interpretation_net_output_monomials
    
    variable_alphabet =  "abcdefghijklmnopqrstuvwxyz"
    
    variable_list = []
    for i in range(n):
        variable_list.append(sym.symbols(variable_alphabet[i]))    
    
    try: #catch if this is lambda-net training
        interpretation_net_output_monomials == None
    except NameError:
        interpretation_net_output_monomials = None
    
    if interpretation_net_output_monomials == None or force_complete_poly_representation:   
        f = 0
        for monomial_identifier, monomial_coefficient in zip(list_of_monomial_identifiers, coefficient_array):
            if round_digits != None:
                subfunction = np.round(monomial_coefficient, round_digits)
            else:
                subfunction = monomial_coefficient        
            for i, monomial_exponent in enumerate(monomial_identifier):
                subfunction *= variable_list[i]**float(monomial_exponent)
            f += subfunction
    else:
        f = 0
        
        # Convert output array to monomial identifier index and corresponding coefficient
        assert coefficient_array.shape[0] == interpretation_net_output_shape
        
        coefficients = coefficient_array[:interpretation_net_output_monomials]
        index_array = coefficient_array[interpretation_net_output_monomials:]
        
        assert index_array.shape[0] == interpretation_net_output_monomials*sparsity
        index_list = np.split(index_array, interpretation_net_output_monomials)
        
        assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
        indices = np.argmax(index_list, axis=1)
        
        for monomial_index, monomial_coefficient in zip(indices, coefficients):
            if round_digits != None:
                subfunction = np.round(monomial_coefficient, round_digits)
            else:
                subfunction = monomial_coefficient
                #REPLACE NAN
            for i, monomial_exponent in enumerate(list_of_monomial_identifiers[monomial_index]):
                subfunction *= variable_list[i]**float(monomial_exponent)
            f += subfunction    
    
    return f


def plot_polynomial_from_coefficients(coefficient_array, force_complete_poly_representation=False):
    
    sympy_function_string = get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=False)
    
    variable_alphabet =  "abcdefghijklmnopqrstuvwxyz"
    
    variable_list = []
    for i in range(n):
        variable_list.append(sym.symbols(variable_alphabet[i]))       
    
    lam_x = lambdify(variable_list, sympy_function_string, modules=['numpy'])
    
    x_vals = linspace(x_min, x_max, 100)
    y_vals = lam_x(x_vals)

    plt.plot(x_vals, y_vals)
    plt.show()
    
                
def get_critical_points_from_polynomial(coefficient_array, force_complete_poly_representation=False): 
    
    
    coefficient_array = return_numpy_representation(coefficient_array)
    
    #assert coefficient_array.shape[0] == interpretation_net_output_shape
        
    f = get_sympy_string_from_coefficients(coefficient_array, force_complete_poly_representation=force_complete_poly_representation)
        
    gradient = sym.derive_by_array(f, tuple(f.free_symbols))
        
    stationary_points = sym.solve(gradient, tuple(f.free_symbols))
    
    
    return f, gradient, stationary_points




#######################################################################################################################################################
###########################Manual calculations for comparison of polynomials based on function values (no TF!)#########################################
#######################################################################################################################################################

def calcualate_function_value(coefficient_list, lambda_input_entry, force_complete_poly_representation=False):
    
    global list_of_monomial_identifiers
    global interpretation_net_output_monomials
    
    result = 0   
        
    try: #catch if this is lambda-net training
        interpretation_net_output_monomials == None
    except NameError:
        interpretation_net_output_monomials = None
        
        
    if interpretation_net_output_monomials == None or force_complete_poly_representation:
        
        #print(force_complete_poly_representation)
        #print(interpretation_net_output_monomials)
    
        assert coefficient_list.shape[0] == sparsity, 'Shape of Coefficient List: ' + str(coefficient_list.shape) + str(interpretation_net_output_monomials)
        
        for coefficient_value, coefficient_multipliers in zip(coefficient_list, list_of_monomial_identifiers):
            value_without_coefficient = [lambda_input_value**int(coefficient_multiplier) for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]

            try:
                result += coefficient_value * reduce(lambda x, y: x*y, value_without_coefficient)
            except TypeError:
                print('ERROR')
                print(lambda_input_entry)
                print(coefficient_list)

                print(coefficient_value)
                print(value_without_coefficient)
    else:
        
        # Convert output array to monomial identifier index and corresponding coefficient
        assert coefficient_list.shape[0] == interpretation_net_output_shape
        
        coefficients = coefficient_list[:interpretation_net_output_monomials]
        index_array = coefficient_list[interpretation_net_output_monomials:]
        
        assert index_array.shape[0] == interpretation_net_output_monomials*sparsity
        index_list = np.split(index_array, interpretation_net_output_monomials)
        
        assert len(index_list) == coefficients.shape[0] == interpretation_net_output_monomials
        indices = np.argmax(index_list, axis=1)

        # Calculate monomial values without coefficient
        value_without_coefficient_list = []
        for coefficient_multipliers in list_of_monomial_identifiers:
            value_without_coefficient = [lambda_input_value**int(coefficient_multiplier) for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]
            value_without_coefficient_list.append(reduce(lambda x, y: x*y, value_without_coefficient))
        value_without_coefficient_by_indices = np.array(value_without_coefficient_list)[[indices]]

        # Select relevant monomial values without coefficient and calculate final polynomial
        for coefficient, monomial_index in zip(coefficients, indices):
            #TODOOOOO
            result += coefficient * value_without_coefficient_list[monomial_index]
        
    return result

def calculate_function_values_from_polynomial(polynomial, lambda_input_data, force_complete_poly_representation=False):        
    function_value_list = []
                
    for lambda_input_entry in lambda_input_data:
        function_value = calcualate_function_value(polynomial, lambda_input_entry, force_complete_poly_representation=force_complete_poly_representation)
        function_value_list.append(function_value)
        
    return np.array(function_value_list)


def parallel_fv_calculation_from_polynomial(polynomial_list, lambda_input_list, force_complete_poly_representation=False):
    
    polynomial_list = return_numpy_representation(polynomial_list)
    lambda_input_list = return_numpy_representation(lambda_input_list)
    
    assert polynomial_list.shape[0] == lambda_input_list.shape[0]
        
    if force_complete_poly_representation:
        assert polynomial_list.shape[1] == sparsity
    else:
        assert polynomial_list.shape[1] == interpretation_net_output_shape
    assert lambda_input_list.shape[2] == n
    
    n_jobs_parallel_fv = 10 if polynomial_list.shape[0] > 10 else polynomial_list.shape[0]
    
    parallel = Parallel(n_jobs=n_jobs_parallel_fv, verbose=0, backend='threading')
    polynomial_true_fv = parallel(delayed(calculate_function_values_from_polynomial)(polynomial, lambda_inputs, force_complete_poly_representation=force_complete_poly_representation) for polynomial, lambda_inputs in zip(polynomial_list, lambda_input_list))  
    del parallel   
    

    return np.array(polynomial_true_fv)

def sleep_minutes(minutes):
    time.sleep(int(60*minutes))
    
def sleep_hours(hours):
    time.sleep(int(60*60*hours))
    
    
def generate_paths(path_type='interpretation_net'):
    
    paths_dict = {}
    
    
      
    if same_training_all_lambda_nets:
        training_string = '_sameX'
    else:
        training_string = '_diffX'
        

    if path_type == 'data_creation' or path_type == 'lambda_net': #Data Generation
  
        path_identifier_polynomial_data = ('polynomials_' + str(polynomial_data_size) + 
                                           '_train_' + str(lambda_dataset_size) + 
                                           '_variables_' + str(n) + 
                                           '_degree_' + str(d) + 
                                           '_sparsity_' + str(sample_sparsity) + 
                                           '_amin_' + str(a_min) + 
                                           '_amax_' + str(a_max) + 
                                           '_xmin_' + str(x_min) + 
                                           '_xmax_' + str(x_max) + 
                                           '_xdistrib_' + str(x_distrib) + 
                                           '_noise_' + str(noise_distrib) + '_' + str(noise) + 
                                           training_string)            

        paths_dict['path_identifier_polynomial_data'] = path_identifier_polynomial_data
    
    if path_type == 'lambda_net' or path_type == 'interpretation_net': #Lambda-Net
        if fixed_seed_lambda_training and fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeedInit'
        elif fixed_seed_lambda_training and not fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixSeed'
        elif not fixed_seed_lambda_training and fixed_initialization_lambda_training:
            seed_init_string = '_' + str(number_different_lambda_trainings) + '-FixInit'
        elif not fixed_seed_lambda_training and not fixed_initialization_lambda_training:            
            seed_init_string = '_NoFixSeedInit'

        early_stopping_string = ''
        if early_stopping_lambda:
            early_stopping_string = '_ES' + str(early_stopping_min_delta_lambda) + '_'

            
        lambda_layer_str = ''.join([str(neurons) + '-' for neurons in lambda_network_layers])
        lambda_net_identifier = '_' + lambda_layer_str + str(epochs_lambda) + 'e' + early_stopping_string + str(batch_lambda) + 'b' + '_' + optimizer_lambda + '_' + loss_lambda

        path_identifier_lambda_net_data = ('lnets_' + str(lambda_nets_total) +
                                           lambda_net_identifier + 
                                           '_train_' + str(lambda_dataset_size) + 
                                           training_string + 
                                           seed_init_string + '_' + str(RANDOM_SEED) +
                                           '/var_' + str(n) + 
                                           '_d_' + str(d) + 
                                           '_sparsity_' + str(sample_sparsity) + 
                                           '_amin_' + str(a_min) + 
                                           '_amax_' + str(a_max) + 
                                           '_xmin_' + str(x_min) + 
                                           '_xmax_' + str(x_max) + 
                                           '_xdist_' + str(x_distrib) + 
                                           '_noise_' + str(noise_distrib) + '_' + str(noise))        

        paths_dict['path_identifier_lambda_net_data'] = path_identifier_lambda_net_data
    
    
    if path_type == 'interpretation_net': #Interpretation-Net   
            
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
                                                   seed_init_string + '_' + str(RANDOM_SEED) +
                                                   '/var_' + str(n) + 
                                                   '_d_' + str(d) + 
                                                   '_sparsity_' + str(sample_sparsity) + 
                                                   '_amin_' + str(a_min) + 
                                                   '_amax_' + str(a_max) + 
                                                   #'_xmin_' + str(x_min) + 
                                                   #'_xmax_' + str(x_max) + 
                                                   '_xdist_' + str(x_distrib) + 
                                                   '_noise_' + str(noise_distrib) + '_' + str(noise))       
        
        
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
################################# code adjusted, originally from: https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Random%20Function%20Generator ##############################################
######################################################################################################################################################################################################################

def symbolize(s):
    """
    Converts a a string (equation) to a SymPy symbol object
    """
        
    s1=s.replace(',','.')
    s2=s1.replace('^','**')
    s3=sympify(s2)
    
    return(s3)

def eval_multinomial(s,vals=None,symbolic_eval=False):
    """
    Evaluates polynomial at vals.
    vals can be simple list, dictionary, or tuple of values.
    vals can also contain symbols instead of real values provided those symbols have been declared before using SymPy
    """
    sym_s=symbolize(s)
    sym_set=sym_s.atoms(Symbol)
    sym_lst=[]

    
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval==False and len(sym_set)!=len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals)==list:
            sub=list(zip(sym_lst,vals))
        elif type(vals)==dict:
            l=list(vals.keys())
            l.sort()
            lst=[]
            for i in l:
                lst.append(vals[i])
            sub=list(zip(sym_lst,lst))
        elif type(vals)==tuple:
            sub=list(zip(sym_lst,list(vals)))
        result=sym_s.subs(sub)
    
    return result

def flip(y,p):
    lst=[]
    for i in range(len(y)):
        f=np.random.choice([1,0],p=[p,1-p])
        lst.append(f)
    lst=np.array(lst)
    return np.array(np.logical_xor(y,lst),dtype=int)


def gen_regression_symbolic(polynomial_array=None,n_samples=100,noise=0.0, noise_dist='normal', seed=42, sympy_calculation=True):
    """
    Generates regression sample based on a symbolic expression. Calculates the output of the symbolic expression 
    at randomly generated (drawn from a Gaussian distribution) points
    m: The symbolic expression. Needs x1, x2, etc as variables and regular python arithmatic symbols to be used.
    n_samples: Number of samples to be generated
    n_features: Number of variables. This is automatically inferred from the symbolic expression. So this is ignored 
                in case a symbolic expression is supplied. However if no symbolic expression is supplied then a 
                default simple polynomial can be invoked to generate regression samples with n_features.
    noise: Magnitude of Gaussian noise to be introduced (added to the output).
    noise_dist: Type of the probability distribution of the noise signal. 
    Currently supports: Normal, Uniform, t, Beta, Gamma, Poission, Laplace

    Returns a numpy ndarray with dimension (n_samples,n_features+1). Last column is the response vector.
    """
        
    np.random.seed(seed)
                
    if polynomial_array is not None:
        sympy_string = get_sympy_string_from_coefficients(polynomial_array)
        sympy_function=sympify(sympy_string)
        
    if polynomial_array is None:
        sympy_function=''
        for i in range(1,n_features+1):
            c='x'+str(i)
            c+=np.random.choice(['+','-'],p=[0.5,0.5])
            sympy_function+=c
        sympy_function=sympy_function[:-1]
        
    n_features=len(sympy_function.atoms(Symbol))
        
    eval_results=[]
    
    eval_dataset = generate_random_data_points(low=x_min, high=x_max, size=n_samples, variables=max(1, n), distrib=x_distrib)
             
    if sympy_calculation:
        for i in range(n_samples):
            eval_results.append(eval_multinomial(sympy_string, vals=list(eval_dataset[i])))
    elif not sympy_calculation and polynomial_array is not None:
        eval_results = calculate_function_values_from_polynomial(polynomial_array, eval_dataset)
        
    eval_results=np.array(eval_results)
    eval_results=eval_results.reshape(n_samples,1)
    
    if noise_dist=='normal':
        noise_sample=noise*np.random.normal(loc=0, scale=1.0,size=n_samples)
    elif noise_dist=='uniform':
        noise_sample=noise*np.random.uniform(low=0, high=1.0,size=n_samples)
        
    noise_sample=noise_sample.reshape(n_samples,1)
    
    eval_results=eval_results+noise_sample
        
    
    return polynomial_array, eval_dataset, eval_results


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
    
    model_lambda_placeholder = keras.models.clone_model(base_model)  
    
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
        
    
    best_result = np.inf

    for current_iteration in range(restarts):
                
        @tf.function(experimental_compile=True) 
        def function_to_optimize():
            
            poly_optimize = poly_optimize_input[0]

            if interpretation_net_output_monomials != None:
                poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]

                poly_optimize_identifiers_list = []
                for i in range(interpretation_net_output_monomials):
                    poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                    poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

            poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, config=config), (random_lambda_input_data))

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
    
    model_lambda_placeholder = keras.models.clone_model(base_model)  
    
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
        
    
    best_result = np.inf

    for current_iteration in range(restarts):
                        
        @tf.function(experimental_compile=True) 
        def function_to_optimize_scipy(poly_optimize_input):    

            
            #poly_optimize = tf.cast(tf.constant(poly_optimize_input), tf.float32)
            poly_optimize = tf.cast(poly_optimize_input, tf.float32)


            if interpretation_net_output_monomials != None:
                poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]

                poly_optimize_identifiers_list = []
                for i in range(interpretation_net_output_monomials):
                    poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                    poly_optimize_identifiers_list.append(poly_optimize_identifiers)
                poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
                poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)

            poly_optimize_fv_list = tf.vectorized_map(calculate_poly_fv_tf_wrapper(list_of_monomial_identifiers_numbers, poly_optimize, config=config), (random_lambda_input_data))


            error = None
            if inet_loss == 'mae':
                error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
            elif inet_loss == 'r2':
                error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
            else:
                raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

            error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)   
    
            

            return error
    
                    
        poly_optimize_input = tf.random.uniform([1, interpretation_net_output_shape])    

        
        stop_counter = 0
        
        opt_res = minimize(function_to_optimize_scipy, poly_optimize_input, method=optimizer, options={'maxfun': None, 'maxiter': max_steps})

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



def per_network_poly_optimization_slow(per_network_dataset_size, 
                                  lambda_network_weights, 
                                  #poly_representation, 
                                  list_of_monomial_identifiers_numbers, 
                                  config, 
                                  lr=0.05, 
                                  max_steps = 1000, 
                                  early_stopping=10, 
                                  restarts=5, 
                                  printing=True):
    
    def function_to_optimize():       
            
        poly_optimize = poly_optimize_input[0]

        
        if interpretation_net_output_monomials != None:
            poly_optimize_coeffs = poly_optimize[:interpretation_net_output_monomials]

            poly_optimize_identifiers_list = []
            for i in range(interpretation_net_output_monomials):
                poly_optimize_identifiers = tf.math.softmax(poly_optimize[sparsity*i+interpretation_net_output_monomials:sparsity*(i+1)+interpretation_net_output_monomials])
                poly_optimize_identifiers_list.append(poly_optimize_identifiers)
            poly_optimize_identifiers_list = tf.keras.backend.flatten(poly_optimize_identifiers_list)
            poly_optimize = tf.concat([poly_optimize_coeffs, poly_optimize_identifiers_list], axis=0)
                    
        poly_optimize = tf.convert_to_tensor(poly_optimize, dtype=tf.float32)
        
        poly_optimize_fv_list = []
        for lambda_input_entry in random_lambda_input_data:
            result = 0   
            
            value_without_coefficient_list = []
            for coefficient_multipliers in list_of_monomial_identifiers:
                value_without_coefficient = [lambda_input_value**int(coefficient_multiplier) for coefficient_multiplier, lambda_input_value in zip(coefficient_multipliers, lambda_input_entry)]
                value_without_coefficient_list.append(reduce(lambda x, y: x*y, value_without_coefficient))
            
            
            if interpretation_net_output_monomials == None:
                result = tf.reduce_sum(tf.vectorized_map(lambda x: x[0]*x[1], (value_without_coefficient_list, poly_optimize)))
            else:
                coefficients = poly_optimize[:interpretation_net_output_monomials]
                index_array = poly_optimize[interpretation_net_output_monomials:]
                
                index_list = tf.split(index_array, interpretation_net_output_monomials)
                
                indices = tf.argmax(index_list, axis=1) 
                
                result = tf.reduce_sum(tf.vectorized_map(lambda x: tf.gather(value_without_coefficient_list, x[0])*x[1], (indices, coefficients)))             
                   
            poly_optimize_fv_list.append(result)
            

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


        error = None
        if inet_loss == 'mae':
            error = tf.keras.losses.MAE(lambda_fv, poly_optimize_fv_list)
        elif inet_loss == 'r2':
            error = r2_keras_loss(lambda_fv, poly_optimize_fv_list)  
        else:
            raise SystemExit('Unknown I-Net Metric: ' + inet_loss)                

        error = tf.where(tf.math.is_nan(error), tf.fill(tf.shape(error), np.inf), error)        
                    

        return error #tf.reduce_mean(poly_optimize)#result        
  
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
    
    #base_model = generate_base_model()
    
    random_lambda_input_data = np.random.uniform(low=x_min, high=x_max, size=(per_network_dataset_size, max(1, n)))

    
    model_lambda_placeholder = keras.models.clone_model(base_model)  
    
    dims = [np_arrays.shape for np_arrays in weights_structure]
    
    
    lambda_network_weights = tf.dtypes.cast(tf.convert_to_tensor(lambda_network_weights), tf.float32)
    
    
    
    
    
        
    
    best_result = np.inf

    for current_iteration in range(restarts):
        
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
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
                print("Current best: {} \n Curr_res: {} \n Iteration {}, Step {}".format(best_result_iteration,current_result, current_iteration, current_step), end='\r')
 
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
        
        
    return per_network_poly





