#######################################################################################################################################################
#######################################################################Imports#########################################################################
#######################################################################################################################################################

import itertools 
#from tqdm import tqdm_notebook as tqdm
#import pickle
import cloudpickle
import dill 

import traceback

import numpy as np
import pandas as pd
import scipy as sp

from functools import reduce
from more_itertools import random_product 

#import math
from joblib import Parallel, delayed


from collections.abc import Iterable
#from scipy.integrate import quad

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score
from similaritymeasures import frechet_dist, area_between_two_curves, dtw

from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Input, Model
import tensorflow as tf

import autokeras as ak
from autokeras import adapters, analysers
from tensorflow.python.util import nest

import random 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout

from matplotlib import pyplot as plt
import seaborn as sns

from sympy import Symbol, sympify, lambdify, abc


#udf import
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *

from tqdm import tqdm_notebook as tqdm

#######################################################################################################################################################
#############################################################Setting relevant parameters from current config###########################################
#######################################################################################################################################################

def initialize_InterpretationNet_config_from_curent_notebook(config):
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
    from utilities.utility_functions import flatten, rec_gen
        
    list_of_monomial_identifiers_extended = []

    if laurent:
        variable_sets = [list(flatten([[_d for _d in range(d+1)], [-_d for _d in range(1, neg_d+1)]])) for _ in range(n)]
        list_of_monomial_identifiers_extended = rec_gen(variable_sets)       
    else:
        variable_sets = [[_d for _d in range(d+1)] for _ in range(n)]  
        list_of_monomial_identifiers_extended = rec_gen(variable_sets)
        
    list_of_monomial_identifiers = []
    for monomial_identifier in list_of_monomial_identifiers_extended:
        if np.sum(monomial_identifier) <= d:
            if monomial_vars == None or len(list(filter(lambda x: x != 0, monomial_identifier))) <= monomial_vars:
                list_of_monomial_identifiers.append(monomial_identifier)
            
            
#######################################################################################################################################################
######################################################################AUTOKERAS BLOCKS#################################################################
#######################################################################################################################################################

class CombinedOutputInet(ak.Head):

    def __init__(self, loss = None, metrics = None, **kwargs):
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = None

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

class ClassificationDenseInet(ak.Block):

    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(sparsity, activation='softmax')
        output_node = layer(input_node)
        return output_node    

class RegressionDenseInet(ak.Block):

    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = Dense(interpretation_net_output_monomials)
        output_node = layer(input_node)
        return output_node     





#######################################################################################################################################################
#################################################################I-NET RESULT CALCULATION##############################################################
#######################################################################################################################################################
    
def calculate_interpretation_net_results(lambda_net_train_dataset_list, 
                                         lambda_net_valid_dataset_list, 
                                         lambda_net_test_dataset_list):
        
    epochs_save_range_lambda = range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda) if each_epochs_save_lambda == 1 else range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda+1) if multi_epoch_analysis else range(1,2)
    
    n_jobs_inet_training = n_jobs
    if n_jobs==1 or (samples_list != None and len(samples_list) == 1) or (len(lambda_net_train_dataset_list) == 1 and samples_list == None) or use_gpu:
        n_jobs_inet_training = 1
    verbose = 0 if n_jobs_inet_training == 1 else 11
        
    save_string_list = []      
    for i in range(len(lambda_net_train_dataset_list)):
        save_string_list.append('')

        
    polynomial_dict_test_list = []  
    polynomial_dict_valid_list = []

    
    for lambda_net_valid_dataset, lambda_net_test_dataset in zip(lambda_net_valid_dataset_list, lambda_net_test_dataset_list):
    
        polynomial_dict_valid = {'lstsq_lambda_pred_polynomials': lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list,
                                'lstsq_target_polynomials': lambda_net_valid_dataset.lstsq_target_polynomial_list,
                                'target_polynomials': lambda_net_valid_dataset.target_polynomial_list}    
    
        polynomial_dict_test = {'lstsq_lambda_pred_polynomials': lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list,
                                'lstsq_target_polynomials': lambda_net_test_dataset.lstsq_target_polynomial_list,
                                'target_polynomials': lambda_net_test_dataset.target_polynomial_list}    
        
        polynomial_dict_test_list.append(polynomial_dict_test)
        polynomial_dict_valid_list.append(polynomial_dict_valid)        
                 

            
    if samples_list == None:      
        
        print('----------------------------------------------- TRAINING INTERPRETATION NET -----------------------------------------------')
        
        start = time.time() 
        
        
        parallel_inet = Parallel(n_jobs=n_jobs_inet_training, verbose=verbose, backend='multiprocessing')     
        results_list = parallel_inet(delayed(train_inet)(lambda_net_train_dataset,
                                                           lambda_net_valid_dataset,
                                                           lambda_net_test_dataset,
                                                           current_jobs=n_jobs_inet_training,
                                                           callback_names=['early_stopping'],
                                                           save_string='epochs_' + str(save_epochs)) for lambda_net_train_dataset,
                                                                                                         lambda_net_valid_dataset,
                                                                                                         lambda_net_test_dataset,
                                                                                                         save_epochs in zip(lambda_net_train_dataset_list,
                                                                                                                                          lambda_net_valid_dataset_list,
                                                                                                                                          lambda_net_test_dataset_list,
                                                                                                                                          list(epochs_save_range_lambda)))          
        del parallel_inet
                
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Training Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')
        
        history_list = [result[0] for result in results_list]

        valid_data_list = [result[1] for result in results_list]
        X_valid_list = [valid_data[0] for valid_data in valid_data_list]
        y_valid_list = [valid_data[1] for valid_data in valid_data_list]
        
        test_data_list = [result[2] for result in results_list]
        X_test_list = [test_data[0] for test_data in test_data_list]
        y_test_list = [test_data[1] for test_data in test_data_list]   
        
        
        loss_function_list = [result[3] for result in results_list]
        metrics_list = [result[4] for result in results_list]
        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------ LOADING MODELS -----------------------------------------------------')
        
        start = time.time() 
        
        identifier_type = 'epochs'
        model_list = load_inets(identifier_type=identifier_type, path_identifier_list=list(epochs_save_range_lambda), loss_function_list=loss_function_list, metrics_list=metrics_list)
        
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Loading Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        
    else:
        parallel_inet = Parallel(n_jobs=n_jobs_inet_training, verbose=verbose, backend='multiprocessing') 
        results_list = parallel_inet(delayed(train_inet)(lambda_net_train_dataset.sample(samples),
                                                          lambda_net_valid_dataset,
                                                          lambda_net_test_dataset, 
                                                          current_jobs=n_jobs_inet_training,
                                                          callback_names=['early_stopping'],
                                                          save_string='samples_' + str(samples)) for samples in samples_list)     

        del parallel_inet
                
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Training Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')
        
        history_list = [result[0] for result in results_list]

        valid_data_list = [result[1] for result in results_list]
        X_valid_list = [valid_data[0] for valid_data in valid_data_list]
        y_valid_list = [valid_data[1] for valid_data in valid_data_list]
        
        test_data_list = [result[2] for result in results_list]
        X_test_list = [test_data[0] for test_data in test_data_list]
        y_test_list = [test_data[1] for test_data in test_data_list]   
        
        loss_function_list = [result[3] for result in results_list]
        metrics_list = [result[4] for result in results_list]
        
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------ LOADING MODELS -----------------------------------------------------')
        
        start = time.time() 
        
        identifier_type = 'samples'
        model_list = load_inets(identifier_type=identifier_type, path_identifier_list=samples_list, loss_function_list=loss_function_list, metrics_list=metrics_list)
        
        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Loading Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------- PREDICT INET ------------------------------------------------------')

    start = time.time() 

    for i, (X_test, model) in enumerate(zip(X_test_list, model_list)):
        y_test_pred = model.predict(X_test)
        polynomial_dict_test_list[i]['inet_polynomials'] = y_test_pred


    end = time.time()     
    inet_train_time = (end - start) 
    minutes, seconds = divmod(int(inet_train_time), 60)
    hours, minutes = divmod(minutes, 60)        
    print('Predict Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
    print('---------------------------------------------------------------------------------------------------------------------------')
    if False:
        print('-------------------------------------------------- CALCULATE METAMODEL POLY -----------------------------------------------')

        start = time.time() 

        for i, lambda_net_test_dataset in enumerate(lambda_net_test_dataset_list): 
            metamodel_functions_test = symbolic_metamodeling_function_generation(lambda_net_test_dataset, return_expression='approx', function_metamodeling=False, force_polynomial=True)
            polynomial_dict_test_list[i]['metamodel_poly'] = metamodel_functions_test       

        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Metamodel Poly Optimization Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        print('---------------------------------------------------------------------------------------------------------------------------') 
    if False:
        print('---------------------------------------------------- CALCULATE METAMODEL --------------------------------------------------')

        start = time.time() 

        for i, lambda_net_test_dataset in enumerate(lambda_net_test_dataset_list): 
            metamodel_functions_test = symbolic_metamodeling_function_generation(lambda_net_test_dataset, return_expression='approx', function_metamodeling=False, force_polynomial=False)
            polynomial_dict_test_list[i]['metamodel_functions'] = metamodel_functions_test       

        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Metamodel Optimization Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        print('---------------------------------------------------------------------------------------------------------------------------')   
    if False:
        print('----------------------------------------------- CALCULATE METAMODEL FUNCTION ----------------------------------------------')

        start = time.time() 

        for i, lambda_net_test_dataset in enumerate(lambda_net_test_dataset_list): 
            metamodel_functions_test = symbolic_metamodeling_function_generation(lambda_net_test_dataset, return_expression='approx', function_metamodeling=True)
            polynomial_dict_test_list[i]['metamodel_functions_no_GD'] = metamodel_functions_test       

        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Metamodel Function Optimization Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        print('---------------------------------------------------------------------------------------------------------------------------') 
    if True:
        print('----------------------------------------- CALCULATE SYMBOLIC REGRESSION FUNCTION ------------------------------------------')

        start = time.time() 

        for i, lambda_net_test_dataset in enumerate(lambda_net_test_dataset_list): 
            symbolic_regression_functions_test = symbolic_regression_function_generation(lambda_net_test_dataset)
            polynomial_dict_test_list[i]['symbolic_regression_functions'] = symbolic_regression_functions_test       

        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Symbolic Regression Optimization Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        print('---------------------------------------------------------------------------------------------------------------------------')          
    if True:
        print('------------------------------------------------ CALCULATE PER NETWORK POLY -----------------------------------------------')

        start = time.time() 

        for i, lambda_net_test_dataset in enumerate(lambda_net_test_dataset_list): 
            per_network_poly_test = per_network_poly_generation(lambda_net_test_dataset, optimization_type='scipy')
            polynomial_dict_test_list[i]['per_network_polynomials'] = per_network_poly_test       

        end = time.time()     
        inet_train_time = (end - start) 
        minutes, seconds = divmod(int(inet_train_time), 60)
        hours, minutes = divmod(minutes, 60)        
        print('Per Network Optimization Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
        print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ CALCULATE FUNCTION VALUES ------------------------------------------------')                

    start = time.time() 

    function_values_test_list = []
    for lambda_net_test_dataset, polynomial_dict_test in zip(lambda_net_test_dataset_list, polynomial_dict_test_list):
        function_values_test = calculate_all_function_values(lambda_net_test_dataset, polynomial_dict_test)
        function_values_test_list.append(function_values_test)

    end = time.time()     
    inet_train_time = (end - start) 
    minutes, seconds = divmod(int(inet_train_time), 60)
    hours, minutes = divmod(minutes, 60)        
    print('FV Calculation Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------- CALCULATE SCORES ----------------------------------------------------')                

    start = time.time() 

    scores_test_list = []
    distrib_dict_test_list = []
    for function_values_test, polynomial_dict_test in zip(function_values_test_list, polynomial_dict_test_list):
        scores_test, distrib_test = evaluate_all_predictions(function_values_test, polynomial_dict_test)
        scores_test_list.append(scores_test)
        distrib_dict_test_list.append(distrib_test)

    end = time.time()     
    inet_train_time = (end - start) 
    minutes, seconds = divmod(int(inet_train_time), 60)
    hours, minutes = divmod(minutes, 60)        
    print('Score Calculation Time: ' +  f'{hours:d}:{minutes:02d}:{seconds:02d}')     
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')         

    if not nas:
        generate_history_plots(history_list, by=identifier_type)
        save_results(history_list, scores_test_list, by=identifier_type)    

    

            
    return (history_list, 
            
            #scores_valid_list
            scores_test_list, 
            
            #function_values_valid_list, 
            function_values_test_list, 
            
            #polynomial_dict_valid_list,
            polynomial_dict_test_list,
            
            #distrib_dict_valid_list,
            distrib_dict_test_list,
            
            model_list)
    
    
#######################################################################################################################################################
######################################################################I-NET TRAINING###################################################################
#######################################################################################################################################################

def load_inets(identifier_type, path_identifier_list, loss_function_list, metrics_list):
    
    
    paths_dict = generate_paths(path_type = 'interpretation_net')

    generic_path_identifier = paths_dict['path_identifier_interpretation_net_data']
    if nas:
        generic_path_identifier = nas_type + '_' + generic_path_identifier
    
    save_string_list = []
    for path_identifier in path_identifier_list:
        save_string_list.append(str(identifier_type) + '_' + str(path_identifier))
       
    directory = './data/saved_models/'
    
    

    model_list = []
    from tensorflow.keras.utils import CustomObjectScope
    for save_string, loss_function, metrics in zip(save_string_list, loss_function_list, metrics_list):
        loss_function = dill.loads(loss_function)
        metrics = dill.loads(metrics)         
        
        #with CustomObjectScope({'custom_loss': loss_function}):
        custom_object_dict = {}
        custom_object_dict[loss_function.__name__] = loss_function
        for metric in  metrics:
            custom_object_dict[metric.__name__] = metric        
        model = tf.keras.models.load_model(directory + generic_path_identifier + save_string, custom_objects=custom_object_dict) # #, compile=False
        model_list.append(model)
        
    return model_list

def train_inet(lambda_net_train_dataset,
              lambda_net_valid_dataset,
              lambda_net_test_dataset, 
              current_jobs,
              callback_names = [],
              save_string = None):
    
   
    global optimizer
    global loss
    global data_reshape_version
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    
    ############################## DATA PREPARATION ###############################

    X_train = np.array(lambda_net_train_dataset.weight_list)
    X_valid = np.array(lambda_net_valid_dataset.weight_list)
    X_test = np.array(lambda_net_test_dataset.weight_list) 
        
    if evaluate_with_real_function: #target polynomial as inet target
        y_train = np.array(lambda_net_train_dataset.target_polynomial_list)
        y_valid = np.array(lambda_net_valid_dataset.target_polynomial_list)
        y_test = np.array(lambda_net_test_dataset.target_polynomial_list)
    else: #lstsq lambda pred polynomial as inet target
        y_train = np.array(lambda_net_train_dataset.lstsq_lambda_pred_polynomial_list)
        y_valid = np.array(lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list)
        y_test = np.array(lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list)
        
        if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
            if data_reshape_version == None:
                data_reshape_version = 2
            X_train, X_train_flat = restructure_data_cnn_lstm(X_train, version=data_reshape_version, subsequences=None)
            X_valid, X_valid_flat = restructure_data_cnn_lstm(X_valid, version=data_reshape_version, subsequences=None)
            X_test, X_test_flat = restructure_data_cnn_lstm(X_test, version=data_reshape_version, subsequences=None)
        
    ############################## OBJECTIVE SPECIFICATION AND LOSS FUNCTION ADJUSTMENTS ###############################
        
    base_model = generate_base_model()
    random_evaluation_dataset = np.random.uniform(low=x_min, high=x_max, size=(random_evaluation_dataset_size, n))
            
    weights_structure = base_model.get_weights()
    dims = [np_arrays.shape for np_arrays in weights_structure]        
    if consider_labels_training: #coefficient-based evaluation
        
        if interpretation_net_output_monomials != None:
            raise SystemExit('No coefficient-based optimization possible with reduced output monomials - Please change settings') 
        
        if evaluate_with_real_function: #based on comparison real and predicted polynomial coefficients        
            if inet_loss == 'mae':
                loss_function = 'mean_absolute_error'
            elif inet_loss == 'r2':      
                loss_function = r2_keras_loss
            else:
                raise SystemExit('Unknown I-Net Metric: ' + inet_loss)   
            
            metrics = ['mean_absolute_error', r2_keras_loss]
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers))
            
            valid_data = (X_valid, y_valid)
            y_train_model = y_train
        else: #based on comparison lstsq-lambda and predicted polynomial coefficients
            loss_function = inet_coefficient_loss_wrapper(inet_loss)
            
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, base_model))            
            
            if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
                y_train_model = np.hstack((y_train, X_train_flat))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
            else:
                y_train_model = np.hstack((y_train, X_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid)))                                  
    else: #fv-based evaluation
        if evaluate_with_real_function: #based on in-loss fv calculation of real and predicted polynomial
            
            loss_function = inet_poly_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers)
            
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers))    
             
            valid_data = (X_valid, y_valid)
            y_train_model = y_train
        else: #in-loss prediction of lambda-nets
            
            #loss_function = inet_lambda_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers, base_model)
            loss_function = inet_lambda_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers, base_model, weights_structure, dims)
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                #metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, base_model)) 
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers, base_model, weights_structure, dims)) 
            
            if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
                y_train_model = np.hstack((y_train, X_train_flat))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
            else:
                y_train_model = np.hstack((y_train, X_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid)))          
                 
        
    ############################## BUILD MODEL ###############################
    
    if nas:
        from tensorflow.keras.utils import CustomObjectScope
        
        custom_object_dict = {}
        loss_function_name = loss_function.__name__
        custom_object_dict[loss_function_name] = loss_function
        metric_names = []
        for metric in  metrics:
            metric_name = metric.__name__
            metric_names.append(metric_name)
            custom_object_dict[metric_name] = metric  
        
        #print(custom_object_dict)    
        #print(metric_names)
        #print(loss_function_name)
        
        with CustomObjectScope(custom_object_dict):
            if nas_type == 'SEQUENTIAL':
                input_node = ak.Input()
                hidden_node = ak.DenseBlock()(input_node)
                
                        
                if interpretation_net_output_monomials == None:
                    output_node = ak.RegressionHead()(hidden_node)  
                    #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                else:
                    #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                    outputs_coeff = RegressionDenseInet()(hidden_node)  
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(interpretation_net_output_monomials):
                        outputs_identifer =  ClassificationDenseInet()(hidden_node)
                        outputs_list.append(outputs_identifer)

                    output_node = CombinedOutputInet()(outputs_list)

            elif nas_type == 'CNN': 
                input_node = ak.Input()
                hidden_node = ak.ConvBlock()(input_node)
                hidden_node = ak.DenseBlock()(hidden_node)
                
                if interpretation_net_output_monomials == None:
                    output_node = ak.RegressionHead()(hidden_node)  
                    #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                else:
                    #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                    outputs_coeff = RegressionDenseInet()(hidden_node)  
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(interpretation_net_output_monomials):
                        outputs_identifer =  ClassificationDenseInet()(hidden_node)
                        outputs_list.append(outputs_identifer)

                    output_node = CombinedOutputInet()(outputs_list)
                    
                    
            elif nas_type == 'LSTM':
                input_node = ak.Input()
                hidden_node = ak.RNNBlock()(input_node)
                hidden_node = ak.DenseBlock()(hidden_node)
                
                if interpretation_net_output_monomials == None:
                    output_node = ak.RegressionHead()(hidden_node)  
                    #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                else:
                    #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                    outputs_coeff = RegressionDenseInet()(hidden_node)  
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(interpretation_net_output_monomials):
                        outputs_identifer =  ClassificationDenseInet()(hidden_node)
                        outputs_list.append(outputs_identifer)

                    output_node = CombinedOutputInet()(outputs_list)            
                
            elif nas_type == 'CNN-LSTM': 
                input_node = ak.Input()
                hidden_node = ak.ConvBlock()(input_node)
                hidden_node = ak.RNNBlock()(hidden_node)
                hidden_node = ak.DenseBlock()(hidden_node)

                if interpretation_net_output_monomials == None:
                    output_node = ak.RegressionHead()(hidden_node)  
                    #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                else:
                    #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                    outputs_coeff = RegressionDenseInet()(hidden_node)  
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(interpretation_net_output_monomials):
                        outputs_identifer =  ClassificationDenseInet()(hidden_node)
                        outputs_list.append(outputs_identifer)

                    output_node = CombinedOutputInet()(outputs_list)           
                
            elif nas_type == 'CNN-LSTM-parallel':                         
                input_node = ak.Input()
                hidden_node1 = ak.ConvBlock()(input_node)
                hidden_node2 = ak.RNNBlock()(input_node)
                hidden_node = ak.Merge()([hidden_node1, hidden_node2])
                hidden_node = ak.DenseBlock()(hidden_node)
                
                if interpretation_net_output_monomials == None:
                    output_node = ak.RegressionHead()(hidden_node)  
                    #output_node = ak.RegressionHead(output_dim=sparsity)(hidden_node)  
                else:
                    #outputs_coeff = ak.RegressionHead(output_dim=interpretation_net_output_monomials)(hidden_node)  
                    outputs_coeff = RegressionDenseInet()(hidden_node)  
                    outputs_list = [outputs_coeff]
                    for outputs_index in range(interpretation_net_output_monomials):
                        outputs_identifer =  ClassificationDenseInet()(hidden_node)
                        outputs_list.append(outputs_identifer)

                    output_node = CombinedOutputInet()(outputs_list)            

            directory = './data/autokeras/' + nas_type + '_' + paths_dict['path_identifier_interpretation_net_data'] + save_string

            auto_model = ak.AutoModel(inputs=input_node, 
                                outputs=output_node,
                                loss=loss_function_name,
                                metrics=metric_names,
                                objective='val_loss',
                                overwrite=True,
                                #tuner='hyperband',#"greedy",
                                max_trials=nas_trials,
                                directory=directory,
                                seed=RANDOM_SEED+1)

            ############################## PREDICTION ###############################
                        
            auto_model.fit(
                x=X_train,
                y=y_train_model,
                validation_data=valid_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=return_callbacks_from_string('early_stopping'),
                )


            history = auto_model.tuner.oracle.get_best_trials(min(nas_trials, 5))
            model = auto_model.export_model()

            model.save('./data/saved_models/' + nas_type + '_' + paths_dict['path_identifier_interpretation_net_data'] + save_string)

    else: 
        inputs = Input(shape=X_train.shape[1], name='input')
        
        hidden = Dense(dense_layers[0], activation='relu', name='hidden1_' + str(dense_layers[0]))(inputs)

        if dropout > 0:
            hidden = Dropout(dropout, name='dropout1_' + str(dropout))(hidden)

        for layer_index, neurons in enumerate(dense_layers[1:]):
            if dropout > 0 and layer_index > 0:
                hidden = Dropout(dropout, name='dropout' + str(layer_index+2) + '_' + str(dropout))(hidden)            
            hidden = Dense(neurons, activation='relu', name='hidden' + str(layer_index+2) + '_' + str(neurons))(hidden)
                
        if dropout_output > 0:
            hidden = Dropout(dropout_output, name='dropout_output_' + str(dropout_output))(hidden)            
        
        if interpretation_net_output_monomials == None:
            outputs = Dense(sparsity, name='output_' + str(neurons))(hidden)
        else:
            outputs_coeff = Dense(interpretation_net_output_monomials, name='output_coeff_' + str(interpretation_net_output_monomials))(hidden)

            outputs_list = [outputs_coeff]
            for outputs_index in range(interpretation_net_output_monomials):
                outputs_identifer = Dense(sparsity, activation='softmax', name='output_identifier' + str(outputs_index+1) + '_' + str(sparsity))(hidden)
                outputs_list.append(outputs_identifer)
                
                
            outputs = concatenate(outputs_list, name='output_combined')
            
            
        model = Model(inputs=inputs, outputs=outputs)
            
        callbacks = return_callbacks_from_string(callback_names)            

        if optimizer == "custom":
            optimizer = keras.optimizers.Adam(learning_rate=2e-05)

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=metrics
                     )

        verbosity = 1 if n_jobs ==1 else 0

        ############################## PREDICTION ###############################
        history = model.fit(X_train,
                  y_train_model,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_data=valid_data,
                  callbacks=callbacks,
                  verbose=verbosity)
    
        history = history.history
        
        model.save('./data/saved_models/' + paths_dict['path_identifier_interpretation_net_data'] + save_string)
        
        
    return history, (X_valid, y_valid), (X_test, y_test), dill.dumps(loss_function), dill.dumps(metrics) 
     
def calculate_all_function_values(lambda_net_dataset, polynomial_dict):
    
    n_jobs_parallel_fv = n_jobs
    
    
    if n_jobs_parallel_fv <= 5:
        n_jobs_parallel_fv = 10
        #backend='threading'
    #else:
        #backend='loky'
        
    backend='threading'
        
    
        
    function_value_dict = {
        'lambda_preds': lambda_net_dataset.make_prediction_on_test_data(),
        'target_polynomials': lambda_net_dataset.return_target_poly_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend),          
        'lstsq_lambda_pred_polynomials': lambda_net_dataset.return_lstsq_lambda_pred_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend),         
        'lstsq_target_polynomials': lambda_net_dataset.return_lstsq_target_polynomial_fvs_on_test_data(n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend),   
        'inet_polynomials': parallel_fv_calculation_from_polynomial(polynomial_dict['inet_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend),      
    }
    
    
    try:
        function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_poly'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
        function_value_dict['metamodel_poly'] =  function_values
    except KeyError:
        print(KeyError)    
             
    try:
        function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metamodel_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
        function_value_dict['metamodel_functions'] = function_values
    except KeyError:
        print(KeyError)   
        
    try:
        function_values = parallel_fv_calculation_from_sympy(polynomial_dict['metametamodel_functions_no_GDmodel_poly'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
        function_value_dict['metamodel_functions_no_GD'] = function_values
    except KeyError:
        print(KeyError)   
        
    try:
        function_values = parallel_fv_calculation_from_sympy(polynomial_dict['symbolic_regression_functions'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)
        function_value_dict['symbolic_regression_functions'] = function_values
    except KeyError as ke:
        print(ke)
        print('EXIT symbolic_regression_functions')
        traceback.print_exc()        
    try:
        function_values = parallel_fv_calculation_from_polynomial(polynomial_dict['per_network_polynomials'], lambda_net_dataset.X_test_data_list, n_jobs_parallel_fv=n_jobs_parallel_fv, backend=backend)        
        function_value_dict['per_network_polynomials'] = function_values
    except KeyError:
        print(KeyError)
    
    
    return function_value_dict
    
def evaluate_all_predictions(function_value_dict, polynomial_dict):
    
    ############################## EVALUATION ###############################
    evaluation_key_list = []
    evaluation_scores_list = []
    evaluation_distrib_list = []
    for combination in itertools.combinations(function_value_dict.keys(), r=2):
        key_1 = combination[0]
        key_2 = combination[1]
        
        try:
            polynomials_1 = polynomial_dict[key_1]
            if type(polynomials_1[0]) != np.ndarray and type(polynomials_1[0]) != list:
                polynomials_1 = None            
        except KeyError:
            polynomials_1 = None
            
        try:
            polynomials_2 = polynomial_dict[key_2]
            if type(polynomials_2[0]) != np.ndarray and type(polynomials_2[0]) != list:
                polynomials_2 = None
        except KeyError:
            polynomials_2 = None
            

            
        function_values_1 = function_value_dict[key_1]
        function_values_2 = function_value_dict[key_2]
        
        
        evaluation_key = key_1 + '_VS_' + key_2
        print(evaluation_key)
        evaluation_key_list.append(evaluation_key)
        
        evaluation_scores, evaluation_distrib = evaluate_interpretation_net(polynomials_1, 
                                                                            polynomials_2, 
                                                                            function_values_1, 
                                                                            function_values_2)        
        evaluation_scores_list.append(evaluation_scores)
        evaluation_distrib_list.append(evaluation_distrib)
        
        
    scores_dict = pd.DataFrame(data=evaluation_scores_list,
                               index=evaluation_key_list)        
        
    
    mae_distrib_dict = pd.DataFrame(data=[evaluation_distrib['MAE'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
    
        
    r2_distrib_dict = pd.DataFrame(data=[evaluation_distrib['R2'] for evaluation_distrib in evaluation_distrib_list],
                                    index=evaluation_key_list)
 
    
    distrib_dicts = {'MAE': mae_distrib_dict, 
                     'R2': r2_distrib_dict}        
    
    
    return scores_dict, distrib_dicts

def per_network_poly_generation(lambda_net_dataset, optimization_type='scipy'): 
    
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    if optimization_type=='tf':
        
        per_network_hyperparams = {
            'optimizer': tf.keras.optimizers.RMSprop,
            'lr': 0.02,
            'max_steps': 500,
            'early_stopping': 10,
            'restarts': 3,
            'per_network_dataset_size': 5000,
        }

        printing = True if n_jobs == 1 else False

        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 #'list_of_monomial_identifiers': list_of_monomial_identifiers,
                 'x_min': x_min,
                 'x_max': x_max,
                 }

        parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')

        per_network_optimization_polynomials = parallel_per_network(delayed(per_network_poly_optimization_tf)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                              lambda_network_weights, 
                                                                                                              list_of_monomial_identifiers, 
                                                                                                              config,
                                                                                                              optimizer = per_network_hyperparams['optimizer'],
                                                                                                              lr = per_network_hyperparams['lr'], 
                                                                                                              max_steps = per_network_hyperparams['max_steps'], 
                                                                                                              early_stopping = per_network_hyperparams['early_stopping'], 
                                                                                                              restarts = per_network_hyperparams['restarts'],
                                                                                                              printing = printing,
                                                                                                              return_error = True) for lambda_network_weights in lambda_network_weights_list)      

        del parallel_per_network
        
    elif optimization_type=='scipy':    

        per_network_hyperparams = {
            'optimizer':  'Powell',
            'jac': 'fprime',
            'max_steps': 5000,#100,
            'restarts': 3,
            'per_network_dataset_size': 500,
        }

        printing = True if n_jobs == 1 else False


        lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)


        config = {
                 'n': n,
                 'inet_loss': inet_loss,
                 'sparsity': sparsity,
                 'lambda_network_layers': lambda_network_layers,
                 'interpretation_net_output_shape': interpretation_net_output_shape,
                 'RANDOM_SEED': RANDOM_SEED,
                 'nas': nas,
                 'number_of_lambda_weights': number_of_lambda_weights,
                 'interpretation_net_output_monomials': interpretation_net_output_monomials,
                 'x_min': x_min,
                 'x_max': x_max,
                 }

        parallel_per_network = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')

        result_list_per_network = parallel_per_network(delayed(per_network_poly_optimization_scipy)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                  lambda_network_weights, 
                                                                                                                  list_of_monomial_identifiers, 
                                                                                                                  config,
                                                                                                                  optimizer = per_network_hyperparams['optimizer'],
                                                                                                                  jac = per_network_hyperparams['jac'],
                                                                                                                  max_steps = per_network_hyperparams['max_steps'], 
                                                                                                                  restarts = per_network_hyperparams['restarts'],
                                                                                                                  printing = printing,
                                                                                                                  return_error = True) for lambda_network_weights in lambda_network_weights_list)      
        per_network_optimization_errors = [result[0] for result in result_list_per_network]
        per_network_optimization_polynomials = [result[1] for result in result_list_per_network]          
        
        del parallel_per_network    
    
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers    
        
    return per_network_optimization_polynomials

def symbolic_regression_function_generation(lambda_net_dataset):
    
    symbolic_regression_hyperparams = {
        'dataset_size': 500,
    }
    
    config = {
            'n': n,
            'd': d,
            'inet_loss': inet_loss,
            'sparsity': sparsity,
            'lambda_network_layers': lambda_network_layers,
            'interpretation_net_output_shape': interpretation_net_output_shape,
            'RANDOM_SEED': RANDOM_SEED,
            'nas': nas,
            'number_of_lambda_weights': number_of_lambda_weights,
            'interpretation_net_output_monomials': interpretation_net_output_monomials,
            'fixed_initialization_lambda_training': fixed_initialization_lambda_training,
            'dropout': dropout,
            'lambda_network_layers': lambda_network_layers,
            'optimizer_lambda': optimizer_lambda,
            'loss_lambda': loss_lambda,        
             #'list_of_monomial_identifiers': list_of_monomial_identifiers,
             'x_min': x_min,
             'x_max': x_max,
             }

    parallel_symbolic_regression = Parallel(n_jobs=n_jobs, verbose=11, backend='loky')

    return_error = False
    
    result_list_symbolic_regression = parallel_symbolic_regression(delayed(symbolic_regression)(lambda_net, 
                                                                                  config,
                                                                                  symbolic_regression_hyperparams,
                                                                                  #printing = printing,
                                                                                  return_error = return_error) for lambda_net in lambda_net_dataset.lambda_net_list)      

    del parallel_symbolic_regression  
    
    if return_error:
        symbolic_regression_errors = [result[0] for result in result_list_symbolic_regression]
        symbolic_regression_functions = [result[1] for result in result_list_symbolic_regression]   
    else:
        return result_list_symbolic_regression
    
    return symbolic_regression_errors, symbolic_regression_functions


def symbolic_metamodeling_function_generation(lambda_net_dataset, return_expression='approx', function_metamodeling=True, force_polynomial=False):
    
    metamodeling_hyperparams = {
        'num_iter': 500,
        'batch_size': None,
        'learning_rate': 0.01,        
        'dataset_size': 500,
    }

    #list_of_monomial_identifiers_numbers = np.array([list(monomial_identifiers) for monomial_identifiers in list_of_monomial_identifiers]).astype(float)  

    #printing = True if n_jobs == 1 else False

    #lambda_network_weights_list = np.array(lambda_net_dataset.weight_list)
    
    config = {
            'n': n,
            'd': d,
            'inet_loss': inet_loss,
            'sparsity': sparsity,
            'lambda_network_layers': lambda_network_layers,
            'interpretation_net_output_shape': interpretation_net_output_shape,
            'RANDOM_SEED': RANDOM_SEED,
            'nas': nas,
            'number_of_lambda_weights': number_of_lambda_weights,
            'interpretation_net_output_monomials': interpretation_net_output_monomials,
            'fixed_initialization_lambda_training': fixed_initialization_lambda_training,
            'dropout': dropout,
            'lambda_network_layers': lambda_network_layers,
            'optimizer_lambda': optimizer_lambda,
            'loss_lambda': loss_lambda,        
             #'list_of_monomial_identifiers': list_of_monomial_identifiers,
             'x_min': x_min,
             'x_max': x_max,
             }

    parallel_metamodeling = Parallel(n_jobs=n_jobs, verbose=11, backend='loky')

    return_error = False 
    
    result_list_metamodeling = parallel_metamodeling(delayed(symbolic_metamodeling)(lambda_net, 
                                                                                  config,
                                                                                  metamodeling_hyperparams,
                                                                                  #printing = printing,
                                                                                  return_error = return_error,
                                                                                  return_expression=return_expression,
                                                                                  function_metamodeling=function_metamodeling,
                                                                                  force_polynomial=force_polynomial) for lambda_net in lambda_net_dataset.lambda_net_list)      

    del parallel_metamodeling  
    
    if return_error:
        metamodeling_errors = [result[0] for result in result_list_metamodeling]
        metamodeling_polynomials = [result[1] for result in result_list_metamodeling]   
    else:
        return result_list_metamodeling
    
    return metamodeling_errors, metamodeling_polynomials
    
    
    
def reduce_polynomials(polynomial_list):
    
    return
    
#######################################################################################################################################################
################################################################SAVING AND PLOTTING RESULTS############################################################
#######################################################################################################################################################    
    
    
def generate_history_plots(history_list, by='epochs'):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    for i, history in enumerate(history_list):  
        
        if by == 'epochs':
            index= (i+1)*each_epochs_save_lambda if each_epochs_save_lambda==1 else i*each_epochs_save_lambda if i > 1 else each_epochs_save_lambda if i==1 else 1
        elif by == 'samples':
            index = i
        
        plt.plot(history[list(history.keys())[1]])
        plt.plot(history[list(history.keys())[len(history.keys())//2+1]])
        plt.title('model ' + list(history.keys())[len(history.keys())//2+1])
        plt.ylabel('metric')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        if by == 'epochs':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/' + list(history.keys())[len(history.keys())//2+1] + '_epoch_' + str(index).zfill(3) + '.png')
        elif by == 'samples':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/' + list(history.keys())[len(history.keys())//2+1] + '_samples_' + str(samples_list[index]).zfill(5) + '.png')
        plt.clf()
        
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        if by == 'epochs':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/loss_' + '_epoch_' + str(index).zfill(3) + '.png')    
        elif by == 'samples':
            plt.savefig('./data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/loss_' + '_samples_' + str(samples_list[index]).zfill(5) + '.png')    
        if i < len(history_list)-1:
            plt.clf() 
            
            
def save_results(history_list, scores_list, by='epochs'):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    if by == 'epochs':
        path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_epochs' + '.pkl'
    elif by == 'samples':
        path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/history_samples' + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(history_list, f, protocol=2)   
        
        
    if by == 'epochs':
        path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/scores_epochs' + '.pkl'
    elif by == 'samples':
        path = './data/results/' + paths_dict['path_identifier_interpretation_net_data'] + '/scores_samples' + '.pkl'    
    with open(path, 'wb') as f:
        pickle.dump(scores_list, f, protocol=2)  
        

def generate_inet_comparison_plot(scores_list, plot_metric_list, ylim=None):
        
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    
    keys = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials', 'inet_polynomials', 'per_network_polynomials',  'metamodel_poly', 'metamodel_functions', 'metamodel_functions_no_GD', 'symbolic_regression_functions']
    
    evaluation_key_list = []
    for combination in itertools.combinations(keys, r=2):
        key_1 = combination[0]
        key_2 = combination[1]
        
        evaluation_key = key_1 + '_VS_' + key_2
        
        if evaluation_key in scores_list.index:
            evaluation_key_list.append(evaluation_key)
    
    epochs_save_range_lambda = range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda) if each_epochs_save_lambda == 1 else range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda+1) if multi_epoch_analysis else range(1,2)


    if samples_list == None:
        x_axis_steps = [(i+1)*each_epochs_save_lambda if each_epochs_save_lambda==1 else i*each_epochs_save_lambda if i > 1 else each_epochs_save_lambda if i==1 else 1 for i in epochs_save_range_lambda]
        x_max = epochs_lambda
    else:
        x_axis_steps = samples_list
        x_max = samples_list[-1]

    #Plot Polynom, lamdba net, and Interpration net
    length_plt = len(plot_metric_list)
    if length_plt >= 2:
        fig, ax = plt.subplots(length_plt//2, 2, figsize=(30,20))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20,10))

    for index, metric in enumerate(plot_metric_list):
        
        plot_scores_dict = {}
        for key in evaluation_key_list:
            try:
                scores_list[-1][metric].loc[key]
                plot_scores_dict[key] = []
            except:
                #print(key + 'not in scores_list')
                continue
            
        
        for scores in scores_list:
            for key in evaluation_key_list:
                try:
                    plot_scores_dict[key].append(scores[metric].loc[key])
                except:
                    #print(key + 'not in scores_list')
                    continue
                                        
            
        plot_df = pd.DataFrame(data=np.vstack(plot_scores_dict.values()).T, 
                               index=x_axis_steps,
                               columns=plot_scores_dict.keys())

        if length_plt >= 2:
            ax[index//2, index%2].set_title(metric)
            sns.set(font_scale = 1.25)
            p = sns.lineplot(data=plot_df, ax=ax[index//2, index%2])
        else:
            ax.set_title(metric)
            sns.set(font_scale = 1.25)
            p = sns.lineplot(data=plot_df, ax=ax)

        if ylim != None:
            p.set(ylim=ylim)

        p.set_yticklabels(np.round(p.get_yticks(), 2), size = 20)
        p.set_xticklabels(p.get_xticks(), size = 20)     
        
        #p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        p.legend(loc='upper center', bbox_to_anchor=(0.47, -0.1),
          fancybox=False, shadow=False, ncol=2, fontsize=12)   
        
    plt.subplots_adjust(wspace=0.1, hspace=0.75)
    
    location = './data/plotting/'
    folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
    if samples_list == None:
        file = 'multi_epoch' + '.pdf'
    else:
        file = 'sample_list' + '-'.join([str(samples_list[0]), str(samples_list[-1])]) + '.pdf'

    path = location + folder + file

    plt.savefig(path, format='pdf')
    plt.show()






def plot_and_save_single_polynomial_prediction_evaluation(lambda_net_test_dataset_list, function_values_test_list, polynomial_dict_test_list, rand_index=1, plot_type=2):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    x_vars = ['x' + str(i) for i in range(1, n+1)]

    columns = x_vars.copy()
    columns.append('FVs')

    columns_single = x_vars.copy()

    vars_plot = lambda_net_test_dataset_list[-1].X_test_data_list[rand_index]    
    
    custom_representation_keys_fixed = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials']
    custom_representation_keys_dynamic = ['inet_polynomials', 'per_network_polynomials']
    sympy_representation_keys = ['metamodel_poly', 'metamodel_functions', 'metamodel_functions_no_GD', 'symbolic_regression_functions']
    
    #keys = ['target_polynomials', 'lstsq_target_polynomials', 'lstsq_lambda_pred_polynomials', 'inet_polynomials', 'per_network_polynomials', 'metamodel_functions']
    
    lambda_train_data = lambda_net_test_dataset_list[-1].y_test_data_list[rand_index].ravel()
    lambda_train_data_size = lambda_train_data.shape[0]
    lambda_train_data_str = np.array(['Lambda Train Data' for i in range(lambda_train_data_size)])  
    columns_single.append('Lambda Train Data')
    
    lambda_model_preds = function_values_test_list[-1]['lambda_preds'][rand_index].ravel()
    eval_size_plot = lambda_model_preds.shape[0]
    lambda_model_preds_str = np.array(['Lambda Model Preds' for i in range(eval_size_plot)])
    columns_single.append('Lambda Model Preds')
    
    identifier_list =[lambda_train_data_str, lambda_model_preds_str]
    plot_data_single_list = [vars_plot, lambda_train_data, lambda_model_preds]
    for key in custom_representation_keys_fixed:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue            
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, force_complete_poly_representation=True, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
    
    for key in custom_representation_keys_dynamic:
        try:
            polynomial_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue        
        polynomial_by_key_string = get_sympy_string_from_coefficients(polynomial_by_key, round_digits=4)
        polynomial_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(polynomial_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))
        
    for key in sympy_representation_keys:
        try:
            function_by_key = polynomial_dict_test_list[-1][key][rand_index]
        except:
            continue
        function_by_key_string = str(function_by_key)
        function_by_key_fvs = function_values_test_list[-1][key][rand_index]                
            
        plot_data_single_list.append(function_by_key_fvs)
        columns_single.append(key)
        identifier_list.append(np.array([key for i in range(eval_size_plot)]))        
    
    identifier = np.concatenate(identifier_list)
    plot_data_single = pd.DataFrame(data=np.column_stack(plot_data_single_list), columns=columns_single)
    vars_plot_all_preds = np.vstack([vars_plot for i in range(len(columns_single[n:]))])
    preds_plot_all = np.vstack(plot_data_single_list[1:]).ravel()         
        
    plot_data = pd.DataFrame(data=np.column_stack([vars_plot_all_preds, preds_plot_all]), columns=columns)
    plot_data['Identifier'] = identifier       
     
    
    location = './data/plotting/'
    folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
        
    if plot_type == 1:
        
        
        pp = sns.pairplot(data=plot_data,
                      #kind='reg',
                      hue='Identifier',
                      y_vars=['FVs'],
                      x_vars=x_vars, 
                      height=7.5,
                      aspect=2)
        file = 'pp3in1_' + str(rand_index) + '.pdf'                 
        
    elif plot_type == 2:

        pp = sns.pairplot(data=plot_data,
                          #kind='reg',
                          hue='Identifier',
                          #y_vars=['FVs'],
                          #x_vars=x_vars, 
                          height=10//n)
             
        file = 'pp3in1_extended_' + str(rand_index) + '.pdf'  
        
    elif plot_type == 3:
        
        pp = sns.pairplot(data=plot_data_single,
                          #kind='reg',
                          y_vars=columns_single[n:],
                          x_vars=x_vars, 
                          height=3,
                          aspect=3)

        file = 'pp1_' + str(rand_index) + '.pdf'                   
        
    path = location + folder + file
    pp.savefig(path, format='pdf')
    plt.show()    
    
    if False:
        real_poly_VS_lstsq_target_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_target_poly)
        real_poly_VS_lstsq_target_poly_r2 = r2_score(real_poly_fvs, lstsq_target_poly)        

        real_poly_VS_inet_poly_mae = mean_absolute_error(real_poly_fvs, inet_poly_fvs)
        real_poly_VS_inet_poly_r2 = r2_score(real_poly_fvs, inet_poly_fvs)    

        real_poly_VS_perNet_poly_mae = mean_absolute_error(real_poly_fvs, per_network_opt_poly_fvs)
        real_poly_VS_perNet_poly_r2 = r2_score(real_poly_fvs, per_network_opt_poly_fvs)    

        real_poly_VS_lambda_model_preds_mae = mean_absolute_error(real_poly_fvs, lambda_model_preds)
        real_poly_VS_lambda_model_preds_r2 = r2_score(real_poly_fvs, lambda_model_preds)

        real_poly_VS_lstsq_lambda_preds_poly_mae = mean_absolute_error(real_poly_fvs, lstsq_lambda_preds_poly)
        real_poly_VS_lstsq_lambda_preds_poly_r2 = r2_score(real_poly_fvs, lstsq_lambda_preds_poly)   

        from prettytable import PrettyTable

        tab = PrettyTable()

        tab.field_names = ["Comparison",  "MAE", "R2-Score", "Poly 1", "Poly 2"]
        tab._max_width = {"Poly 1" : 50, "Poly 2" : 50}

        tab.add_row(["Target Poly \n vs. \n LSTSQ Target Poly \n", real_poly_VS_lstsq_target_poly_mae, real_poly_VS_lstsq_target_poly_r2, polynomial_target_string, polynomial_lstsq_target_string])
        tab.add_row(["Target Poly \n vs. \n I-Net Poly \n", real_poly_VS_inet_poly_mae, real_poly_VS_inet_poly_r2, polynomial_target_string, polynomial_inet_string])
        tab.add_row(["Target Poly \n vs. \n Per Network Opt Poly \n", real_poly_VS_perNet_poly_mae, real_poly_VS_perNet_poly_r2, polynomial_target_string, polynomial_per_network_opt_string])
        tab.add_row(["Target Poly \n vs. \n Lambda Preds \n", real_poly_VS_lambda_model_preds_mae, real_poly_VS_lambda_model_preds_r2, polynomial_target_string, '-'])
        tab.add_row(["Target Poly \n vs. \n LSTSQ Lambda Preds Poly \n", real_poly_VS_lstsq_lambda_preds_poly_mae, real_poly_VS_lstsq_lambda_preds_poly_r2, polynomial_target_string, polynomial_lstsq_lambda_string])

        print(tab)

            
def restructure_data_cnn_lstm(X_data, version=2, subsequences=None):

    #version == 0: one sequence for biases and one sequence for weights per layer (padded to maximum size)
    #version == 1: each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer (no. columns == number of paths and no. rows = number of layers/length of path)
    #version == 2:each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer + transpose matrices  (no. columns == number of layers/length of path and no. rows = number of paths )
    
    base_model = generate_base_model()
       
    X_data_flat = X_data

    shaped_weights_list = []
    for data in tqdm(X_data):
        shaped_weights = shape_flat_weights(data, base_model.get_weights())
        shaped_weights_list.append(shaped_weights)

    max_size = 0
    for weights in shaped_weights:
        max_size = max(max_size, max(weights.shape))      


    if version == 0: #one sequence for biases and one sequence for weights per layer (padded to maximum size)
        X_data_list = []
        for shaped_weights in tqdm(shaped_weights_list):
            padded_network_parameters_list = []
            for layer_weights, biases in pairwise(shaped_weights):
                padded_weights_train_list = []
                for weights in layer_weights:
                    padded_weights = np.pad(weights, (int(np.floor((max_size-weights.shape[0])/2)), int(np.ceil((max_size-weights.shape[0])/2))), 'constant')
                    padded_weights_list.append(padded_weights)
                padded_biases = np.pad(biases, (int(np.floor((max_size-biases.shape[0])/2)), int(np.ceil((max_size-biases.shape[0])/2))), 'constant')
                padded_network_parameters_list.append(padded_biases)
                padded_network_parameters_list.extend(padded_weights_list)   
            X_data_list.append(padded_network_parameters_list)
        X_data = np.array(X_data_list)    

    elif version == 1 or version == 2: #each path from input bias to output bias combines in one sequence for biases and one sequence for weights per layer
        lambda_net_structure = list(flatten([n, lambda_network_layers, 1]))                    
        number_of_paths = reduce(lambda x, y: x * y, lambda_net_structure)

        X_data_list = []
        for shaped_weights in tqdm(shaped_weights_list):        
            network_parameters_sequence_list = np.array([]).reshape(number_of_paths, 0)    
            for layer_index, (weights, biases) in zip(range(1, len(lambda_net_structure)), pairwise(shaped_weights)):

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
                print("Number of non-unique rows: " + str(number_of_nonUnique_paths))
                print(network_parameters_sequence_list)

            X_data_list.append(network_parameters_sequence_list)
        X_data = np.array(X_data_list)          
        
        if version == 2: #transpose matrices (if false, no. columns == number of paths and no. rows = number of layers/length of path)
            X_data = np.transpose(X_data, (0, 2, 1))

    if lstm_layers != None and cnn_layers != None: #generate subsequences for cnn-lstm
        subsequences = 1 #for each bias+weights
        timesteps = X_train.shape[1]//subsequences

        X_data = X_data.reshape((X_data.shape[0], subsequences, timesteps, X_data.shape[2]))

    return X_data, X_data_flat