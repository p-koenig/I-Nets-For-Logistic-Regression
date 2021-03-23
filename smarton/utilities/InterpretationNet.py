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
        
    
    return_model = False
    n_jobs_inet_training = n_jobs
    if n_jobs==1 or (samples_list != None and len(samples_list) == 1) or (len(lambda_net_train_dataset_list) == 1 and samples_list == None) or use_gpu:
        n_jobs_inet_training = 1
        return_model = True
                        
    if samples_list == None: 
        
        results_list = Parallel(n_jobs=n_jobs_inet_training, 
                                verbose=11, 
                                backend='multiprocessing')(delayed(train_nn_and_pred)(lambda_net_train_dataset,
                                                                                       lambda_net_valid_dataset,
                                                                                       lambda_net_test_dataset,
                                                                                       current_jobs=n_jobs_inet_training,
                                                                                       callback_names=['early_stopping'],
                                                                                       return_model=return_model) for lambda_net_train_dataset,
                                                                                                                      lambda_net_valid_dataset,
                                                                                                                      lambda_net_test_dataset  in zip(lambda_net_train_dataset_list,
                                                                                                                                                      lambda_net_valid_dataset_list,
                                                                                                                                                      lambda_net_test_dataset_list))          

        history_list = [result[0] for result in results_list]

        scores_list = [result[1] for result in results_list]

        function_values_complete_list = [result[2] for result in results_list]
        function_values_valid_list = [function_values[0] for function_values in function_values_complete_list]
        function_values_test_list = [function_values[1] for function_values in function_values_complete_list]

        preds_list =  [result[3] for result in results_list]
        inet_preds_list = [preds[0] for preds in preds_list]
        inet_preds_valid_list = [inet_preds[0] for inet_preds in inet_preds_list]
        inet_preds_test_list = [inet_preds[1] for inet_preds in inet_preds_list]

        per_network_preds_list = [preds[1] for preds in preds_list]
        
        distrib_dict_list = [result[4] for result in results_list]

        if not nas:
            generate_history_plots(history_list, by='epochs')
            save_results(history_list, scores_list, by='epochs')    
        
        model_list = []
        if return_model:
            model_list = [result[5] for result in results_list]

    else:
        results_list = Parallel(n_jobs=n_jobs_inet_training, verbose=11, backend='multiprocessing')(delayed(train_nn_and_pred)(lambda_net_train_dataset.sample(samples),
                                                                                                      lambda_net_valid_dataset,
                                                                                                      lambda_net_test_dataset, 
                                                                                                      current_jobs=n_jobs_inet_training,
                                                                                                      callback_names=['early_stopping'],
                                                                                                      return_model=return_model) for samples in samples_list)     

        history_list = [result[0] for result in results_list]

        scores_list = [result[1] for result in results_list]

        function_values_complete_list = [result[2] for result in results_list]
        function_values_valid_list = [function_values[0] for function_values in function_values_complete_list]
        function_values_test_list = [function_values[1] for function_values in function_values_complete_list]

        preds_list =  [result[3] for result in results_list]
        inet_preds_list = [preds[0] for preds in preds_list]
        inet_preds_valid_list = [inet_preds[0] for inet_preds in inet_preds_list]
        inet_preds_test_list = [inet_preds[1] for inet_preds in inet_preds_list]

        per_network_preds_list = [preds[1] for preds in preds_list]


        distrib_dict_list = [result[4] for result in results_list]

        if not nas:
            generate_history_plots(history_list, by='samples')
            save_results(history_list, scores_list, by='samples')
    
        model_list = []
        if return_model:
            model_list = [result[5] for result in results_list]
            
    return (history_list, 
            scores_list, 
            
            function_values_complete_list, 
            function_values_valid_list, 
            function_values_test_list, 
            
            inet_preds_list, 
            inet_preds_valid_list, 
            inet_preds_test_list, 
            
            per_network_preds_list,
            
            distrib_dict_list,
            model_list)
        
    

    
    
    
    
#######################################################################################################################################################
######################################################################I-NET TRAINING###################################################################
#######################################################################################################################################################

def train_nn_and_pred(lambda_net_train_dataset,
                      lambda_net_valid_dataset,
                      lambda_net_test_dataset, 
                      current_jobs,
                      callback_names = [],
                      return_model = False ):       
   
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
    list_of_monomial_identifiers_numbers = np.array([list(monomial_identifiers) for monomial_identifiers in list_of_monomial_identifiers]).astype(float)
            
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
                metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers_numbers))
            
            valid_data = (X_valid, y_valid)
            y_train_model = y_train
        else: #based on comparison lstsq-lambda and predicted polynomial coefficients
            loss_function = inet_coefficient_loss_wrapper(inet_loss)
            
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers_numbers, base_model))            
            
            if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
                y_train_model = np.hstack((y_train, X_train_flat))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
            else:
                y_train_model = np.hstack((y_train, X_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid)))                                  
    else: #fv-based evaluation
        if evaluate_with_real_function: #based on in-loss fv calculation of real and predicted polynomial
            
            loss_function = inet_poly_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers_numbers)
            
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                metrics.append(inet_poly_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers_numbers))    
             
            valid_data = (X_valid, y_valid)
            y_train_model = y_train
        else: #in-loss prediction of lambda-nets
            
            #loss_function = inet_lambda_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers_numbers, base_model)
            loss_function = inet_lambda_fv_loss_wrapper(inet_loss, random_evaluation_dataset, list_of_monomial_identifiers_numbers, base_model, weights_structure, dims)
            metrics = []
            for inet_metric in list(flatten([inet_metrics, inet_loss])):
                metrics.append(inet_coefficient_loss_wrapper(inet_metric))            
                #metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers_numbers, base_model)) 
                metrics.append(inet_lambda_fv_loss_wrapper(inet_metric, random_evaluation_dataset, list_of_monomial_identifiers_numbers, base_model, weights_structure, dims)) 
            
            if convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL'):
                y_train_model = np.hstack((y_train, X_train_flat))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid_flat)))   
            else:
                y_train_model = np.hstack((y_train, X_train))   
                valid_data = (X_valid, np.hstack((y_valid, X_valid)))          
                 
        
    ############################## BUILD MODEL ###############################
    
    if nas:
        from tensorflow.keras.utils import CustomObjectScope
        with CustomObjectScope({'custom_loss': loss_function}):
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

            directory = './data/autokeras/' + nas_type + '_' + paths_dict['path_identifier_interpretation_net_data']

            auto_model = ak.AutoModel(inputs=input_node, 
                                outputs=output_node,
                                loss='custom_loss',
                                objective='val_loss',
                                overwrite=True,
                                tuner='hyperband',#"greedy",
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
            print(auto_model.evaluate(valid_data[0], valid_data[1]))
            print(model.evaluate(valid_data[0], valid_data[1]))

            y_valid_pred = model.predict(X_valid)[:,:interpretation_net_output_shape]
            y_test_pred = model.predict(X_test)[:,:interpretation_net_output_shape]


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
    
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)
    
    pred_list = [y_valid_pred, y_test_pred]
        
    ############################## PER NETWORK OPTIMIZATION ###############################
        
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    if False:
        
        per_network_hyperparams = {
            'optimizer': tf.keras.optimizers.RMSprop,
            'lr': 0.02,
            'max_steps': 500,
            'early_stopping': 10,
            'restarts': 3,
            'per_network_dataset_size': 5000,
        }

        list_of_monomial_identifiers_numbers = np.array([list(monomial_identifiers) for monomial_identifiers in list_of_monomial_identifiers]).astype(float)  

        if n_jobs != -1:
            n_jobs_per_network = min(n_jobs, os.cpu_count() // current_jobs)
        else: 
            n_jobs_per_network = os.cpu_count() // current_jobs - 1

        printing = True if n_jobs_per_network == 1 else False


        lambda_network_weights_list = np.array(lambda_net_test_dataset.weight_list)


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

        parallel_per_network = Parallel(n_jobs=n_jobs_per_network, verbose=1, backend='loky')

        per_network_optimization_polynomials = parallel_per_network(delayed(per_network_poly_optimization_tf)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                              lambda_network_weights, 
                                                                                                              list_of_monomial_identifiers_numbers, 
                                                                                                              config,
                                                                                                              optimizer = per_network_hyperparams['optimizer'],
                                                                                                              lr = per_network_hyperparams['lr'], 
                                                                                                              max_steps = per_network_hyperparams['max_steps'], 
                                                                                                              early_stopping = per_network_hyperparams['early_stopping'], 
                                                                                                              restarts = per_network_hyperparams['restarts'],
                                                                                                              printing = printing,
                                                                                                              return_error = True) for lambda_network_weights in lambda_network_weights_list)      

        del parallel_per_network
        
    else:    

        per_network_hyperparams = {
            'optimizer':  'Powell',
            'jac': 'fprime',
            'max_steps': 5000,#100,
            'restarts': 3,
            'per_network_dataset_size': 500,
        }

        list_of_monomial_identifiers_numbers = np.array([list(monomial_identifiers) for monomial_identifiers in list_of_monomial_identifiers]).astype(float)  

        if n_jobs != -1:
            n_jobs_per_network = min(n_jobs, os.cpu_count() // current_jobs)
        else: 
            n_jobs_per_network = os.cpu_count() // current_jobs - 1

        printing = True if n_jobs_per_network == 1 else False


        lambda_network_weights_list = np.array(lambda_net_test_dataset.weight_list)


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

        parallel_per_network = Parallel(n_jobs=n_jobs_per_network, verbose=1, backend='loky')

        result_list_per_network = parallel_per_network(delayed(per_network_poly_optimization_scipy)(per_network_hyperparams['per_network_dataset_size'], 
                                                                                                                  lambda_network_weights, 
                                                                                                                  list_of_monomial_identifiers_numbers, 
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
    
    pred_list = [pred_list, per_network_optimization_polynomials]
    
                
    ############################## FUNCTION VALUE CALCULATION ###############################
    
    lambda_test_data_preds_valid = lambda_net_valid_dataset.make_prediction_on_test_data()
    lambda_test_data_preds_test = lambda_net_test_dataset.make_prediction_on_test_data() 
              
    target_poly_test_data_fvs_valid = lambda_net_valid_dataset.return_target_poly_fvs_on_test_data()
    target_poly_test_data_fvs_test = lambda_net_test_dataset.return_target_poly_fvs_on_test_data() 
                
    lstsq_lambda_pred_polynomial_test_data_fvs_valid = lambda_net_valid_dataset.return_lstsq_lambda_pred_polynomial_fvs_on_test_data()
    lstsq_lambda_pred_polynomial_test_data_fvs_test = lambda_net_test_dataset.return_lstsq_lambda_pred_polynomial_fvs_on_test_data() 
             
    lstsq_target_polynomial_test_data_fvs_valid = lambda_net_valid_dataset.return_lstsq_target_polynomial_fvs_on_test_data()
    lstsq_target_polynomial_test_data_fvs_test = lambda_net_test_dataset.return_lstsq_target_polynomial_fvs_on_test_data() 
        
    inet_poly_test_data_fvs_valid = parallel_fv_calculation_from_polynomial(y_valid_pred, lambda_net_valid_dataset.X_test_data_list)
    inet_poly_test_data_fvs_test = parallel_fv_calculation_from_polynomial(y_test_pred, lambda_net_test_dataset.X_test_data_list) 
    
    per_network_optimization_poly_test_data_fvs_test = parallel_fv_calculation_from_polynomial(per_network_optimization_polynomials, lambda_net_test_dataset.X_test_data_list)     
    
    function_values_valid = [lambda_test_data_preds_valid, 
                            target_poly_test_data_fvs_valid, 
                            lstsq_lambda_pred_polynomial_test_data_fvs_valid, 
                            lstsq_target_polynomial_test_data_fvs_valid,
                            inet_poly_test_data_fvs_valid]
    
    function_values_test = [lambda_test_data_preds_test, 
                            target_poly_test_data_fvs_test, 
                            lstsq_lambda_pred_polynomial_test_data_fvs_test, 
                            lstsq_target_polynomial_test_data_fvs_test,
                            inet_poly_test_data_fvs_test,
                            per_network_optimization_poly_test_data_fvs_test]
    
    function_values = [function_values_valid, function_values_test]    
    
    
    ############################## EVALUATION ###############################
    
    #evaluate inet poly against target polynomial on fv-basis
    scores_inetPoly_VS_targetPoly_test_data_fv_valid, distrib_inetPoly_VS_targetPoly_test_data_fv_valid = evaluate_interpretation_net(y_valid_pred,
                                                                                   lambda_net_valid_dataset.target_polynomial_list, 
                                                                                   inet_poly_test_data_fvs_valid, 
                                                                                   target_poly_test_data_fvs_valid)  
    scores_inetPoly_VS_targetPoly_test_data_fv_test, distrib_inetPoly_VS_targetPoly_test_data_fv_test = evaluate_interpretation_net(y_test_pred, 
                                                                                  lambda_net_test_dataset.target_polynomial_list, 
                                                                                  inet_poly_test_data_fvs_test, 
                                                                                  target_poly_test_data_fvs_test)

    #evaluate inet poly against lambda-net preds on fv-basis
    scores_inetPoly_VS_predLambda_test_data_fv_valid, distrib_inetPoly_VS_predLambda_test_data_fv_valid = evaluate_interpretation_net(y_valid_pred, 
                                                                                   None, 
                                                                                   inet_poly_test_data_fvs_valid, 
                                                                                   lambda_test_data_preds_valid)
    scores_inetPoly_VS_predLambda_test_data_fv_test, distrib_inetPoly_VS_predLambda_test_data_fv_test = evaluate_interpretation_net(y_test_pred, 
                                                                                  None, 
                                                                                  inet_poly_test_data_fvs_test, 
                                                                                  lambda_test_data_preds_test)       
        
    #evaluate inet poly against lstsq target poly on fv-basis
    scores_inetPoly_VS_lstsqTarget_test_data_fv_valid, distrib_inetPoly_VS_lstsqTarget_test_data_fv_valid = evaluate_interpretation_net(y_valid_pred, 
                                                                                    lambda_net_valid_dataset.lstsq_target_polynomial_list, 
                                                                                    inet_poly_test_data_fvs_valid, 
                                                                                    lstsq_target_polynomial_test_data_fvs_valid)
    scores_inetPoly_VS_lstsqTarget_test_data_fv_test, distrib_inetPoly_VS_lstsqTarget_test_data_fv_test = evaluate_interpretation_net(y_test_pred, 
                                                                                   lambda_net_test_dataset.lstsq_target_polynomial_list, 
                                                                                   inet_poly_test_data_fvs_test, 
                                                                                   lstsq_target_polynomial_test_data_fvs_test)  

    #evaluate inet poly against lstsq lambda poly on fv-basis
    scores_inetPoly_VS_lstsqLambda_test_data_fv_valid, distrib_inetPoly_VS_lstsqLambda_test_data_fv_valid = evaluate_interpretation_net(y_valid_pred, 
                                                                                    lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                    inet_poly_test_data_fvs_valid, 
                                                                                    lstsq_lambda_pred_polynomial_test_data_fvs_valid)
    scores_inetPoly_VS_lstsqLambda_test_data_fv_test, distrib_inetPoly_VS_lstsqLambda_test_data_fv_test = evaluate_interpretation_net(y_test_pred, 
                                                                                   lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                   inet_poly_test_data_fvs_test, 
                                                                                   lstsq_lambda_pred_polynomial_test_data_fvs_test)     
      
    #evaluate lstsq lambda pred poly against lambda-net preds on fv-basis
    scores_lstsqLambda_VS_predLambda_test_data_fv_valid, distrib_lstsqLambda_VS_predLambda_test_data_fv_valid = evaluate_interpretation_net(lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                      None, 
                                                                                      lstsq_lambda_pred_polynomial_test_data_fvs_valid, 
                                                                                      lambda_test_data_preds_valid)
    scores_lstsqLambda_VS_predLambda_test_data_fv_test, distrib_lstsqLambda_VS_predLambda_test_data_fv_test = evaluate_interpretation_net(lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                     None, 
                                                                                     lstsq_lambda_pred_polynomial_test_data_fvs_test, 
                                                                                     lambda_test_data_preds_test)
    
    #evaluate lstsq lambda pred poly against lstsq target poly on fv-basis
    scores_lstsqLambda_VS_lstsqTarget_test_data_fv_valid, distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_valid = evaluate_interpretation_net(lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                       lambda_net_valid_dataset.lstsq_target_polynomial_list, 
                                                                                       lstsq_lambda_pred_polynomial_test_data_fvs_valid, 
                                                                                       lstsq_target_polynomial_test_data_fvs_valid)
    scores_lstsqLambda_VS_lstsqTarget_test_data_fv_test, distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_test = evaluate_interpretation_net(lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                      lambda_net_test_dataset.lstsq_target_polynomial_list, 
                                                                                      lstsq_lambda_pred_polynomial_test_data_fvs_test, 
                                                                                      lstsq_target_polynomial_test_data_fvs_test)    
    
    #evaluate lstsq lambda pred poly against target poly on fv-basis
    scores_lstsqLambda_VS_targetPoly_test_data_fv_valid, distrib_lstsqLambda_VS_targetPoly_test_data_fv_valid = evaluate_interpretation_net(lambda_net_valid_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                      lambda_net_valid_dataset.target_polynomial_list, 
                                                                                      lstsq_lambda_pred_polynomial_test_data_fvs_valid, 
                                                                                      target_poly_test_data_fvs_valid)
    scores_lstsqLambda_VS_targetPoly_test_data_fv_test, distrib_lstsqLambda_VS_targetPoly_test_data_fv_test = evaluate_interpretation_net(lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                     lambda_net_test_dataset.target_polynomial_list, 
                                                                                     lstsq_lambda_pred_polynomial_test_data_fvs_test, 
                                                                                     target_poly_test_data_fvs_test)    
    
    #evaluate lambda-net preds against lstsq target poly on fv-basis
    scores_predLambda_VS_lstsqTarget_test_data_fv_valid, distrib_predLambda_VS_lstsqTarget_test_data_fv_valid = evaluate_interpretation_net(None, 
                                                                                      lambda_net_valid_dataset.lstsq_target_polynomial_list, 
                                                                                      lambda_test_data_preds_valid, 
                                                                                      lstsq_target_polynomial_test_data_fvs_valid)
    scores_predLambda_VS_lstsqTarget_test_data_fv_test, distrib_predLambda_VS_lstsqTarget_test_data_fv_test = evaluate_interpretation_net(None, 
                                                                                     lambda_net_test_dataset.lstsq_target_polynomial_list, 
                                                                                     lambda_test_data_preds_test, 
                                                                                     lstsq_target_polynomial_test_data_fvs_test)
        
    #evaluate lambda-net preds against target poly on fv-basis
    scores_predLambda_VS_targetPoly_test_data_fv_valid, distrib_predLambda_VS_targetPoly_test_data_fv_valid = evaluate_interpretation_net(None, 
                                                                                     lambda_net_valid_dataset.target_polynomial_list, 
                                                                                     lambda_test_data_preds_valid, 
                                                                                     target_poly_test_data_fvs_valid)
    scores_predLambda_VS_targetPoly_test_data_fv_test, distrib_predLambda_VS_targetPoly_test_data_fv_test = evaluate_interpretation_net(None, 
                                                                                    lambda_net_test_dataset.target_polynomial_list, 
                                                                                    lambda_test_data_preds_test, 
                                                                                    target_poly_test_data_fvs_test)
      
    #evaluate lstsq target poly against target poly on fv-basis
    scores_lstsqTarget_VS_targetPoly_test_data_fv_valid, distrib_lstsqTarget_VS_targetPoly_test_data_fv_valid = evaluate_interpretation_net(lambda_net_valid_dataset.lstsq_target_polynomial_list, 
                                                                                      lambda_net_valid_dataset.target_polynomial_list, 
                                                                                      lstsq_target_polynomial_test_data_fvs_valid, 
                                                                                      target_poly_test_data_fvs_valid)
    scores_lstsqTarget_VS_targetPoly_test_data_fv_test, distrib_lstsqTarget_VS_targetPoly_test_data_fv_test = evaluate_interpretation_net(lambda_net_test_dataset.lstsq_target_polynomial_list, 
                                                                                     lambda_net_test_dataset.target_polynomial_list, 
                                                                                     lstsq_target_polynomial_test_data_fvs_test, 
                                                                                     target_poly_test_data_fvs_test)
        
    
    
    
    
    
    
    #evaluate per-network poly against target poly on fv-basis
    scores_perNetworkPoly_VS_targetPoly_test_data_fv_test, distrib_perNetworkPoly_VS_targetPoly_test_data_fv_test = evaluate_interpretation_net(per_network_optimization_polynomials, 
                                                                                  lambda_net_test_dataset.target_polynomial_list, 
                                                                                  per_network_optimization_poly_test_data_fvs_test, 
                                                                                  target_poly_test_data_fvs_test)    
    
    #evaluate per-network poly against inet poly on fv-basis
    scores_perNetworkPoly_VS_inetPoly_test_data_fv_test, distrib_perNetworkPoly_VS_inetPoly_test_data_fv_test = evaluate_interpretation_net(per_network_optimization_polynomials, 
                                                                                     y_test_pred, 
                                                                                     per_network_optimization_poly_test_data_fvs_test, 
                                                                                     target_poly_test_data_fvs_test)    
    
 
    
    #evaluate per-network poly against lambda-net preds on fv-basis
    scores_perNetworkPoly_VS_predLambda_test_data_fv_test, distrib_perNetworkPoly_VS_predLambda_test_data_fv_test = evaluate_interpretation_net(per_network_optimization_polynomials, 
                                                                                  None, 
                                                                                  per_network_optimization_poly_test_data_fvs_test, 
                                                                                  lambda_test_data_preds_test)         
    
   
    
    #evaluate per-network poly against lstsq target poly on fv-basis
    scores_perNetworkPoly_VS_lstsqTarget_test_data_fv_test, distrib_perNetworkPoly_VS_lstsqTarget_test_data_fv_test = evaluate_interpretation_net(per_network_optimization_polynomials, 
                                                                                   lambda_net_test_dataset.lstsq_target_polynomial_list, 
                                                                                   per_network_optimization_poly_test_data_fvs_test, 
                                                                                   lstsq_target_polynomial_test_data_fvs_test)       
    

          
    
    #evaluate per-network poly against lstsq lambda poly on fv-basis
    scores_perNetworkPoly_VS_lstsqLambda_test_data_fv_test, distrib_perNetworkPoly_VS_lstsqLambda_test_data_fv_test = evaluate_interpretation_net(per_network_optimization_polynomials, 
                                                                                   lambda_net_test_dataset.lstsq_lambda_pred_polynomial_list, 
                                                                                   per_network_optimization_poly_test_data_fvs_test, 
                                                                                   lstsq_lambda_pred_polynomial_test_data_fvs_test)    
    
    
    scores_dict = pd.DataFrame(data=[scores_inetPoly_VS_targetPoly_test_data_fv_valid, 
                                     scores_inetPoly_VS_targetPoly_test_data_fv_test, 
                                     scores_inetPoly_VS_predLambda_test_data_fv_valid,
                                     scores_inetPoly_VS_predLambda_test_data_fv_test,
                                     scores_inetPoly_VS_lstsqTarget_test_data_fv_valid,
                                     scores_inetPoly_VS_lstsqTarget_test_data_fv_test,
                                     scores_inetPoly_VS_lstsqLambda_test_data_fv_valid,
                                     scores_inetPoly_VS_lstsqLambda_test_data_fv_test,
                                     scores_lstsqLambda_VS_predLambda_test_data_fv_valid,
                                     scores_lstsqLambda_VS_predLambda_test_data_fv_test,
                                     scores_lstsqLambda_VS_lstsqTarget_test_data_fv_valid,
                                     scores_lstsqLambda_VS_lstsqTarget_test_data_fv_test,
                                     scores_lstsqLambda_VS_targetPoly_test_data_fv_valid,
                                     scores_lstsqLambda_VS_targetPoly_test_data_fv_test,
                                     scores_predLambda_VS_lstsqTarget_test_data_fv_valid,
                                     scores_predLambda_VS_lstsqTarget_test_data_fv_test,
                                     scores_predLambda_VS_targetPoly_test_data_fv_valid,
                                     scores_predLambda_VS_targetPoly_test_data_fv_test,
                                     scores_lstsqTarget_VS_targetPoly_test_data_fv_valid,
                                     scores_lstsqTarget_VS_targetPoly_test_data_fv_test,
                                     scores_perNetworkPoly_VS_targetPoly_test_data_fv_test,
                                     scores_perNetworkPoly_VS_inetPoly_test_data_fv_test,
                                     scores_perNetworkPoly_VS_predLambda_test_data_fv_test,
                                     scores_perNetworkPoly_VS_lstsqTarget_test_data_fv_test,
                                     scores_perNetworkPoly_VS_lstsqLambda_test_data_fv_test],
                               index=['inetPoly_VS_targetPoly_valid', 
                                      'inetPoly_VS_targetPoly_test', 
                                      'inetPoly_VS_predLambda_valid',
                                      'inetPoly_VS_predLambda_test',
                                      'inetPoly_VS_lstsqTarget_valid',
                                      'inetPoly_VS_lstsqTarget_test',
                                      'inetPoly_VS_lstsqLambda_valid',
                                      'inetPoly_VS_lstsqLambda_test',
                                      'lstsqLambda_VS_predLambda_valid',
                                      'lstsqLambda_VS_predLambda_test',
                                      'lstsqLambda_VS_lstsqTarget_valid',
                                      'lstsqLambda_VS_lstsqTarget_test',
                                      'lstsqLambda_VS_targetPoly_valid',
                                      'lstsqLambda_VS_targetPoly_test',
                                      'predLambda_VS_lstsqTarget_valid',
                                      'predLambda_VS_lstsqTarget_test',
                                      'predLambda_VS_targetPoly_valid',
                                      'predLambda_VS_targetPoly_test',
                                      'lstsqTarget_VS_targetPoly_valid',
                                      'lstsqTarget_VS_targetPoly_test',
                                      'perNetworkPoly_VS_targetPoly_test',
                                      'perNetworkPoly_VS_inetPoly_test',
                                      'perNetworkPoly_VS_predLambda_test',
                                      'perNetworkPoly_VS_lstsqTarget_test',
                                      'perNetworkPoly_VS_lstsqLambda_test'])
    
    mae_distrib_dict = pd.DataFrame(data=[distrib_inetPoly_VS_targetPoly_test_data_fv_valid['MAE'], 
                                     distrib_inetPoly_VS_targetPoly_test_data_fv_test['MAE'], 
                                     distrib_inetPoly_VS_predLambda_test_data_fv_valid['MAE'],
                                     distrib_inetPoly_VS_predLambda_test_data_fv_test['MAE'],
                                     distrib_inetPoly_VS_lstsqTarget_test_data_fv_valid['MAE'],
                                     distrib_inetPoly_VS_lstsqTarget_test_data_fv_test['MAE'],
                                     distrib_inetPoly_VS_lstsqLambda_test_data_fv_valid['MAE'],
                                     distrib_inetPoly_VS_lstsqLambda_test_data_fv_test['MAE'],
                                     distrib_lstsqLambda_VS_predLambda_test_data_fv_valid['MAE'],
                                     distrib_lstsqLambda_VS_predLambda_test_data_fv_test['MAE'],
                                     distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_valid['MAE'],
                                     distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_test['MAE'],
                                     distrib_lstsqLambda_VS_targetPoly_test_data_fv_valid['MAE'],
                                     distrib_lstsqLambda_VS_targetPoly_test_data_fv_test['MAE'],
                                     distrib_predLambda_VS_lstsqTarget_test_data_fv_valid['MAE'],
                                     distrib_predLambda_VS_lstsqTarget_test_data_fv_test['MAE'],
                                     distrib_predLambda_VS_targetPoly_test_data_fv_valid['MAE'],
                                     distrib_predLambda_VS_targetPoly_test_data_fv_test['MAE'],
                                     distrib_lstsqTarget_VS_targetPoly_test_data_fv_valid['MAE'],
                                     distrib_lstsqTarget_VS_targetPoly_test_data_fv_test['MAE'],
                                     distrib_perNetworkPoly_VS_targetPoly_test_data_fv_test['MAE'],
                                     distrib_perNetworkPoly_VS_inetPoly_test_data_fv_test['MAE'],
                                     distrib_perNetworkPoly_VS_predLambda_test_data_fv_test['MAE'],
                                     distrib_perNetworkPoly_VS_lstsqTarget_test_data_fv_test['MAE'],
                                     distrib_perNetworkPoly_VS_lstsqLambda_test_data_fv_test['MAE']],
                               index=['inetPoly_VS_targetPoly_valid', 
                                      'inetPoly_VS_targetPoly_test', 
                                      'inetPoly_VS_predLambda_valid',
                                      'inetPoly_VS_predLambda_test',
                                      'inetPoly_VS_lstsqTarget_valid',
                                      'inetPoly_VS_lstsqTarget_test',
                                      'inetPoly_VS_lstsqLambda_valid',
                                      'inetPoly_VS_lstsqLambda_test',
                                      'lstsqLambda_VS_predLambda_valid',
                                      'lstsqLambda_VS_predLambda_test',
                                      'lstsqLambda_VS_lstsqTarget_valid',
                                      'lstsqLambda_VS_lstsqTarget_test',
                                      'lstsqLambda_VS_targetPoly_valid',
                                      'lstsqLambda_VS_targetPoly_test',
                                      'predLambda_VS_lstsqTarget_valid',
                                      'predLambda_VS_lstsqTarget_test',
                                      'predLambda_VS_targetPoly_valid',
                                      'predLambda_VS_targetPoly_test',
                                      'lstsqTarget_VS_targetPoly_valid',
                                      'lstsqTarget_VS_targetPoly_test',
                                      'perNetworkPoly_VS_targetPoly_test',
                                      'perNetworkPoly_VS_inetPoly_test',
                                      'perNetworkPoly_VS_predLambda_test',
                                      'perNetworkPoly_VS_lstsqTarget_test',
                                      'perNetworkPoly_VS_lstsqLambda_test'])
    
    r2_distrib_dict = pd.DataFrame(data=[distrib_inetPoly_VS_targetPoly_test_data_fv_valid['R2'], 
                                     distrib_inetPoly_VS_targetPoly_test_data_fv_test['R2'], 
                                     distrib_inetPoly_VS_predLambda_test_data_fv_valid['R2'],
                                     distrib_inetPoly_VS_predLambda_test_data_fv_test['R2'],
                                     distrib_inetPoly_VS_lstsqTarget_test_data_fv_valid['R2'],
                                     distrib_inetPoly_VS_lstsqTarget_test_data_fv_test['R2'],
                                     distrib_inetPoly_VS_lstsqLambda_test_data_fv_valid['R2'],
                                     distrib_inetPoly_VS_lstsqLambda_test_data_fv_test['R2'],
                                     distrib_lstsqLambda_VS_predLambda_test_data_fv_valid['R2'],
                                     distrib_lstsqLambda_VS_predLambda_test_data_fv_test['R2'],
                                     distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_valid['R2'],
                                     distrib_lstsqLambda_VS_lstsqTarget_test_data_fv_test['R2'],
                                     distrib_lstsqLambda_VS_targetPoly_test_data_fv_valid['R2'],
                                     distrib_lstsqLambda_VS_targetPoly_test_data_fv_test['R2'],
                                     distrib_predLambda_VS_lstsqTarget_test_data_fv_valid['R2'],
                                     distrib_predLambda_VS_lstsqTarget_test_data_fv_test['R2'],
                                     distrib_predLambda_VS_targetPoly_test_data_fv_valid['R2'],
                                     distrib_predLambda_VS_targetPoly_test_data_fv_test['R2'],
                                     distrib_lstsqTarget_VS_targetPoly_test_data_fv_valid['R2'],
                                     distrib_lstsqTarget_VS_targetPoly_test_data_fv_test['R2'],
                                     distrib_perNetworkPoly_VS_targetPoly_test_data_fv_test['R2'],
                                     distrib_perNetworkPoly_VS_inetPoly_test_data_fv_test['R2'],
                                     distrib_perNetworkPoly_VS_predLambda_test_data_fv_test['R2'],
                                     distrib_perNetworkPoly_VS_lstsqTarget_test_data_fv_test['R2'],
                                     distrib_perNetworkPoly_VS_lstsqLambda_test_data_fv_test['R2']],
                               index=['inetPoly_VS_targetPoly_valid', 
                                      'inetPoly_VS_targetPoly_test', 
                                      'inetPoly_VS_predLambda_valid',
                                      'inetPoly_VS_predLambda_test',
                                      'inetPoly_VS_lstsqTarget_valid',
                                      'inetPoly_VS_lstsqTarget_test',
                                      'inetPoly_VS_lstsqLambda_valid',
                                      'inetPoly_VS_lstsqLambda_test',
                                      'lstsqLambda_VS_predLambda_valid',
                                      'lstsqLambda_VS_predLambda_test',
                                      'lstsqLambda_VS_lstsqTarget_valid',
                                      'lstsqLambda_VS_lstsqTarget_test',
                                      'lstsqLambda_VS_targetPoly_valid',
                                      'lstsqLambda_VS_targetPoly_test',
                                      'predLambda_VS_lstsqTarget_valid',
                                      'predLambda_VS_lstsqTarget_test',
                                      'predLambda_VS_targetPoly_valid',
                                      'predLambda_VS_targetPoly_test',
                                      'lstsqTarget_VS_targetPoly_valid',
                                      'lstsqTarget_VS_targetPoly_test',
                                      'perNetworkPoly_VS_targetPoly_test',
                                      'perNetworkPoly_VS_inetPoly_test',
                                      'perNetworkPoly_VS_predLambda_test',
                                      'perNetworkPoly_VS_lstsqTarget_test',
                                      'perNetworkPoly_VS_lstsqLambda_test'])    
    
    distrib_dicts = {'MAE': mae_distrib_dict, 
                     'R2': r2_distrib_dict}
    
    if return_model or n_jobs==1:
        return history, scores_dict, function_values, pred_list, distrib_dicts, model         
    else: 
        return history, scores_dict, function_values, pred_list, distrib_dicts       
    
    
    
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
    
    epochs_save_range_lambda = range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda) if each_epochs_save_lambda == 1 else range(epoch_start//each_epochs_save_lambda, epochs_lambda//each_epochs_save_lambda+1) if multi_epoch_analysis else range(1,2)


    if samples_list == None:
        x_axis_steps = [(i+1)*each_epochs_save_lambda if each_epochs_save_lambda==1 else i*each_epochs_save_lambda if i > 1 else each_epochs_save_lambda if i==1 else 1 for i in epochs_save_range_lambda]
        x_max = epochs_lambda
    else:
        x_axis_steps = samples_list
        x_max = samples_list[-1]

    if evaluate_with_real_function:
        #Plot Polynom, lamdba net, and Interpration net
        length_plt = len(plot_metric_list)
        if length_plt >= 2:
            fig, ax = plt.subplots(length_plt//2, 2, figsize=(30,20))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15,10))

        for index, metric in enumerate(plot_metric_list):

            inetPoly_VS_targetPoly_test = []
            perNetworkPoly_VS_targetPoly_test = []
            #inetPoly_VS_predLambda_test = []
            #inetPoly_VS_lstsqTarget_test = []
            #inetPoly_VS_lstsqLambda_test = []
            #lstsqLambda_VS_predLambda_test = []
            #lstsqLambda_VS_lstsqTarget_test = []
            lstsqLambda_VS_targetPoly_test = []
            #predLambda_VS_lstsqTarget_test = []
            predLambda_VS_targetPoly_test = []
            lstsqTarget_VS_targetPoly_test = []

            for scores in scores_list:
                inetPoly_VS_targetPoly_test.append(scores[metric].loc['inetPoly_VS_targetPoly_test'])
                perNetworkPoly_VS_targetPoly_test.append(scores[metric].loc['perNetworkPoly_VS_targetPoly_test'])
                predLambda_VS_targetPoly_test.append(scores[metric].loc['predLambda_VS_targetPoly_test'])
                lstsqLambda_VS_targetPoly_test.append(scores[metric].loc['lstsqLambda_VS_targetPoly_test'])     
                lstsqTarget_VS_targetPoly_test.append(scores[metric].loc['lstsqTarget_VS_targetPoly_test'])

            plot_df = pd.DataFrame(data=np.vstack([inetPoly_VS_targetPoly_test, perNetworkPoly_VS_targetPoly_test, predLambda_VS_targetPoly_test, lstsqLambda_VS_targetPoly_test, lstsqTarget_VS_targetPoly_test]).T, 
                                   index=x_axis_steps,
                                   columns=['inetPoly_VS_targetPoly_test', 'perNetworkPoly_VS_targetPoly_test', 'predLambda_VS_targetPoly_test', 'lstsqLambda_VS_targetPoly_test', 'lstsqTarget_VS_targetPoly_test'])
            
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
                
            p.set_yticklabels(p.get_yticks(), size = 20)
            p.set_xticklabels(p.get_xticks(), size = 20)        

        location = './data/plotting/'
        folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
        if samples_list == None:
            file = 'multi_epoch_REAL' + '.pdf'
        else:
            file = 'sample_list' + '-'.join([str(samples_list[0]), str(samples_list[-1])]) +'_REAL' + '.pdf'

        path = location + folder + file

        plt.savefig(path, format='pdf')
        plt.show()

    else:
        #Plot Polynom, lamdba net, and Interpration net
        length_plt = len(plot_metric_list)
        if length_plt >= 2:
            fig, ax = plt.subplots(length_plt//2, 2, figsize=(30,20))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15,10))
        for index, metric in enumerate(plot_metric_list):

            #inetPoly_VS_targetPoly_test = []
            inetPoly_VS_predLambda_test = []
            #inetPoly_VS_lstsqTarget_test = []
            inetPoly_VS_lstsqLambda_test = []
            perNetworkPoly_VS_predLambda_test = []
            perNetworkPoly_VS_lstsqLambda_test = []
            lstsqLambda_VS_predLambda_test = []
            #lstsqLambda_VS_lstsqTarget_test = []
            #lstsqLambda_VS_targetPoly_test = []
            #predLambda_VS_lstsqTarget_test = []
            predLambda_VS_targetPoly_test = []
            #lstsqTarget_VS_targetPoly_test = []

            for scores in scores_list:
                inetPoly_VS_lstsqLambda_test.append(scores[metric].loc['inetPoly_VS_lstsqLambda_test'])
                inetPoly_VS_predLambda_test.append(scores[metric].loc['inetPoly_VS_predLambda_test'])
                perNetworkPoly_VS_lstsqLambda_test.append(scores[metric].loc['perNetworkPoly_VS_lstsqLambda_test'])
                perNetworkPoly_VS_predLambda_test.append(scores[metric].loc['perNetworkPoly_VS_predLambda_test'])                
                lstsqLambda_VS_predLambda_test.append(scores[metric].loc['lstsqLambda_VS_predLambda_test'])     
                predLambda_VS_targetPoly_test.append(scores[metric].loc['predLambda_VS_targetPoly_test'])     

            plot_df = pd.DataFrame(data=np.vstack([inetPoly_VS_predLambda_test, inetPoly_VS_lstsqLambda_test, perNetworkPoly_VS_predLambda_test, perNetworkPoly_VS_lstsqLambda_test, lstsqLambda_VS_predLambda_test, predLambda_VS_targetPoly_test]).T, 
                                   index=x_axis_steps,
                                   columns=['inetPoly_VS_predLambda_test', 'inetPoly_VS_lstsqLambda_test', 'perNetworkPoly_VS_predLambda_test', 'perNetworkPoly_VS_lstsqLambda_test', 'lstsqLambda_VS_predLambda_test', 'predLambda_VS_targetPoly_test'])

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
                
            p.set_yticklabels(p.get_yticks(), size = 20)
            p.set_xticklabels(p.get_xticks(), size = 20)  

        location = './data/plotting/'
        folder = paths_dict['path_identifier_interpretation_net_data'] + '/'
        if samples_list == None:
            file = 'multi_epoch_MODEL' + '.pdf'
        else: 
            file = 'sample_list' + '-'.join([str(samples_list[0]), str(samples_list[-1])]) +'_MODEL' + '.pdf'

        path = location + folder + file

        plt.savefig(path, format='pdf')
        plt.show()



def plot_and_save_single_polynomial_prediction_evaluation(lambda_net_test_dataset_list, function_values_test_list, inet_preds_test_list, per_network_preds_list, rand_index=1, plot_type=2):
    
    paths_dict = generate_paths(path_type = 'interpretation_net')
    
    polynomial_target = lambda_net_test_dataset_list[-1].target_polynomial_list[rand_index]
    polynomial_lstsq_target = lambda_net_test_dataset_list[-1].lstsq_target_polynomial_list[rand_index]
    polynomial_lstsq_lambda = lambda_net_test_dataset_list[-1].lstsq_lambda_pred_polynomial_list[rand_index]
    polynomial_inet = inet_preds_test_list[-1][rand_index]
    polynomial_per_network_opt = per_network_preds_list[-1][rand_index]
    
    
    polynomial_target_string = get_sympy_string_from_coefficients(polynomial_target, force_complete_poly_representation=True, round_digits=4)
    polynomial_lstsq_target_string = get_sympy_string_from_coefficients(polynomial_lstsq_target, force_complete_poly_representation=True, round_digits=4)
    polynomial_lstsq_lambda_string = get_sympy_string_from_coefficients(polynomial_lstsq_lambda, force_complete_poly_representation=True, round_digits=4)
    polynomial_inet_string = get_sympy_string_from_coefficients(polynomial_inet, round_digits=4)
    polynomial_per_network_opt_string = get_sympy_string_from_coefficients(polynomial_per_network_opt, round_digits=4)
    
    #print('Target Poly:')
    #print_polynomial_from_coefficients(polynomial_target, force_complete_poly_representation=True, round_digits=4)
    #print('LSTSQ Target Poly:')
    #print_polynomial_from_coefficients(polynomial_lstsq_target, force_complete_poly_representation=True, round_digits=4)
    #print('LSTSQ Lambda Poly:')
    #print_polynomial_from_coefficients(polynomial_lstsq_lambda, force_complete_poly_representation=True, round_digits=4)
    #print('I-Net Poly:')
    #print_polynomial_from_coefficients(polynomial_inet, round_digits=4)    
    
    real_poly_fvs = function_values_test_list[-1][1][rand_index]
    lstsq_target_poly = function_values_test_list[-1][3][rand_index]
    lambda_model_preds = function_values_test_list[-1][0][rand_index].ravel()
    lstsq_lambda_preds_poly = function_values_test_list[-1][2][rand_index]
    inet_poly_fvs = function_values_test_list[-1][4][rand_index]  
    per_network_opt_poly_fvs = function_values_test_list[-1][5][rand_index]  
    lambda_train_data = lambda_net_test_dataset_list[-1].y_test_data_list[rand_index].ravel()
    
    x_vars = ['x' + str(i) for i in range(1, n+1)]

    columns = x_vars.copy()
    columns.append('FVs')

    columns_single = x_vars.copy()

    eval_size_plot = inet_poly_fvs.shape[0]
    lambda_train_data_size = lambda_train_data.shape[0]
    vars_plot = lambda_net_test_dataset_list[-1].X_test_data_list[rand_index]
    
    if evaluate_with_real_function:
        columns_single.extend(['Target Poly FVs', 'Lambda Train Data', 'LSTSQ Target Poly FVs', 'I-Net Poly FVs', 'Per Network Opt Poly FVs', 'Lambda Model Preds', 'LSTSQ Lambda Poly FVs'])
        plot_data_single = pd.DataFrame(data=np.column_stack([vars_plot, real_poly_fvs, lambda_train_data, lstsq_target_poly, inet_poly_fvs, per_network_opt_poly_fvs, lambda_model_preds, lstsq_lambda_preds_poly]), columns=columns_single)
        preds_plot_all = np.vstack([real_poly_fvs, lambda_train_data, lstsq_target_poly, inet_poly_fvs, per_network_opt_poly_fvs, lambda_model_preds, lstsq_lambda_preds_poly]).ravel()
        vars_plot_all_preds = np.vstack([vars_plot for i in range(len(columns_single[n:]))])

        real_poly_fvs_str = np.array(['Target Poly FVs' for i in range(eval_size_plot)])
        lambda_train_data_str = np.array(['Lambda Train Data' for i in range(lambda_train_data_size)])
        lstsq_target_poly_str = np.array(['LSTSQ Target Poly FVs' for i in range(eval_size_plot)])
        inet_poly_fvs_str = np.array(['I-Net Poly FVs' for i in range(eval_size_plot)])
        per_network_opt_poly_fvs_str = np.array(['Per Network Opt Poly FVs' for i in range(eval_size_plot)])
        lambda_model_preds_str = np.array(['Lambda Model Preds' for i in range(eval_size_plot)])
        lstsq_lambda_preds_poly_str = np.array(['LSTSQ Lambda Poly FVs' for i in range(eval_size_plot)])

        identifier = np.concatenate([real_poly_fvs_str, lambda_train_data_str, lstsq_target_poly_str, inet_poly_fvs_str, per_network_opt_poly_fvs_str, lambda_model_preds_str, lstsq_lambda_preds_poly_str])
    else:
        columns_single.extend(['Lambda Model Preds', 'LSTSQ Lambda Poly FVs', 'I-Net Poly FVs', 'Per Network Opt Poly FVs', 'Target Poly FVs', 'Lambda Train Data', 'LSTSQ Target Poly FVs'])
        plot_data_single = pd.DataFrame(data=np.column_stack([vars_plot, lambda_model_preds, lstsq_lambda_preds_poly, inet_poly_fvs, per_network_opt_poly_fvs, real_poly_fvs, lambda_train_data, lstsq_target_poly]), columns=columns_single)
        preds_plot_all = np.vstack([lambda_model_preds, lstsq_lambda_preds_poly, inet_poly_fvs, per_network_opt_poly_fvs, real_poly_fvs, lambda_train_data, lstsq_target_poly]).ravel()
        vars_plot_all_preds = np.vstack([vars_plot for i in range(len(columns_single[n:]))])

        lambda_model_preds_str = np.array(['Lambda Model Preds' for i in range(eval_size_plot)])
        lstsq_lambda_preds_poly_str = np.array(['LSTSQ Lambda Poly FVs' for i in range(eval_size_plot)])        
        inet_poly_fvs_str = np.array(['I-Net Poly FVs' for i in range(eval_size_plot)])
        per_network_opt_poly_fvs_str = np.array(['Per Network Opt Poly FVs' for i in range(eval_size_plot)])
        real_poly_fvs_str = np.array(['Target Poly FVs' for i in range(eval_size_plot)])
        lambda_train_data_str = np.array(['Lambda Train Data' for i in range(lambda_train_data_size)])
        lstsq_target_poly_str = np.array(['LSTSQ Target Poly FVs' for i in range(eval_size_plot)])

        identifier = np.concatenate([lambda_model_preds_str, lstsq_lambda_preds_poly_str, inet_poly_fvs_str, per_network_opt_poly_fvs_str, real_poly_fvs_str, lambda_train_data_str, lstsq_target_poly_str])
        
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
        if evaluate_with_real_function:
            file = 'pp3in1_REAL_' + str(rand_index) + '.pdf'        
        else:
            file = 'pp3in1_PRED_' + str(rand_index) + '.pdf'            
        
    elif plot_type == 2:

        pp = sns.pairplot(data=plot_data,
                          #kind='reg',
                          hue='Identifier',
                          #y_vars=['FVs'],
                          #x_vars=x_vars, 
                          height=10//n)
        
        if evaluate_with_real_function:        
            file = 'pp3in1_extended_REAL_' + str(rand_index) + '.pdf'        
        else:
            file = 'pp3in1_extended_PRED_' + str(rand_index) + '.pdf'  
        
    elif plot_type == 3:
        
        pp = sns.pairplot(data=plot_data_single,
                          #kind='reg',
                          y_vars=columns_single[n:],
                          x_vars=x_vars, 
                          height=3,
                          aspect=3)

        if evaluate_with_real_function:        
            file = 'pp1_REAL_' + str(rand_index) + '.pdf'        
        else:
            file = 'pp1_PRED_' + str(rand_index) + '.pdf'            
        
    path = location + folder + file
    pp.savefig(path, format='pdf')
    plt.show()    
    
    if evaluate_with_real_function:
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
        
        
    else:
        lambda_model_preds_VS_lstsq_lambda_preds_poly_mae = mean_absolute_error(lambda_model_preds, lstsq_lambda_preds_poly)
        lambda_model_preds_VS_lstsq_lambda_preds_poly_r2 = r2_score(lambda_model_preds, lstsq_lambda_preds_poly)
        
        lambda_model_preds_VS_inet_poly_mae = mean_absolute_error(lambda_model_preds, inet_poly_fvs)
        lambda_model_preds_VS_inet_poly_r2 = r2_score(lambda_model_preds, inet_poly_fvs)
        
        lambda_model_preds_VS_perNet_poly_mae = mean_absolute_error(lambda_model_preds, per_network_opt_poly_fvs)
        lambda_model_preds_VS_perNet_poly_r2 = r2_score(lambda_model_preds, per_network_opt_poly_fvs)
        
        lambda_model_preds_VS_real_poly_mae = mean_absolute_error(lambda_model_preds, real_poly_fvs)
        lambda_model_preds_VS_real_poly_r2 = r2_score(lambda_model_preds, real_poly_fvs)
        
        lambda_model_preds_VS_lstsq_target_poly_mae = mean_absolute_error(lambda_model_preds, lstsq_target_poly)
        lambda_model_preds_VS_lstsq_target_poly_r2 = r2_score(lambda_model_preds, lstsq_target_poly)    
        
        from prettytable import PrettyTable
    
        tab = PrettyTable()

        tab.field_names = ["Comparison", "MAE", "R2-Score", "Poly 1", "Poly 2"]
        tab._max_width = {"Poly 1" : 50, "Poly 2" : 50}
        
        tab.add_row(["Lambda Preds \n vs. \n LSTSQ Lambda Preds Poly \n", lambda_model_preds_VS_lstsq_lambda_preds_poly_mae, lambda_model_preds_VS_lstsq_lambda_preds_poly_r2, '-', polynomial_lstsq_lambda_string])
        tab.add_row(["Lambda Preds \n vs. \n I-Net Poly \n", lambda_model_preds_VS_inet_poly_mae, lambda_model_preds_VS_inet_poly_r2, '-', polynomial_inet_string])
        tab.add_row(["Lambda Preds \n vs. \n Per Network Opt Poly \n", lambda_model_preds_VS_perNet_poly_mae, lambda_model_preds_VS_perNet_poly_r2, '-', polynomial_per_network_opt_string])
        tab.add_row(["Lambda Preds \n vs. \n Target Poly \n", lambda_model_preds_VS_real_poly_mae, lambda_model_preds_VS_real_poly_r2, '-', polynomial_target_string])
        tab.add_row(["Lambda Preds \n vs. \n LSTSQ Target Poly \n", lambda_model_preds_VS_lstsq_target_poly_mae, lambda_model_preds_VS_lstsq_target_poly_r2, '-', polynomial_lstsq_target_string])
        
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