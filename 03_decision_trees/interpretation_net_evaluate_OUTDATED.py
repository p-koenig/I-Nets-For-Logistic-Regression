#!/usr/bin/env python
# coding: utf-8

# # Inerpretation-Net Training

# ## Specitication of Experiment Settings

# In[1]:

#######################################################################################################################################
##################################################### IMPORT LIBRARIES ################################################################
#######################################################################################################################################
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from itertools import product       
from tqdm.notebook import tqdm
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import timeit
import psutil

from functools import reduce
from more_itertools import random_product 
from sklearn.preprocessing import Normalizer

import sys
import shutil

from copy import deepcopy
import math
import random 


import time
from datetime import datetime
from collections.abc import Iterable


from joblib import Parallel, delayed

from scipy.integrate import quad

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold, ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score, log_loss
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

#import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


import tensorflow.keras.backend as K
from livelossplot import PlotLossesKerasTF
#from keras_tqdm import TQDMNotebookCallback

from matplotlib import pyplot as plt
import seaborn as sns

from IPython.display import Image
from IPython.display import display, Math, Latex, clear_output

from prettytable import PrettyTable

from contextlib import redirect_stdout

from utilities.InterpretationNet import *
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *
from utilities.DecisionTree_BASIC import *
                   
#@ray.remote(num_cpus=3)              
def run_evaluation(parameter_setting, evaluate_all_datasets=True):
    
    if parameter_setting['dt_setting_experiment'] == 1:
        parameter_setting['dt_type_experiment'] = 'vanilla'
        parameter_setting['decision_sparsity_experiment'] = 1     
        parameter_setting['function_representation_type_experiment'] = 3
    if parameter_setting['dt_setting_experiment'] == 2:
        parameter_setting['dt_type_experiment'] = 'SDT'
        parameter_setting['decision_sparsity_experiment'] = 1     
        parameter_setting['function_representation_type_experiment'] = 3
    if parameter_setting['dt_setting_experiment'] == 3:
        parameter_setting['dt_type_experiment'] = 'SDT'
        parameter_setting['decision_sparsity_experiment'] = -1      
        parameter_setting['function_representation_type_experiment'] = 1
    
    filename = './running_evaluations/04_interpretation_net_script-' + str(parameter_setting['dt_type_experiment']) + str(parameter_setting['decision_sparsity_experiment']) + '_n' + str(parameter_setting['variable_number_experiment']) + '_d' + str(parameter_setting['depth_experiment']) +  '.txt'
    
    with open(filename, 'a+') as f:
        with redirect_stdout(f):    
            #######################################################################################################################################
            ###################################################### CONFIG FILE ####################################################################
            #######################################################################################################################################
            sleep_time = 0 #minutes

            config = {
                'function_family': {
                    'maximum_depth': parameter_setting['depth_experiment'],
                    'beta': 1,
                    'decision_sparsity': parameter_setting['decision_sparsity_experiment'],
                    'fully_grown': True,    
                    'dt_type': parameter_setting['dt_type_experiment'], #'SDT', 'vanilla'
                },
                'data': {
                    'number_of_variables': parameter_setting['variable_number_experiment'], 
                    'num_classes': 2,
                    'categorical_indices': [],

                    'dt_type_train': 'vanilla', # (None, 'vanilla', 'SDT')
                    'maximum_depth_train': 5, #None or int
                    'decision_sparsity_train': 1, #None or int

                    'function_generation_type': parameter_setting['function_generation_type_experiment'],# 'make_classification', 'make_classification_trained', 'random_decision_tree', 'random_decision_tree_trained'
                    'objective': 'classification', # 'regression'

                    'x_max': 1,
                    'x_min': 0,
                    'x_distrib': 'uniform', #'normal', 'uniform',       

                    'lambda_dataset_size': 5000, #number of samples per function
                    #'number_of_generated_datasets': 10000,

                    'noise_injected_level': 0, 
                    'noise_injected_type': 'flip_percentage', # '' 'normal' 'uniform' 'normal_range' 'uniform_range'
                }, 
                'lambda_net': {
                    'epochs_lambda': 1000,
                    'early_stopping_lambda': True, 
                    'early_stopping_min_delta_lambda': 1e-2,
                    'batch_lambda': 64,
                    'dropout_lambda': 0,
                    'lambda_network_layers': [128],
                    'optimizer_lambda': 'adam',
                    'loss_lambda': 'binary_crossentropy', #categorical_crossentropy

                    'number_of_lambda_weights': None,

                    'number_initializations_lambda': 1, 

                    'number_of_trained_lambda_nets': 10000,
                },     

                'i_net': {
                    'dense_layers': parameter_setting['i_net_structure_experiment'],
                    'convolution_layers': None,
                    'lstm_layers': None,
                    'dropout': parameter_setting['i_net_dropout_experiment'],
                    'additional_hidden': False,

                    'optimizer': 'adam', #adam
                    'learning_rate': parameter_setting['i_net_learning_rate_experiment'],
                    'loss': parameter_setting['i_net_loss_experiment'], #mse; binary_crossentropy; 'binary_accuracy'
                    'metrics': ['binary_crossentropy', 'soft_binary_crossentropy', 'binary_accuracy'], #soft_ or _penalized

                    'epochs': 500, 
                    'early_stopping': True,
                    'batch_size': 256,

                    'interpretation_dataset_size': 10000,

                    'test_size': 3, #Float for fraction, Int for number 0

                    'function_representation_type': parameter_setting['function_representation_type_experiment'], # 1=standard representation; 2=sparse representation with classification for variables; 3=softmax to select classes (n top probabilities)
                    'normalize_lambda_nets': False,

                    'optimize_decision_function': True, #False
                    'function_value_loss': True, #False
                    'soft_labels': False,

                    'data_reshape_version': None, #default to 2 options:(None, 0,1 2,3) #3=autoencoder dimensionality reduction

                    'nas': False,
                    'nas_type': 'SEQUENTIAL', #options:(None, 'SEQUENTIAL', 'CNN', 'LSTM', 'CNN-LSTM', 'CNN-LSTM-parallel')      
                    'nas_trials': 100,
                },    

                'evaluation': {   
                    #'inet_holdout_seed_evaluation': False,

                    'random_evaluation_dataset_size': 500, 
                    'per_network_optimization_dataset_size': 5000,

                    'sklearn_dt_benchmark': False,
                    'sdt_benchmark': False,

                    'different_eval_data': False,

                    'eval_data_description': {
                        ######### data #########
                        'eval_data_function_generation_type': 'make_classification',
                        'eval_data_lambda_dataset_size': 5000, #number of samples per function
                        'eval_data_noise_injected_level': 0, 
                        'eval_data_noise_injected_type': 'flip_percentage', # '' 'normal' 'uniform' 'normal_range' 'uniform_range'     
                        ######### lambda_net #########
                        'eval_data_number_of_trained_lambda_nets': 100,
                        ######### i_net #########
                        'eval_data_interpretation_dataset_size': 100,

                    }

                },    

                'computation':{
                    'load_model': False,
                    'n_jobs': 7,
                    'use_gpu': False,
                    'gpu_numbers': '2',
                    'RANDOM_SEED': 42,   
                    'verbosity': 0
                }
            }


            # ## Imports

            # In[2]:


            #######################################################################################################################################
            ########################################### IMPORT GLOBAL VARIABLES FROM CONFIG #######################################################
            #######################################################################################################################################
            globals().update(config['function_family'])
            globals().update(config['data'])
            globals().update(config['lambda_net'])
            globals().update(config['i_net'])
            globals().update(config['evaluation'])
            globals().update(config['computation'])


            #######################################################################################################################################
            ################################################### VARIABLE ADJUSTMENTS ##############################################################
            #######################################################################################################################################

            config['i_net']['data_reshape_version'] = 2 if data_reshape_version == None and (convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL')) else data_reshape_version
            config['function_family']['decision_sparsity'] = config['function_family']['decision_sparsity'] if config['function_family']['decision_sparsity'] != -1 else config['data']['number_of_variables'] 

            #######################################################################################################################################
            ###################################################### SET VARIABLES + DESIGN #########################################################
            #######################################################################################################################################

            #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers if use_gpu else ''
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' if use_gpu else ''

            #os.environ['XLA_FLAGS'] =  '--xla_gpu_cuda_data_dir=/usr/local/cuda-10.1'

            #os.environ['XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
            #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

            os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-11.4' if use_gpu else ''#-10.1' #--xla_gpu_cuda_data_dir=/usr/local/cuda, 
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 ,--tf_xla_enable_xla_devices' if use_gpu else ''#'--tf_xla_auto_jit=2' #, --tf_xla_enable_xla_devices


            sns.set_style("darkgrid")

            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            if int(tf.__version__[0]) >= 2:
                tf.random.set_seed(RANDOM_SEED)
            else:
                tf.set_random_seed(RANDOM_SEED)


            pd.set_option('display.float_format', lambda x: '%.3f' % x)
            pd.set_option('display.max_columns', 200)
            np.set_printoptions(threshold=200)
            np.set_printoptions(suppress=True)


            # In[6]:


            #######################################################################################################################################
            ########################################### IMPORT GLOBAL VARIABLES FROM CONFIG #######################################################
            #######################################################################################################################################
            globals().update(config['function_family'])
            globals().update(config['data'])
            globals().update(config['lambda_net'])
            globals().update(config['evaluation'])
            globals().update(config['computation'])


            # In[7]:




            #######################################################################################################################################
            ####################################################### CONFIG ADJUSTMENTS ############################################################
            #######################################################################################################################################

            config['lambda_net']['number_of_lambda_weights'] = get_number_of_lambda_net_parameters(lambda_network_layers, number_of_variables, num_classes)
            config['function_family']['basic_function_representation_length'] = get_number_of_function_parameters(dt_type, maximum_depth, number_of_variables, num_classes)
            config['function_family']['function_representation_length'] = ( 
                   #((2 ** maximum_depth - 1) * decision_sparsity) * 2 + (2 ** maximum_depth - 1) + (2 ** maximum_depth) * num_classes  if function_representation_type == 1 and dt_type == 'SDT'
                   (2 ** maximum_depth - 1) * (number_of_variables + 1) + (2 ** maximum_depth) * num_classes if function_representation_type == 1 and dt_type == 'SDT'
              else (2 ** maximum_depth - 1) * decision_sparsity + (2 ** maximum_depth - 1) + ((2 ** maximum_depth - 1)  * decision_sparsity * number_of_variables) + (2 ** maximum_depth) * num_classes if function_representation_type == 2 and dt_type == 'SDT'
              else ((2 ** maximum_depth - 1) * decision_sparsity) * 2 + (2 ** maximum_depth)  if function_representation_type == 1 and dt_type == 'vanilla'
              else (2 ** maximum_depth - 1) * decision_sparsity + ((2 ** maximum_depth - 1)  * decision_sparsity * number_of_variables) + (2 ** maximum_depth) if function_representation_type == 2 and dt_type == 'vanilla'
              else ((2 ** maximum_depth - 1) * number_of_variables * 2) + (2 ** maximum_depth)  if function_representation_type == 3 and dt_type == 'vanilla'
              else ((2 ** maximum_depth - 1) * number_of_variables * 2) + (2 ** maximum_depth - 1) + (2 ** maximum_depth) * num_classes if function_representation_type == 3 and dt_type == 'SDT'
              else None
                                                                        )
            #######################################################################################################################################
            ################################################## UPDATE VARIABLES ###################################################################
            #######################################################################################################################################
            globals().update(config['function_family'])
            globals().update(config['data'])
            globals().update(config['lambda_net'])
            globals().update(config['i_net'])
            globals().update(config['evaluation'])
            globals().update(config['computation'])

            #initialize_LambdaNet_config_from_curent_notebook(config)
            #initialize_metrics_config_from_curent_notebook(config)
            #initialize_utility_functions_config_from_curent_notebook(config)
            #initialize_InterpretationNet_config_from_curent_notebook(config)


            #######################################################################################################################################
            ###################################################### PATH + FOLDER CREATION #########################################################
            #######################################################################################################################################
            globals().update(generate_paths(config, path_type='interpretation_net'))

            create_folders_inet(config)

            #######################################################################################################################################
            ############################################################ SLEEP TIMER ##############################################################
            #######################################################################################################################################
            sleep_minutes(sleep_time)


            # In[8]:


            print(path_identifier_interpretation_net)

            print(path_identifier_lambda_net_data)


            # In[9]:


            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            print("Num XLA-GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))


            # ## Load Data and Generate Datasets

            # In[10]:


            def load_lambda_nets(config, no_noise=False, n_jobs=1):

                #def generate_lambda_net()

                #if psutil.virtual_memory().percent > 80:
                    #raise SystemExit("Out of RAM!")

                if no_noise==True:
                    config['noise_injected_level'] = 0
                path_dict = generate_paths(config, path_type='interpretation_net')        

                directory = './data/weights/' + 'weights_' + path_dict['path_identifier_lambda_net_data'] + '/'
                path_network_parameters = directory + 'weights' + '.txt'
                #path_X_data = directory + 'X_test_lambda.txt'
                #path_y_data = directory + 'y_test_lambda.txt'        

                network_parameters = pd.read_csv(path_network_parameters, sep=",", header=None)
                network_parameters = network_parameters.sort_values(by=0)
                if no_noise == False:
                    network_parameters = network_parameters.sample(n=config['i_net']['interpretation_dataset_size'], random_state=config['computation']['RANDOM_SEED'])


                parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky') #loky

                lambda_nets = parallel(delayed(LambdaNet)(network_parameters_row, 
                                                          #X_test_lambda_row, 
                                                          #y_test_lambda_row, 
                                                          config) for network_parameters_row in network_parameters.values)          
                del parallel

                base_model = generate_base_model(config)  

                #def initialize_network_wrapper(config, lambda_net, base_model):
                #    lambda_net.initialize_network(config, base_model)

                #parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='sequential')
                #_ = parallel(delayed(initialize_network_wrapper)(config, lambda_net, base_model) for lambda_net in lambda_nets)   
                #del parallel

                #def initialize_target_function_wrapper(config, lambda_net):
                #    lambda_net.initialize_target_function(config)

                #parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='sequential')
                #_ = parallel(delayed(initialize_target_function_wrapper)(config, lambda_net) for lambda_net in lambda_nets)   
                #del parallel

                lambda_net_dataset = LambdaNetDataset(lambda_nets)

                return lambda_net_dataset



            # In[11]:


            #LOAD DATA
            if different_eval_data:
                config_train = deepcopy(config)
                config_eval = deepcopy(config)

                config_eval['data']['function_generation_type'] = config['evaluation']['eval_data_description']['eval_data_function_generation_type']
                config_eval['data']['lambda_dataset_size'] = config['evaluation']['eval_data_description']['eval_data_lambda_dataset_size']
                config_eval['data']['noise_injected_level'] = config['evaluation']['eval_data_description']['eval_data_noise_injected_level']
                config_eval['data']['noise_injected_type'] = config['evaluation']['eval_data_description']['eval_data_noise_injected_type'] 
                config_eval['lambda_net']['number_of_trained_lambda_nets'] = config['evaluation']['eval_data_description']['eval_data_number_of_trained_lambda_nets']   
                config_eval['i_net']['interpretation_dataset_size'] = config['evaluation']['eval_data_description']['eval_data_interpretation_dataset_size']   

                if False:
                    lambda_net_dataset_train = load_lambda_nets(config_train, n_jobs=n_jobs)
                    lambda_net_dataset_eval = load_lambda_nets(config_eval, n_jobs=n_jobs)

                    lambda_net_dataset_valid, lambda_net_dataset_test = split_LambdaNetDataset(lambda_net_dataset_eval, test_split=test_size)   
                else:
                    lambda_net_dataset_train_with_valid = load_lambda_nets(config_train, n_jobs=n_jobs)
                    lambda_net_dataset_eval = load_lambda_nets(config_eval, n_jobs=n_jobs)

                    _, lambda_net_dataset_test = split_LambdaNetDataset(lambda_net_dataset_eval, test_split=test_size)   
                    lambda_net_dataset_train, lambda_net_dataset_valid = split_LambdaNetDataset(lambda_net_dataset_train_with_valid, test_split=0.1)   


            else:
                lambda_net_dataset = load_lambda_nets(config, n_jobs=n_jobs)

                lambda_net_dataset_train_with_valid, lambda_net_dataset_test = split_LambdaNetDataset(lambda_net_dataset, test_split=test_size)
                lambda_net_dataset_train, lambda_net_dataset_valid = split_LambdaNetDataset(lambda_net_dataset_train_with_valid, test_split=0.1)    


            # ## Data Inspection

            # In[12]:


            lambda_net_dataset_train.shape


            # In[13]:


            lambda_net_dataset_valid.shape


            # In[14]:


            lambda_net_dataset_test.shape


            # In[15]:


            lambda_net_dataset_train.as_pandas(config).head()


            # In[16]:


            lambda_net_dataset_valid.as_pandas(config).head()


            # In[17]:


            lambda_net_dataset_test.as_pandas(config).head()


            # ## Interpretation Network Training

            # In[18]:


            #%load_ext tensorboard
            #%tensorboard --logdir data/logging/ --port=8811


            # In[19]:


            #%load_ext autoreload


            # In[20]:


            #%autoreload 2
            ((X_valid, y_valid), 
             (X_test, y_test),

             history,
             loss_function,
             metrics,

             model,
             encoder_model) = interpretation_net_training(
                                                  lambda_net_dataset_train, 
                                                  lambda_net_dataset_valid, 
                                                  lambda_net_dataset_test,
                                                  config,
                                                  #callback_names=['tensorboard'] #plot_losses
                                                 )


            # In[21]:


            if nas:
                for trial in history: 
                    print(trial.summary())

                writepath_nas = './results_nas.csv'

                if different_eval_data:
                    flat_config = flatten_dict(config_train)
                else:
                    flat_config = flatten_dict(config)    

                if not os.path.exists(writepath_nas):
                    with open(writepath_nas, 'w+') as text_file:       
                        for key in flat_config.keys():
                            text_file.write(key)
                            text_file.write(';')         

                        for hp in history[0].hyperparameters.values.keys():
                            text_file.write(hp + ';')    

                        text_file.write('score')

                        text_file.write('\n')

                with open(writepath_nas, 'a+') as text_file:  
                    for value in flat_config.values():
                        text_file.write(str(value))
                        text_file.write(';')

                    for hp, value in history[0].hyperparameters.values.items():
                        text_file.write(str(value) + ';')        


                    text_file.write(str(history[0].score))

                    text_file.write('\n')            

                    text_file.close()      

            else:
                plt.plot(history['loss'])
                plt.plot(history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'valid'], loc='upper left')    


            # In[22]:


            index = 0
            network_parameters = np.array([lambda_net_dataset_test.network_parameters_array[index]])
            if config['i_net']['data_reshape_version'] == 1 or config['i_net']['data_reshape_version'] == 2:
                network_parameters, network_parameters_flat = restructure_data_cnn_lstm(network_parameters, config, subsequences=None)
            elif config['i_net']['data_reshape_version'] == 3: #autoencoder
                network_parameters, network_parameters_flat, _ = autoencode_data(network_parameters, config, encoder_model)    
            dt_parameters = model.predict(network_parameters)[0]

            if config['function_family']['dt_type'] == 'vanilla':
                image, nodes = anytree_decision_tree_from_parameters(dt_parameters, config=config)
            else:
                tree = generate_random_decision_tree(config)
                tree.initialize_from_parameter_array(dt_parameters, reshape=True, config=config)
                image = tree.plot_tree()
            image


            # In[23]:


            model.summary()


            # In[24]:


            mean_train_parameters = np.round(np.mean(lambda_net_dataset_train.network_parameters_array, axis=0), 5)
            std_train_parameters = np.round(np.std(lambda_net_dataset_train.network_parameters_array, axis=0), 5)

            (inet_evaluation_result_dict_train, 
             inet_evaluation_result_dict_mean_train, 
             dt_distilled_list_train,
             distances_dict) = evaluate_interpretation_net_synthetic_data(lambda_net_dataset_train.network_parameters_array, 
                                                                           lambda_net_dataset_train.X_test_lambda_array,
                                                                           model,
                                                                           config,
                                                                           identifier='train',
                                                                           mean_train_parameters=mean_train_parameters,
                                                                           std_train_parameters=std_train_parameters)


            (inet_evaluation_result_dict_valid, 
             inet_evaluation_result_dict_mean_valid, 
             dt_distilled_list_valid,
             distances_dict) = evaluate_interpretation_net_synthetic_data(lambda_net_dataset_valid.network_parameters_array, 
                                                                           lambda_net_dataset_valid.X_test_lambda_array,
                                                                           model,
                                                                           config,
                                                                           identifier='valid',
                                                                           mean_train_parameters=mean_train_parameters,
                                                                           std_train_parameters=std_train_parameters,
                                                                           distances_dict=distances_dict)

            (inet_evaluation_result_dict_test, 
             inet_evaluation_result_dict_mean_test, 
             dt_distilled_list_test,
             distances_dict) = evaluate_interpretation_net_synthetic_data(lambda_net_dataset_test.network_parameters_array, 
                                                                           lambda_net_dataset_test.X_test_lambda_array,
                                                                           model,
                                                                           config,
                                                                           identifier='test',
                                                                           mean_train_parameters=mean_train_parameters,
                                                                           std_train_parameters=std_train_parameters,
                                                                           distances_dict=distances_dict)

            print_results_synthetic_evaluation(inet_evaluation_result_dict_mean_train, 
                                               inet_evaluation_result_dict_mean_valid, 
                                               inet_evaluation_result_dict_mean_test, 
                                               distances_dict)


            # # REAL DATA EVALUATION

            # In[25]:


            dataset_size_list = [1_000, 10_000, 100_000, 1_000_000, 'TRAIN_DATA']
            dataset_size_list_print = []
            for size in dataset_size_list:
                if type(size) is int:
                    size = size//1000
                    size = str(size) + 'k'
                    dataset_size_list_print.append(size)
                else:
                    dataset_size_list_print.append(size)


            # In[26]:


            #distances_dict = {}
            evaluation_result_dict = {}
            results_dict = {}
            dt_inet_dict = {}
            dt_distilled_list_dict = {}
            data_dict = {}
            normalizer_list_dict = {}

            identifier_list = []


            # ## ADULT DATASET

            # In[27]:


            feature_names = [
                             "Age", #0
                             "Workclass",  #1
                             "fnlwgt",  #2
                             "Education",  #3
                             "Education-Num",  #4
                             "Marital Status", #5
                             "Occupation",  #6
                             "Relationship",  #7
                             "Race",  #8
                             "Sex",  #9
                             "Capital Gain",  #10
                             "Capital Loss", #11
                             "Hours per week",  #12
                             "Country", #13
                             "capital_gain" #14
                            ] 

            adult_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=feature_names, index_col=False)


            #adult_data['Workclass'][adult_data['Workclass'] != ' Private'] = 'Other'
            #adult_data['Race'][adult_data['Race'] != ' White'] = 'Other'

            #adult_data.head()

            features_select = [
                             "Sex",  #9 
                             "Race",  #8
                             "Workclass",  #1
                             "Age", #0
                             "fnlwgt",  #2
                             "Education",  #3
                             "Education-Num",  #4
                             "Marital Status", #5
                             "Occupation",  #6
                             "Relationship",  #7
                             "Capital Gain",  #10
                             "Capital Loss", #11
                             "Hours per week",  #12
                             #"Country", #13 
                             "capital_gain"
                              ]

            adult_data = adult_data[features_select]

            nominal_features_adult = ['Race', 'Workclass', 'Education', "Marital Status", "Occupation", "Relationship"]
            ordinal_features_adult = ['Sex']

            X_data_adult = adult_data.drop(['capital_gain'], axis = 1)

            #y_data_adult = pd.Series(OrdinalEncoder().fit_transform(adult_data['capital_gain'].values.reshape(-1, 1)).flatten(), name='capital_gain')
            y_data_adult = ((adult_data['capital_gain'] != ' <=50K') * 1)


            # In[28]:


            config_train_network_adult = deepcopy(config)
            #config_train_network_adult['lambda_net']['batch_lambda'] = 32
            #config_train_network_adult['lambda_net']['learning_rate_lambda'] = 0.0003
            #config_train_network_adult['lambda_net']['dropout_lambda'] = 0.25
            #config_train_network_adult['lambda_net']['epochs_lambda'] = 5


            # In[29]:


            identifier = 'Adult'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_adult, 
                                                                            y_data_adult, 
                                                                            nominal_features = nominal_features_adult, 
                                                                            ordinal_features = ordinal_features_adult,
                                                                            config = config,
                                                                            config_train_network = config_train_network_adult)

            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict['Adult'], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # ## Titanic Dataset

            # In[30]:


            titanic_data = pd.read_csv("./real_world_datasets/Titanic/train.csv")

            titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace = True)
            titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace = True)

            titanic_data['Embarked'].fillna('S', inplace = True)

            features_select = [
                                #'Cabin', 
                                #'Ticket', 
                                #'Name', 
                                #'PassengerId'    
                                'Sex',    
                                'Embarked',
                                'Pclass',
                                'Age',
                                'SibSp',    
                                'Parch',
                                'Fare',    
                                'Survived',    
                              ]

            titanic_data = titanic_data[features_select]

            nominal_features_titanic = ['Embarked']#[1, 2, 7]
            ordinal_features_titanic = ['Sex']

            X_data_titanic = titanic_data.drop(['Survived'], axis = 1)
            y_data_titanic = titanic_data['Survived']


            #     survival	Survival	0 = No, 1 = Yes
            #     pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
            #     sex	Sex	
            #     Age	Age in years	
            #     sibsp	# of siblings / spouses aboard the Titanic	
            #     parch	# of parents / children aboard the Titanic	
            #     ticket	Ticket number	
            #     fare	Passenger fare	
            #     cabin	Cabin number	
            #     embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

            # In[31]:


            identifier = 'Titanic'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_titanic, 
                                                                            y_data_titanic, 
                                                                            nominal_features = nominal_features_titanic, 
                                                                            ordinal_features = ordinal_features_titanic,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # ## Absenteeism at Work Dataset

            # In[32]:


            absenteeism_data = pd.read_csv('real_world_datasets/Absenteeism/absenteeism.csv', delimiter=';')

            features_select = [
                                       'Disciplinary failure', #CATEGORICAL
                                       'Social drinker', #CATEGORICAL
                                       'Social smoker', #CATEGORICAL
                                       'Transportation expense', 
                                       'Distance from Residence to Work',
                                       'Service time', 
                                       'Age', 
                                       'Work load Average/day ', 
                                       'Hit target',
                                       'Education', 
                                       'Son', 
                                       'Pet', 
                                       'Weight', 
                                       'Height', 
                                       'Body mass index', 
                                       'Absenteeism time in hours'
                                    ]

            absenteeism_data = absenteeism_data[features_select]

            nominal_features_absenteeism = []
            ordinal_features_absenteeism = []

            X_data_absenteeism = absenteeism_data.drop(['Absenteeism time in hours'], axis = 1)
            y_data_absenteeism = ((absenteeism_data['Absenteeism time in hours'] > 4) * 1) #absenteeism_data['Absenteeism time in hours']


            #     3. Month of absence
            #     4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
            #     5. Seasons (summer (1), autumn (2), winter (3), spring (4))
            #     6. Transportation expense
            #     7. Distance from Residence to Work (kilometers)
            #     8. Service time
            #     9. Age
            #     10. Work load Average/day
            #     11. Hit target
            #     12. Disciplinary failure (yes=1; no=0)
            #     13. Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
            #     14. Son (number of children)
            #     15. Social drinker (yes=1; no=0)
            #     16. Social smoker (yes=1; no=0)
            #     17. Pet (number of pet)
            #     18. Weight
            #     19. Height
            #     20. Body mass index
            #     21. Absenteeism time in hours (target)

            # In[33]:


            identifier = 'Absenteeism'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_absenteeism, 
                                                                            y_data_absenteeism, 
                                                                            nominal_features = nominal_features_absenteeism, 
                                                                            ordinal_features = ordinal_features_absenteeism,
                                                                            config = config,
                                                                            config_train_network = None)

            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Loan Dataset

            # In[34]:


            loan_data = pd.read_csv('real_world_datasets/Loan/loan-train.csv', delimiter=',')

            loan_data['Gender'].fillna(loan_data['Gender'].mode()[0], inplace=True)
            loan_data['Dependents'].fillna(loan_data['Dependents'].mode()[0], inplace=True)
            loan_data['Married'].fillna(loan_data['Married'].mode()[0], inplace=True)
            loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0], inplace=True)
            loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean(), inplace=True)
            loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mean(), inplace=True)
            loan_data['Credit_History'].fillna(loan_data['Credit_History'].mean(), inplace=True)

            features_select = [
                                #'Loan_ID', 
                                'Gender', #
                                'Married', 
                                'Dependents', 
                                'Education',
                                'Self_Employed', 
                                'ApplicantIncome', 
                                'CoapplicantIncome', 
                                'LoanAmount',
                                'Loan_Amount_Term', 
                                'Credit_History', 
                                'Property_Area', 
                                'Loan_Status'
                                ]

            loan_data = loan_data[features_select]

            nominal_features_loan = [
                                    'Dependents',
                                    'Education',
                                    'Property_Area',    
                                    ]
            ordinal_features_loan = [
                                'Gender', 
                                'Married', 
                                'Self_Employed',
                               ]

            X_data_loan = loan_data.drop(['Loan_Status'], axis = 1)
            y_data_loan = ((loan_data['Loan_Status'] == 'Y') * 1) #absenteeism_data['Absenteeism time in hours']


            # In[35]:


            identifier = 'Loan House'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_loan, 
                                                                            y_data_loan, 
                                                                            nominal_features = nominal_features_loan, 
                                                                            ordinal_features = ordinal_features_loan,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Loan Credit

            # In[36]:


            loan_credit_data = pd.read_csv('real_world_datasets/Credit Loan/train_split.csv', delimiter=',')

            loan_credit_data['emp_title'].fillna(loan_credit_data['emp_title'].mode()[0], inplace=True)
            loan_credit_data['emp_length'].fillna(loan_credit_data['emp_length'].mode()[0], inplace=True)
            #loan_credit_data['desc'].fillna(loan_credit_data['desc'].mode()[0], inplace=True)
            loan_credit_data['title'].fillna(loan_credit_data['title'].mode()[0], inplace=True)
            #loan_credit_data['mths_since_last_delinq'].fillna(loan_credit_data['mths_since_last_delinq'].mode()[0], inplace=True)
            #loan_credit_data['mths_since_last_record'].fillna(loan_credit_data['mths_since_last_record'].mode()[0], inplace=True)
            loan_credit_data['revol_util'].fillna(loan_credit_data['revol_util'].mode()[0], inplace=True)
            loan_credit_data['collections_12_mths_ex_med'].fillna(loan_credit_data['collections_12_mths_ex_med'].mode()[0], inplace=True)
            #loan_credit_data['mths_since_last_major_derog'].fillna(loan_credit_data['mths_since_last_major_derog'].mode()[0], inplace=True)
            #loan_credit_data['verification_status_joint'].fillna(loan_credit_data['verification_status_joint'].mode()[0], inplace=True)
            loan_credit_data['tot_coll_amt'].fillna(loan_credit_data['tot_coll_amt'].mode()[0], inplace=True)
            loan_credit_data['tot_cur_bal'].fillna(loan_credit_data['tot_cur_bal'].mode()[0], inplace=True)
            loan_credit_data['total_rev_hi_lim'].fillna(loan_credit_data['total_rev_hi_lim'].mode()[0], inplace=True)


            ##remove too many null
            #'mths_since_last_delinq','mths_since_last_record', 'mths_since_last_major_derog','pymnt_plan','desc', 'verification_status_joint'


            features_select = [
                                #'member_id', 
                                'loan_amnt', 
                                'funded_amnt', 
                                'funded_amnt_inv', 
                                'term',
                                #'batch_enrolled',
                                'int_rate', 
                                'grade', 
                                #'sub_grade', 
                                #'emp_title',
                                'emp_length',
                                'home_ownership', 
                                'annual_inc', 
                                'verification_status',
                                #'pymnt_plan', 
                                #'desc', 
                                'purpose', 
                                'title', 
                                #'zip_code', 
                                #'addr_state',
                                'dti', 
                                'delinq_2yrs', 
                                'inq_last_6mths', 
                                #'mths_since_last_delinq',
                                #'mths_since_last_record',
                                'open_acc', 
                                'pub_rec', 
                                'revol_bal',
                                'revol_util', 
                                'total_acc', 
                                'initial_list_status', 
                                'total_rec_int',
                                'total_rec_late_fee', 
                                'recoveries', 
                                'collection_recovery_fee',
                                'collections_12_mths_ex_med', 
                                #'mths_since_last_major_derog',
                                'application_type', 
                                #'verification_status_joint', 
                                'last_week_pay',
                                'acc_now_delinq', 
                                'tot_coll_amt', 
                                'tot_cur_bal', 
                                'total_rev_hi_lim',
                                'loan_status'
                                ]

            loan_credit_data = loan_credit_data[features_select]

            nominal_features_loan_credit = [

                                    ]
            ordinal_features_loan_credit = [
                                #'member_id', 
                                'loan_amnt', 
                                'funded_amnt', 
                                'funded_amnt_inv', 
                                'term',
                                #'batch_enrolled',
                                'int_rate', 
                                'grade', 
                                #'sub_grade', 
                                #'emp_title',
                                'emp_length',
                                'home_ownership', 
                                'annual_inc', 
                                'verification_status',
                                #'pymnt_plan', 
                                #'desc', 
                                'purpose', 
                                'title', 
                                #'zip_code', 
                                #'addr_state',
                                'dti', 
                                'delinq_2yrs', 
                                'inq_last_6mths', 
                                #'mths_since_last_delinq',
                                #'mths_since_last_record',
                                'open_acc', 
                                'pub_rec', 
                                'revol_bal',
                                'revol_util', 
                                'total_acc', 
                                'initial_list_status', 
                                'total_rec_int',
                                'total_rec_late_fee', 
                                'recoveries', 
                                'collection_recovery_fee',
                                'collections_12_mths_ex_med', 
                                #'mths_since_last_major_derog',
                                'application_type', 
                                #'verification_status_joint', 
                                'last_week_pay',
                                'acc_now_delinq', 
                                'tot_coll_amt', 
                                'tot_cur_bal', 
                                'total_rev_hi_lim',
                               ]

            X_data_loan_credit = loan_credit_data.drop(['loan_status'], axis = 1)
            y_data_loan_credit = pd.Series(OrdinalEncoder().fit_transform(loan_credit_data['loan_status'].values.reshape(-1, 1)).flatten(), name='loan_status')


            # In[37]:


            identifier = 'Loan Credit'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_loan_credit, 
                                                                            y_data_loan_credit, 
                                                                            nominal_features = nominal_features_loan_credit, 
                                                                            ordinal_features = ordinal_features_loan_credit,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Medical Insurance

            # In[38]:


            medical_insurance_data = pd.read_csv('real_world_datasets/Medical Insurance/insurance.csv', delimiter=',')

            features_select = [
                                'age', 
                                'sex', 
                                'bmi', 
                                'children', 
                                'smoker',
                                'region',
                                'charges'
                                ]

            medical_insurance_data = medical_insurance_data[features_select]

            nominal_features_medical_insurance = [

                                    ]
            ordinal_features_medical_insurance = [
                                'sex',
                                'region',
                                'smoker'
                               ]


            X_data_medical_insurance = medical_insurance_data.drop(['charges'], axis = 1)
            y_data_medical_insurance = ((medical_insurance_data['charges'] > 10_000) * 1)


            print(X_data_medical_insurance.shape)
            X_data_medical_insurance.head()


            # In[39]:


            identifier = 'Medical Insurance'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_medical_insurance, 
                                                                            y_data_medical_insurance, 
                                                                            nominal_features = nominal_features_medical_insurance, 
                                                                            ordinal_features = ordinal_features_medical_insurance,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Bank Marketing

            # In[40]:


            bank_data = pd.read_csv('real_world_datasets/Bank Marketing/bank-full.csv', delimiter=';') #bank

            features_select = [
                                'age',
                                #'job', 
                                'marital', 
                                'education', 
                                'default',
                                'housing',
                                'loan',
                                #'contact',
                                #'day',
                                #'month',
                                'duration',
                                'campaign',
                                'pdays',
                                'previous',
                                'poutcome',
                                'y',
                                ]

            bank_data = bank_data[features_select]

            nominal_features_bank = [
                                    #'job',
                                    'education',
                                    #'contact',
                                    #'day',
                                    #'month',
                                    'poutcome',
                                    ]
            ordinal_features_bank = [
                                'marital',
                                'default',
                                'housing',
                                'loan',
                               ]


            X_data_bank = bank_data.drop(['y'], axis = 1)
            y_data_bank = pd.Series(OrdinalEncoder().fit_transform(bank_data['y'].values.reshape(-1, 1)).flatten(), name='y')


            # In[41]:


            identifier = 'Bank Marketing'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_bank, 
                                                                            y_data_bank, 
                                                                            nominal_features = nominal_features_bank, 
                                                                            ordinal_features = ordinal_features_bank,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Brest Cancer Wisconsin

            # In[42]:


            feature_names = [
                            'Sample code number',
                            'Clump Thickness',
                            'Uniformity of Cell Size',
                            'Uniformity of Cell Shape',
                            'Marginal Adhesion',
                            'Single Epithelial Cell Size',
                            'Bare Nuclei',
                            'Bland Chromatin',
                            'Normal Nucleoli',
                            'Mitoses',
                            'Class',
                            ]

            bcw_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=feature_names, index_col=False)

            bcw_data['Clump Thickness'][bcw_data['Clump Thickness'] == '?'] = bcw_data['Clump Thickness'].mode()[0]
            bcw_data['Uniformity of Cell Size'][bcw_data['Uniformity of Cell Size'] == '?'] = bcw_data['Uniformity of Cell Size'].mode()[0]
            bcw_data['Uniformity of Cell Shape'][bcw_data['Uniformity of Cell Shape'] == '?'] = bcw_data['Uniformity of Cell Shape'].mode()[0]
            bcw_data['Marginal Adhesion'][bcw_data['Marginal Adhesion'] == '?'] = bcw_data['Marginal Adhesion'].mode()[0]
            bcw_data['Single Epithelial Cell Size'][bcw_data['Single Epithelial Cell Size'] == '?'] = bcw_data['Single Epithelial Cell Size'].mode()[0]
            bcw_data['Bare Nuclei'][bcw_data['Bare Nuclei'] == '?'] = bcw_data['Bare Nuclei'].mode()[0]
            bcw_data['Bland Chromatin'][bcw_data['Bland Chromatin'] == '?'] = bcw_data['Bland Chromatin'].mode()[0]
            bcw_data['Normal Nucleoli'][bcw_data['Normal Nucleoli'] == '?'] = bcw_data['Normal Nucleoli'].mode()[0]
            bcw_data['Mitoses'][bcw_data['Mitoses'] == '?'] = bcw_data['Mitoses'].mode()[0]

            features_select = [
                            #'Sample code number',
                            'Clump Thickness',
                            'Uniformity of Cell Size',
                            'Uniformity of Cell Shape',
                            'Marginal Adhesion',
                            'Single Epithelial Cell Size',
                            'Bare Nuclei',
                            'Bland Chromatin',
                            'Normal Nucleoli',
                            'Mitoses',
                            'Class',
                                ]

            bcw_data = bcw_data[features_select]

            nominal_features_bcw = [
                                    ]
            ordinal_features_bcw = [
                               ]


            X_data_bcw = bcw_data.drop(['Class'], axis = 1)
            y_data_bcw = pd.Series(OrdinalEncoder().fit_transform(bcw_data['Class'].values.reshape(-1, 1)).flatten(), name='Class')


            # In[43]:


            identifier = 'Brest Cancer Wisconsin'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_bcw, 
                                                                            y_data_bcw, 
                                                                            nominal_features = nominal_features_bcw, 
                                                                            ordinal_features = ordinal_features_bcw,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Wisconsin Diagnostic Breast Cancer

            # In[44]:


            feature_names = [
                            'ID number',
                            'Diagnosis',
                            'radius',# (mean of distances from center to points on the perimeter)
                            'texture',# (standard deviation of gray-scale values)
                            'perimeter',
                            'area',
                            'smoothness',# (local variation in radius lengths)
                            'compactness',# (perimeter^2 / area - 1.0)
                            'concavity',# (severity of concave portions of the contour)
                            'concave points',# (number of concave portions of the contour)
                            'symmetry',
                            'fractal dimension',# ("coastline approximation" - 1)
                            ]
            #Wisconsin Diagnostic Breast Cancer
            wdbc_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', names=feature_names, index_col=False)

            features_select = [
                                #'ID number',
                                'Diagnosis',
                                'radius',# (mean of distances from center to points on the perimeter)
                                'texture',# (standard deviation of gray-scale values)
                                'perimeter',
                                'area',
                                'smoothness',# (local variation in radius lengths)
                                'compactness',# (perimeter^2 / area - 1.0)
                                'concavity',# (severity of concave portions of the contour)
                                'concave points',# (number of concave portions of the contour)
                                'symmetry',
                                'fractal dimension',# ("coastline approximation" - 1)
                                ]

            wdbc_data = wdbc_data[features_select]

            nominal_features_wdbc = [
                                    ]
            ordinal_features_wdbc = [
                               ]


            X_data_wdbc = wdbc_data.drop(['Diagnosis'], axis = 1)
            y_data_wdbc= pd.Series(OrdinalEncoder().fit_transform(wdbc_data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')


            # In[45]:


            identifier = 'Wisconsin Diagnostic Breast Cancer'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_wdbc, 
                                                                            y_data_wdbc, 
                                                                            nominal_features = nominal_features_wdbc, 
                                                                            ordinal_features = ordinal_features_wdbc,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Wisconsin Prognostic Breast Cancer

            # In[46]:


            feature_names = [
                            'ID number',
                            'Diagnosis',
                            'radius',# (mean of distances from center to points on the perimeter)
                            'texture',# (standard deviation of gray-scale values)
                            'perimeter',
                            'area',
                            'smoothness',# (local variation in radius lengths)
                            'compactness',# (perimeter^2 / area - 1.0)
                            'concavity',# (severity of concave portions of the contour)
                            'concave points',# (number of concave portions of the contour)
                            'symmetry',
                            'fractal dimension',# ("coastline approximation" - 1)
                            ]
            #Wisconsin Prognostic Breast Cancer
            wpbc_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data', names=feature_names, index_col=False)

            features_select = [
                                #'ID number',
                                'Diagnosis',
                                'radius',# (mean of distances from center to points on the perimeter)
                                'texture',# (standard deviation of gray-scale values)
                                'perimeter',
                                'area',
                                'smoothness',# (local variation in radius lengths)
                                'compactness',# (perimeter^2 / area - 1.0)
                                'concavity',# (severity of concave portions of the contour)
                                'concave points',# (number of concave portions of the contour)
                                'symmetry',
                                'fractal dimension',# ("coastline approximation" - 1)
                                ]

            wpbc_data = wpbc_data[features_select]

            nominal_features_wpbc = [
                                    ]
            ordinal_features_wpbc = [
                               ]

            X_data_wpbc = wpbc_data.drop(['Diagnosis'], axis = 1)
            y_data_wpbc= pd.Series(OrdinalEncoder().fit_transform(wpbc_data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')


            # In[47]:


            identifier = 'Wisconsin Prognostic Breast Cancer'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_wpbc, 
                                                                            y_data_wpbc, 
                                                                            nominal_features = nominal_features_wpbc, 
                                                                            ordinal_features = ordinal_features_wpbc,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Abalone

            # In[48]:


            feature_names = [
                            'Sex',#		nominal			M, F, and I (infant)
                            'Length',#	continuous	mm	Longest shell measurement
                            'Diameter',#	continuous	mm	perpendicular to length
                            'Height',#		continuous	mm	with meat in shell
                            'Whole weight',#	continuous	grams	whole abalone
                            'Shucked weight',#	continuous	grams	weight of meat
                            'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                            'Shell weight',#	continuous	grams	after being dried
                            'Rings',#		integer			+1.5 gives the age in years
                            ]

            abalone_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', names=feature_names, index_col=False)


            features_select = [
                            'Sex',#		nominal			M, F, and I (infant)
                            'Length',#	continuous	mm	Longest shell measurement
                            'Diameter',#	continuous	mm	perpendicular to length
                            'Height',#		continuous	mm	with meat in shell
                            'Whole weight',#	continuous	grams	whole abalone
                            'Shucked weight',#	continuous	grams	weight of meat
                            'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                            'Shell weight',#	continuous	grams	after being dried
                            'Rings',#		integer			+1.5 gives the age in years
                                ]

            abalone_data = abalone_data[features_select]

            nominal_features_abalone = [
                                    'Sex',
                                    ]
            ordinal_features_abalone = [
                               ]

            X_data_abalone = abalone_data.drop(['Rings'], axis = 1)
            y_data_abalone = ((abalone_data['Rings'] > 10) * 1)



            # In[49]:


            identifier = 'Abalone'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_abalone, 
                                                                            y_data_abalone, 
                                                                            nominal_features = nominal_features_abalone, 
                                                                            ordinal_features = ordinal_features_abalone,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Car

            # In[50]:


            feature_names = [
               'buying',#       v-high, high, med, low
               'maint',#        v-high, high, med, low
               'doors',#        2, 3, 4, 5-more
               'persons',#      2, 4, more
               'lug_boot',#     small, med, big
               'safety',#       low, med, high
               'class',#        unacc, acc, good, v-good
                            ]

            car_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=feature_names, index_col=False)

            features_select = [
                               'buying',#       v-high, high, med, low
                               'maint',#        v-high, high, med, low
                               'doors',#        2, 3, 4, 5-more
                               'persons',#      2, 4, more
                               'lug_boot',#     small, med, big
                               'safety',#       low, med, high
                               'class',#        unacc, acc, good, v-good
                                ]

            car_data = car_data[features_select]

            nominal_features_car = [
                                   'buying',#       v-high, high, med, low
                                   'maint',#        v-high, high, med, low
                                   'doors',#        2, 3, 4, 5-more
                                   'persons',#      2, 4, more
                                   'lug_boot',#     small, med, big
                                   'safety',#       low, med, high
                                    ]

            ordinal_features_car = [
                               ]



            X_data_car = car_data.drop(['class'], axis = 1)
            y_data_car = ((car_data['class'] != 'unacc') * 1)


            # In[51]:


            identifier = 'Car'
            identifier_list.append(identifier)

            (distances_dict[identifier], 
             evaluation_result_dict[identifier], 
             results_dict[identifier], 
             dt_inet_dict[identifier], 
             dt_distilled_list_dict[identifier], 
             data_dict[identifier],
             normalizer_list_dict[identifier]) = evaluate_real_world_dataset(model,
                                                                            dataset_size_list,
                                                                            mean_train_parameters,
                                                                            std_train_parameters,
                                                                            lambda_net_dataset_train.network_parameters_array,
                                                                            X_data_car, 
                                                                            y_data_car, 
                                                                            nominal_features = nominal_features_car, 
                                                                            ordinal_features = ordinal_features_car,
                                                                            config = config,
                                                                            config_train_network = None)
            print_head = None
            if verbosity > 0:
                print_results_different_data_sizes(results_dict[identifier], dataset_size_list_print)
                print_network_distances(distances_dict)

                dt_inet_plot = plot_decision_tree_from_parameters(dt_inet_dict[identifier], normalizer_list_dict[identifier], config)
                dt_distilled_plot = plot_decision_tree_from_model(dt_distilled_list_dict[identifier][-2], config)

                display(dt_inet_plot, dt_distilled_plot)

                print_head = data_dict[identifier]['X_train'].head()
            print_head


            # # Plot and Save Results

            # In[66]:


            #print_complete_performance_evaluation_results(results_dict, identifier_list, dataset_size_list, dataset_size=10000)
            complete_performance_evaluation_results = get_complete_performance_evaluation_results_dataframe(results_dict, identifier_list, dataset_size_list, dataset_size=10000)
            complete_performance_evaluation_results.head(20)


            # In[69]:


            #print_network_distances(distances_dict)
            network_distances = get_print_network_distances_dataframe(distances_dict)
            network_distances.head(20)


            # In[57]:


            writepath_complete = './results_complete.csv'
            writepath_summary = './results_summary.csv'

            #TODO: ADD COMPLEXITY FOR DTS

            if different_eval_data:
                flat_config = flatten_dict(config_train)
            else:
                flat_config = flatten_dict(config)    

            flat_dict_train = flatten_dict(inet_evaluation_result_dict_train)
            flat_dict_valid = flatten_dict(inet_evaluation_result_dict_valid)
            flat_dict_test = flatten_dict(inet_evaluation_result_dict_test)

            if not os.path.exists(writepath_complete):
                with open(writepath_complete, 'w+') as text_file:       
                    for key in flat_config.keys():
                        text_file.write(key)
                        text_file.write(';')      

                    number_of_evaluated_networks = np.array(flat_dict_train['inet_scores_binary_crossentropy']).shape[0]
                    for key in flat_dict_train.keys():
                        if 'function_values' not in key:
                            for i in range(number_of_evaluated_networks):
                                text_file.write(key + '_train_' + str(i) + ';')    

                    number_of_evaluated_networks = np.array(flat_dict_valid['inet_scores_binary_crossentropy']).shape[0]
                    for key in flat_dict_valid.keys():
                        if 'function_values' not in key:
                            for i in range(number_of_evaluated_networks):
                                text_file.write(key + '_valid_' + str(i) + ';')       

                    number_of_evaluated_networks = np.array(flat_dict_test['inet_scores_binary_crossentropy']).shape[0]
                    for key in flat_dict_test.keys():
                        if 'function_values' not in key:
                            for i in range(number_of_evaluated_networks):
                                text_file.write(key + '_test_' + str(i) + ';')        

                    text_file.write('\n')

            with open(writepath_complete, 'a+') as text_file:  
                for value in flat_config.values():
                    text_file.write(str(value))
                    text_file.write(';')


                number_of_evaluated_networks = np.array(flat_dict_train['inet_scores_binary_crossentropy']).shape[0]
                for key, values in flat_dict_train.items():
                    if 'function_values' not in key:
                        for score in values:
                            text_file.write(str(score) + ';')   

                number_of_evaluated_networks = np.array(flat_dict_valid['inet_scores_binary_crossentropy']).shape[0]
                for key, values in flat_dict_valid.items():
                    if 'function_values' not in key:
                        for score in values:
                            text_file.write(str(score) + ';')   

                number_of_evaluated_networks = np.array(flat_dict_test['inet_scores_binary_crossentropy']).shape[0]
                for key, values in flat_dict_test.items():
                    if 'function_values' not in key:
                        for score in values:
                            text_file.write(str(score) + ';')   

                text_file.write('\n')            

                text_file.close()  



            # In[58]:


            inet_evaluation_result_dict_mean_train_flat = flatten_dict(inet_evaluation_result_dict_mean_train)
            inet_evaluation_result_dict_mean_valid_flat = flatten_dict(inet_evaluation_result_dict_mean_valid)
            inet_evaluation_result_dict_mean_test_flat = flatten_dict(inet_evaluation_result_dict_mean_test)

            identifier_list_synthetic = ['train', 'valid', 'test']
            identifier_list_combined = list(flatten_list([identifier_list_synthetic, identifier_list]))

            if not os.path.exists(writepath_summary):
                with open(writepath_summary, 'w+') as text_file: 

                    for key in flat_config.keys():
                        text_file.write(key + ';')

                    for identifier in identifier_list_synthetic:
                        for key in inet_evaluation_result_dict_mean_train_flat.keys():
                            text_file.write(identifier + '_' + key + ';')


                    for dataset_size in dataset_size_list:
                        for identifier in identifier_list:
                            results_dict_flat = flatten_dict(results_dict[identifier][-2])
                            del results_dict_flat['function_values_y_test_inet_dt']
                            del results_dict_flat['function_values_y_test_distilled_dt']

                            for key in results_dict_flat.keys():
                                text_file.write(key + '_' + identifier + '_' + str(dataset_size) + ';')                                   


                    for key in distances_dict['train'].keys():
                        for identifier in identifier_list_combined:
                            text_file.write(key + '_' + identifier + ';') 

                    text_file.write('\n')

            with open(writepath_summary, 'a+') as text_file: 

                for value in flat_config.values():
                    text_file.write(str(value) + ';')

                for value in inet_evaluation_result_dict_mean_train_flat.values():
                    text_file.write(str(value) + ';')
                for value in inet_evaluation_result_dict_mean_valid_flat.values():
                    text_file.write(str(value) + ';')            
                for value in inet_evaluation_result_dict_mean_test_flat.values():
                    text_file.write(str(value) + ';')

                for i in range(len(dataset_size_list)):
                    for identifier in identifier_list:
                        evaluation_result_dict_flat = flatten_dict(evaluation_result_dict[identifier])
                        del evaluation_result_dict_flat['function_values_y_test_inet_dt']
                        del evaluation_result_dict_flat['function_values_y_test_distilled_dt']

                        for values in evaluation_result_dict_flat.values():
                            text_file.write(str(values[i]) + ';')            

                for key in distances_dict['train'].keys():
                    for identifier in identifier_list_combined:
                        text_file.write(str(distances_dict[identifier][key]) + ';')      

                text_file.write('\n')

                text_file.close()      


            # In[59]:


            if use_gpu:
                from numba import cuda 
                device = cuda.get_current_device()
                device.reset()



