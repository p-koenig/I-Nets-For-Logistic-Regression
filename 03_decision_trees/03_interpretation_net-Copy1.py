#!/usr/bin/env python
# coding: utf-8

# # Inerpretation-Net Training

# ## Specitication of Experiment Settings

# In[1]:


#######################################################################################################################################
###################################################### CONFIG FILE ####################################################################
#######################################################################################################################################
sleep_time = 0 #minutes



config = {
    'function_family': {
        'maximum_depth': 4,
        'beta': 1,
        'decision_sparsity': 1,
        'fully_grown': True,                      
    },
    'data': {
        'number_of_variables': 5, 
        'num_classes': 2,
        
        'function_generation_type': 'random_decision_tree', # 'make_classification' 'random_decision_tree'
        'objective': 'classification', # 'regression'
        
        'x_max': 1,
        'x_min': 0,
        'x_distrib': 'uniform', #'normal', 'uniform',       
                
        'lambda_dataset_size': 1000, #number of samples per function
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
        'lambda_network_layers': [64],
        'optimizer_lambda': 'adam',
        'loss_lambda': 'binary_crossentropy', #categorical_crossentropy
        
        'number_of_lambda_weights': None,
        
        'number_initializations_lambda': 1, 
        
        'number_of_trained_lambda_nets': 10000,
    },     
    
    'i_net': {
        'dense_layers': [1056, 512],
        'convolution_layers': None,
        'lstm_layers': None,
        'dropout': [0.2, 0.1],
        
        'optimizer': 'adam', #adam
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['binary_accuracy'],
        
        'epochs': 10, 
        'early_stopping': True,
        'batch_size': 256,

        'interpretation_dataset_size': 500,
                
        'test_size': 50, #Float for fraction, Int for number 0
        
        'function_representation_type': 3, # 1=standard representation; 2=sparse representation, 3=vanilla_dt

        'optimize_decision_function': True, #False
        'function_value_loss': True, #False
                      
        'data_reshape_version': None, #default to 2 options:(None, 0,1 2)
        
        'nas': False,
        'nas_type': 'SEQUENTIAL', #options:(None, 'SEQUENTIAL', 'CNN', 'LSTM', 'CNN-LSTM', 'CNN-LSTM-parallel')      
        'nas_trials': 100,
    },    
    
    'evaluation': {   
        #'inet_holdout_seed_evaluation': False,
            
        'random_evaluation_dataset_size': 50, 
        'per_network_optimization_dataset_size': 5000,

        'sklearn_dt_benchmark': False,
        'sdt_benchmark': False,
        
    },    
    
    'computation':{
        'load_model': False,
        
        'n_jobs': -3,
        'use_gpu': False,
        'gpu_numbers': '0',
        'RANDOM_SEED': 42,   
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


# In[ ]:


#######################################################################################################################################
##################################################### IMPORT LIBRARIES ################################################################
#######################################################################################################################################
from itertools import product       
from tqdm import tqdm_notebook as tqdm
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
import os
import shutil

import logging

#from prettytable import PrettyTable
#import colored
import math

import time
from datetime import datetime
from collections.abc import Iterable


from joblib import Parallel, delayed

from scipy.integrate import quad

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold, ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, r2_score

#from similaritymeasures import frechet_dist, area_between_two_curves, dtw
import tensorflow as tf
#import tensorflow_addons as tfa
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


import tensorflow.keras.backend as K
from livelossplot import PlotLossesKerasTF
#from keras_tqdm import TQDMNotebookCallback

from matplotlib import pyplot as plt
import seaborn as sns


import random 


import warnings

from IPython.display import Image
from IPython.display import display, Math, Latex, clear_output


# In[ ]:


tf.__version__


# In[ ]:


#######################################################################################################################################
########################################### IMPORT GLOBAL VARIABLES FROM CONFIG #######################################################
#######################################################################################################################################
globals().update(config['function_family'])
globals().update(config['data'])
globals().update(config['lambda_net'])
globals().update(config['evaluation'])
globals().update(config['computation'])


# In[ ]:


#######################################################################################################################################
################################################### VARIABLE ADJUSTMENTS ##############################################################
#######################################################################################################################################

config['i_net']['data_reshape_version'] = 2 if data_reshape_version == None and (convolution_layers != None or lstm_layers != None or (nas and nas_type != 'SEQUENTIAL')) else data_reshape_version

#######################################################################################################################################
###################################################### SET VARIABLES + DESIGN #########################################################
#######################################################################################################################################

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_numbers if use_gpu else ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['XLA_FLAGS'] =  '--xla_gpu_cuda_data_dir=/usr/lib/cuda-10.1'

logging.getLogger('tensorflow').disabled = True

sns.set_style("darkgrid")
#np.set_printoptions(suppress=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if int(tf.__version__[0]) >= 2:
    tf.random.set_seed(RANDOM_SEED)
else:
    tf.set_random_seed(RANDOM_SEED)
    
    
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 200)
np.set_printoptions(threshold=200)

warnings.filterwarnings('ignore')


# In[ ]:


from utilities.InterpretationNet import *
from utilities.LambdaNet import *
from utilities.metrics import *
from utilities.utility_functions import *
from utilities.DecisionTree_BASIC import *

#######################################################################################################################################
####################################################### CONFIG ADJUSTMENTS ############################################################
#######################################################################################################################################

config['lambda_net']['number_of_lambda_weights'] = get_number_of_lambda_net_parameters(lambda_network_layers, number_of_variables, num_classes)
config['function_family']['basic_function_representation_length'] = (2 ** maximum_depth - 1) * number_of_variables + (2 ** maximum_depth - 1) + (2 ** maximum_depth) * num_classes
config['function_family']['function_representation_length'] = ( (2 ** maximum_depth - 1) * number_of_variables + (2 ** maximum_depth - 1) + (2 ** maximum_depth) * num_classes  if function_representation_type == 1 
                                                              else (2 ** maximum_depth - 1) * decision_sparsity + (2 ** maximum_depth - 1) + ((2 ** maximum_depth - 1)  * decision_sparsity * number_of_variables) + (2 ** maximum_depth) * num_classes if function_representation_type == 2
                                                              else (2 ** maximum_depth - 1) * decision_sparsity + ((2 ** maximum_depth - 1)  * decision_sparsity * number_of_variables) + (2 ** maximum_depth) * num_classes)

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


# In[ ]:


print(path_identifier_interpretation_net)

print(path_identifier_lambda_net_data)


# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num XLA-GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))


# ## Load Data and Generate Datasets

# In[ ]:


def load_lambda_nets(config, no_noise=False, n_jobs=1):
    
    #def generate_lambda_net()
    
    if psutil.virtual_memory().percent > 80:
        raise SystemExit("Out of RAM!")
    
    if no_noise==True:
        config['noise_injected_level'] = 0
    path_dict = generate_paths(config, path_type='interpretation_net')        
        
    directory = './data/weights/' + 'weights_' + path_dict['path_identifier_lambda_net_data'] + '/'
    path_network_parameters = directory + 'weights' + '.txt'
    path_X_data = directory + 'X_test_lambda.txt'
    path_y_data = directory + 'y_test_lambda.txt'        
    
    network_parameters = pd.read_csv(path_network_parameters, sep=",", header=None)
    network_parameters = network_parameters.sort_values(by=0)
    if no_noise == False:
        network_parameters = network_parameters.sample(n=config['i_net']['interpretation_dataset_size'], random_state=config['computation']['RANDOM_SEED'])
    
    X_test_lambda = pd.read_csv(path_X_data, sep=",", header=None)
    X_test_lambda = X_test_lambda.sort_values(by=0)
    if no_noise == False:
        X_test_lambda = X_test_lambda.sample(n=config['i_net']['interpretation_dataset_size'], random_state=config['computation']['RANDOM_SEED'])
    
    y_test_lambda = pd.read_csv(path_y_data, sep=",", header=None)
    y_test_lambda = y_test_lambda.sort_values(by=0)
    if no_noise == False:
        y_test_lambda = y_test_lambda.sample(n=config['i_net']['interpretation_dataset_size'], random_state=config['computation']['RANDOM_SEED'])
        
        
    parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='loky') #loky

    lambda_nets = parallel(delayed(LambdaNet)(network_parameters_row, 
                                              X_test_lambda_row, 
                                              y_test_lambda_row, 
                                              config) for network_parameters_row, X_test_lambda_row, y_test_lambda_row in zip(network_parameters.values, X_test_lambda.values, y_test_lambda.values))          
    del parallel
    
    base_model = generate_base_model(config)  
    
    def initialize_network_wrapper(config, lambda_net, base_model):
        lambda_net.initialize_network(config, base_model)
    
    parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='sequential')
    _ = parallel(delayed(initialize_network_wrapper)(config, lambda_net, base_model) for lambda_net in lambda_nets)   
    del parallel
    
    def initialize_target_function_wrapper(config, lambda_net):
        lambda_net.initialize_target_function(config)
    
    parallel = Parallel(n_jobs=n_jobs, verbose=3, backend='sequential')
    _ = parallel(delayed(initialize_target_function_wrapper)(config, lambda_net) for lambda_net in lambda_nets)   
    del parallel
        
    
    #lambda_nets = [None] * network_parameters.shape[0]
    #for i, (network_parameters_row, X_test_lambda_row, y_test_lambda_row) in tqdm(enumerate(zip(network_parameters.values, X_test_lambda.values, y_test_lambda.values)), total=network_parameters.values.shape[0]):        
    #    lambda_net = LambdaNet(network_parameters_row, X_test_lambda_row, y_test_lambda_row, config)
    #    lambda_nets[i] = lambda_net
                
    lambda_net_dataset = LambdaNetDataset(lambda_nets)
        
    return lambda_net_dataset
    


# In[11]:


#LOAD DATA
if noise_injected_level > 0:
    lambda_net_dataset_training = load_lambda_nets(config, no_noise=True, n_jobs=n_jobs)
    lambda_net_dataset_evaluation = load_lambda_nets(config, n_jobs=n_jobs)

    lambda_net_dataset_train, lambda_net_dataset_valid = split_LambdaNetDataset(lambda_net_dataset_training, test_split=0.1)
    _, lambda_net_dataset_test = split_LambdaNetDataset(lambda_net_dataset_evaluation, test_split=test_size)
    
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


#get_ipython().run_line_magic('load_ext', 'autoreload')

# In[19]:


#get_ipython().run_line_magic('autoreload', '2')
((X_valid, y_valid), 
 (X_test, y_test),
 history,

 model) = interpretation_net_training(
                                      lambda_net_dataset_train, 
                                      lambda_net_dataset_valid, 
                                      lambda_net_dataset_test,
                                      config,
                                      #callback_names=['plot_losses']
                                     )


# In[20]:


model.summary()


# In[21]:


lambda_net = np.array([lambda_net_dataset_test.network_parameters_array[0]])
X_data = lambda_net_dataset_test.X_test_lambda_array[2]
y_data = lambda_net_dataset_test.y_test_lambda_array[0]
print(lambda_net.shape)
dt_pred = model.predict(lambda_net)[0]
print(dt_pred)


# In[ ]:





# In[22]:


from math import log2
import queue

def level_to_pre(arr,ind,new_arr):
    if ind>=len(arr): return new_arr #nodes at ind don't exist
    new_arr.append(arr[ind]) #append to back of the array
    new_arr = level_to_pre(arr,ind*2+1,new_arr) #recursive call to left
    new_arr = level_to_pre(arr,ind*2+2,new_arr) #recursive call to right
    return new_arr

def pre_to_level(arr):
    def left_tree_size(n):
        if n<=1: return 0
        l = int(log2(n+1)) #l = no of completely filled levels
        ans = 2**(l-1)
        last_level_nodes = min(n-2**l+1,ans)
        return ans + last_level_nodes -1       
    
    que = queue.Queue()
    que.put((0,len(arr)))
    ans = [] #this will be answer
    while not que.empty():
        iroot,size = que.get() #index of root and size of subtree
        if iroot>=len(arr) or size==0: continue ##nodes at iroot don't exist
        else : ans.append(arr[iroot]) #append to back of output array
        sz_of_left = left_tree_size(size) 
        que.put((iroot+1,sz_of_left)) #insert left sub-tree info to que
        que.put((iroot+1+sz_of_left,size-sz_of_left-1)) #right sub-tree info 

    return ans


# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

sklearn_dt = DecisionTreeClassifier(max_depth=2)
#print(sklearn_dt.tree_)
sklearn_dt.fit(X_data, y_data)
print(sklearn_dt.tree_)
sklearn_dt.get_params()


# In[24]:


plot_tree(sklearn_dt)


# In[25]:


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))


# In[26]:


gini(1)


# In[27]:


clf=sklearn_dt
n_nodes = clf.tree_.node_count
print('n_nodes', n_nodes)
children_left = clf.tree_.children_left
print('children_left', children_left)
children_right = clf.tree_.children_right
print('children_right', children_right)
feature = clf.tree_.feature
print('feature', feature)
threshold = clf.tree_.threshold
print('threshold', threshold)

print('clf.tree_.value', clf.tree_.value)
print('clf.tree_.impurity', clf.tree_.impurity)
print('clf.tree_n_node_samples', clf.tree_.n_node_samples)
print('clf.tree_.weighted_n_node_samples', clf.tree_.weighted_n_node_samples)


node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print('node_depth', node_depth)
print('is_leaves', is_leaves)  

print("The binary tree structure has {n} nodes and has "
      "the following tree structure:\n".format(n=n_nodes))
for i in range(n_nodes):
    if is_leaves[i]:
        print("{space}node={node} is a leaf node.".format(
            space=node_depth[i] * "\t", node=i))
    else:
        print("{space}node={node} is a split node: "
              "go to node {left} if X[:, {feature}] <= {threshold} "
              "else to node {right}.".format(
                  space=node_depth[i] * "\t",
                  node=i,
                  left=children_left[i],
                  feature=feature[i],
                  threshold=threshold[i],
                  right=children_right[i]))


# In[28]:


splits, leaf_classes = get_shaped_parameters_for_decision_tree(dt_pred, config)
print(splits)
print(leaf_classes)


# In[29]:

def dt_array_to_sklearn(vanilla_dt_array, config,X_data, y_data, printing=False):
    import copy
    def gini(p):
        return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))    
    
    splits, leaf_classes= get_shaped_parameters_for_decision_tree(vanilla_dt_array, config)
    
    if printing:
        print('splits', splits)
        print('leaf_classes', leaf_classes)
        
    internal_node_num = 2 ** config['function_family']['maximum_depth'] -1    
    leaf_node_num = 2 ** config['function_family']['maximum_depth']    
    n_nodes = internal_node_num + leaf_node_num

    indices_list = [i for i in range(internal_node_num + leaf_node_num)]
    pre_order_from_level = np.array(level_to_pre(indices_list, 0, []))

    level_order_from_pre = np.array(pre_to_level(indices_list))
    children_left = []
    children_right = []
    counter = 0
    for i in pre_order_from_level:#pre_order_from_level:
        left = 2*i+1 
        right = 2*i+2 
        if left < n_nodes:
            children_left.append(level_order_from_pre[left])
        else:
            children_left.append(-1)
        if left < n_nodes:
            children_right.append(level_order_from_pre[right])
        else:
            children_right.append(-1)            
            
        #try:
        #    children_left.append(level_order_from_pre[left])
        #except:
        #    children_left.append(-1)
        #try:
        #    children_right.append(level_order_from_pre[right])
        #except:
        #    children_right.append(-1)            
        
    children_left = np.array(children_left)
    children_right = np.array(children_right)
    
    #print('children_left', children_left.shape, children_left)
    #print('children_right', children_right.shape, children_right)
    
    indices_list = [i for i in range(internal_node_num+leaf_node_num)]
    new_order = np.array(level_to_pre(indices_list, 0, []))
    
    feature = [np.argmax(split) for split in splits]
    feature.extend([-2 for i in range(leaf_node_num)])
    feature = np.array(feature)[new_order]
    threshold = [np.max(split) for split in splits]
    threshold.extend([-2 for i in range(leaf_node_num)])
    threshold = np.array(threshold)[new_order]    
    
    samples = 500
    value_list = []
    n_node_samples_list = []
    impurity_list = []
    
    value_list_previous = None
    for current_depth in reversed(range(1, (config['function_family']['maximum_depth']+1)+1)):
        internal_node_num_current_depth = (2 ** current_depth - 1) - (2 ** (current_depth-1) - 1)
        #print(internal_node_num_current_depth)
        n_node_samples = [samples for _ in range(internal_node_num_current_depth)]
        if current_depth > config['function_family']['maximum_depth']: #is leaf
            values = []
            impurity = []
            for leaf_class in leaf_classes:
                current_value = [samples, 0] if leaf_class == 0 else [0, samples]
                curent_impurity = 0
                values.append(current_value)
                impurity.append(curent_impurity)
            #values = [[0, samples] for _ in range(internal_node_num_current_depth)]
            #impurity = [0.5 for _ in range(internal_node_num_current_depth)]
        else:
            value_list_previous_left = value_list_previous[::2]
            value_list_previous_right = value_list_previous[1::2]
            samples_sum_list = np.add(value_list_previous_left, value_list_previous_right)
            
            values = [samples_sum for samples_sum in samples_sum_list]
            impurity = [gini(value[0]/sum(value)) for value in values]
        samples = samples*2
        
        value_list_previous = values

        
        n_node_samples_list[0:0] = n_node_samples
        value_list[0:0] = values
        impurity_list[0:0] = impurity        
        #n_node_samples_list.extend(n_node_samples)
        #value_list.extend(values)
        #impurity_list.extend(impurity)
        
    value = np.expand_dims(np.array(value_list), axis=1) #shape [node_count, n_outputs, max_n_classes]; number of samples for each class
    value = value[new_order].astype(np.float64)
    impurity =  np.array(impurity_list) #
    impurity = impurity[new_order]
    n_node_samples = np.array(n_node_samples_list) #number of samples at each node
    n_node_samples = n_node_samples[new_order]
    weighted_n_node_samples = 1 * np.array(n_node_samples_list) #same as tree_n_node_samples, but weighted    
    weighted_n_node_samples = weighted_n_node_samples.astype(np.float64)
    n_node_samples = n_node_samples[new_order]
    
    if printing:
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print("The binary tree structure has {n} nodes and has "
              "the following tree structure:\n".format(n=n_nodes))
        for i in range(n_nodes):
            if is_leaves[i]:
                print("{space}node={node} is a leaf node.".format(space=node_depth[i] * "\t", node=i))
            else:
                print("{space}node={node} is a split node: "
                      "go to node {left} if X[:, {feature}] <= {threshold} "
                      "else to node {right}.".format(
                          space=node_depth[i] * "\t",
                          node=i,
                          left=children_left[i],
                          feature=feature[i],
                          threshold=threshold[i],
                          right=children_right[i]))    

        
    clf=DecisionTreeClassifier(max_depth=config['function_family']['maximum_depth'])
    #y_data = [i for i in range(config['data']['num_classes'])]
    #X_data = [[0 for i in range(config['data']['number_of_variables'])] for _ in range(config['data']['num_classes'])]
    clf.fit(X_data, y_data)
    
    print(type(clf.tree_.node_count), type(n_nodes))
    print(type(clf.tree_.capacity), type(n_nodes))    
    clf.tree_.node_count = n_nodes
    clf.tree_.capacity = n_nodes
    
    print(type(clf.tree_.node_count), type(n_nodes))
    print(type(clf.tree_.capacity), type(n_nodes))       
    
    #print(clf.tree_.value, np.array(clf.tree_.value.shape))
    #print(value, np.array(value).shape)
    
    #TODO: FÜR VALUES NICHT IMMER 50/50 BEI INNER UND 100/0 BEI LEAF, SONDERN: BEI LEAFS ANFANGEN UND DANN DEN PFADEN ENTLANG HOCH-ADDIEREN FÜR JEDEN PARENT NODE
    print('-------------------------------------------------------------')
    print(clf.tree_.value.dtype, value.dtype)
    print(clf.tree_.value, value)
    #print(clf.tree_.impurity.dtype, impurity.dtype)
    #print(clf.tree_.n_node_samples.dtype, n_node_samples.dtype)
    #print(clf.tree_.weighted_n_node_samples.dtype, weighted_n_node_samples.dtype)
    #print(clf.tree_.children_left.dtype, children_left.dtype)
    #print(clf.tree_.children_right.dtype, children_right.dtype)
    #print(clf.tree_.feature.dtype, feature.dtype)
    #print(clf.tree_.threshold.dtype, threshold.dtype)
    if True:
        for i in indices_list:
            print(clf.tree_.value[i].dtype, value[i].dtype)
            print(type(clf.tree_.value[i]), type(value[i]))
            print(clf.tree_.value[i], value[i])
            #print(clf.tree_.impurity[i].dtype, impurity[i].dtype)
            #print(clf.tree_.n_node_samples[i].dtype, n_node_samples[i].dtype)
            #print(clf.tree_.weighted_n_node_samples[i].dtype, weighted_n_node_samples[i].dtype)
            #print(clf.tree_.children_left[i].dtype, children_left[i].dtype)
            #print(clf.tree_.children_right[i].dtype, children_right[i].dtype)
            #print(clf.tree_.feature[i].dtype, feature[i].dtype)
            #print(clf.tree_.threshold[i].dtype, threshold[i].dtype)            
            
            
            clf.tree_.children_left[i] = copy.deepcopy(children_left[i])
            clf.tree_.children_right[i] = copy.deepcopy(children_right[i])           
            clf.tree_.value[i] = copy.deepcopy(value[i])
            clf.tree_.impurity[i] = copy.deepcopy(impurity[i])
            clf.tree_.n_node_samples[i] = copy.deepcopy(n_node_samples[i])
            clf.tree_.weighted_n_node_samples[i] = copy.deepcopy(weighted_n_node_samples[i])
            clf.tree_.feature[i] = copy.deepcopy(feature[i])
            clf.tree_.threshold[i] = copy.deepcopy(threshold[i])
    print('-------------------------------------------------------------')
    #print(clf.tree_.children_left)
    print(clf.tree_.value)
    #print(clf.tree_.impurity)
    #print(clf.tree_.n_node_samples)
    #print(clf.tree_.weighted_n_node_samples)
    #print(clf.tree_.children_right)
    #print(clf.tree_.feature)
    #print(clf.tree_.threshold)            
    #print('clf.tree_.max_depth', clf.tree_.max_depth)
    
    return clf



# In[30]:


"""
Attributes
----------
node_count : int
    The number of nodes (internal nodes + leaves) in the tree.
capacity : int
    The current capacity (i.e., size) of the arrays, which is at least as
    great as `node_count`.
max_depth : int
    The depth of the tree, i.e. the maximum depth of its leaves.
children_left : array of int, shape [node_count]
    children_left[i] holds the node id of the left child of node i.
    For leaves, children_left[i] == TREE_LEAF. Otherwise,
    children_left[i] > i. This child handles the case where
    X[:, feature[i]] <= threshold[i].
children_right : array of int, shape [node_count]
    children_right[i] holds the node id of the right child of node i.
    For leaves, children_right[i] == TREE_LEAF. Otherwise,
    children_right[i] > i. This child handles the case where
    X[:, feature[i]] > threshold[i].
feature : array of int, shape [node_count]
    feature[i] holds the feature to split on, for the internal node i.
threshold : array of double, shape [node_count]
    threshold[i] holds the threshold for the internal node i.
value : array of double, shape [node_count, n_outputs, max_n_classes]
    Contains the constant prediction value of each node.
impurity : array of double, shape [node_count]
    impurity[i] holds the impurity (i.e., the value of the splitting
    criterion) at node i.
n_node_samples : array of int, shape [node_count]
    n_node_samples[i] holds the number of training samples reaching node i.
weighted_n_node_samples : array of int, shape [node_count]
    weighted_n_node_samples[i] holds the weighted number of training samples
    reaching node i.
"""


# In[31]:


#new_tree = dt_array_to_sklearn(dt_pred, config, X_data, y_data, printing=True)
new_tree = dt_array_to_sklearn(dt_pred, config, X_data, y_data, printing=True)
print('DONE')

# In[32]:


plt.figure(figsize=(36,12))  # set plot size (denoted in inches)
plot_tree(new_tree, fontsize=10)
plt.show()


# In[33]:


X_data.shape


# In[34]:


sklearn_dt.predict(X_data)


# In[35]:


new_tree.predict(X_data[:50])


# In[36]:


calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(X_data[:50], config)(dt_pred)


# In[37]:


z


# In[ ]:


max_depth = 3
indices_list = [i for i in range(2**(max_depth+1)-1)]
print('indices_list', indices_list)
pre_order_from_level = np.array(level_to_pre(indices_list, 0, []))
print('pre_order_from_level', pre_order_from_level)
leaf_indices_pre_order = np.argwhere(pre_order_from_level>=2**max_depth-1).ravel()
print(leaf_indices_pre_order)
left_indices_pre_order = np.argwhere(pre_order_from_level % 2 != 0).ravel()
right_indices_pre_order = np.argwhere(pre_order_from_level % 2 == 0).ravel()[1:]
print('left_indices_pre_order', left_indices_pre_order)
print('right_indices_pre_order', right_indices_pre_order)

counter = 0
order = []
children_left = []
children_right = []
for i in range(2**(max_depth+1)-1):
    if i in leaf_indices_pre_order:
        order.append(-1)
        if i in left_indices_pre_order:
            children_left.append(-1)
        if i in right_indices_pre_order:
            children_right.append(-1)        
        continue
    else:
        order.append(counter)
        if i in left_indices_pre_order:
            children_left.append(counter)
        if i in right_indices_pre_order:
            children_right.append(counter)           
        counter += 1
order = np.array(order)
children_left = np.array(children_left)
children_right = np.array(children_right)

print('order', order)
print('children_left', children_left)
print('children_right', children_right)


# In[ ]:


model.summary()


# In[ ]:


acc_target_lambda_list = []
bc_target_lambda_list = []

acc_lambda_decision_list = []
bc_lambda_decision_list = []

acc_target_decision_list = []
bc_target_decision_list = []

decision_function_parameters_list = []
decision_functio_list = []

for lambda_net in tqdm(lambda_net_dataset_test.lambda_net_list):
    
    target_function_parameters = lambda_net.target_function_parameters
    target_function = lambda_net.target_function
    
    X_test_lambda = lambda_net.X_test_lambda
    y_test_lambda = lambda_net.y_test_lambda
    
    network = lambda_net.network
    network_parameters = lambda_net.network_parameters
    
    if config['i_net']['convolution_layers'] != None or config['i_net']['lstm_layers'] != None or (config['i_net']['nas'] and config['nas_type']['convolution_layers'] != 'SEQUENTIAL'):
        network_parameters, network_parameters_flat = restructure_data_cnn_lstm(np.array([network_parameters]), config, subsequences=None)    
      
    decision_function_parameters= model.predict(np.array([network_parameters]))[0]
    decision_function = generate_decision_tree_from_array(decision_function_parameters, config)
    
    decision_function_parameters_list.append(decision_function_parameters)
    decision_functio_list.append(decision_function)
    
    y_test_network = network.predict(X_test_lambda)
    y_test_decision_function = decision_function.predict_proba(X_test_lambda)
    y_test_target_function = target_function.predict_proba(X_test_lambda)  
    
    acc_target_lambda = accuracy_score(np.round(y_test_target_function), np.round(y_test_network))
    bc_target_lambda = log_loss(np.round(y_test_target_function), y_test_network, labels=[0, 1])
    
    acc_lambda_decision = accuracy_score(np.round(y_test_network), np.round(y_test_decision_function))
    bc_lambda_decision = log_loss(np.round(y_test_network), y_test_decision_function, labels=[0, 1])        
    
    acc_target_decision = accuracy_score(np.round(y_test_target_function), np.round(y_test_decision_function))
    bc_target_decision = log_loss(np.round(y_test_target_function), y_test_decision_function, labels=[0, 1])   
    
    
    acc_target_lambda_list.append(acc_target_lambda)
    bc_target_lambda_list.append(bc_target_lambda)

    acc_lambda_decision_list.append(acc_lambda_decision)
    bc_lambda_decision_list.append(bc_lambda_decision)

    acc_target_decision_list.append(acc_target_decision)
    bc_target_decision_list.append(bc_target_decision)
    

acc_target_lambda_array = np.array(acc_target_lambda_list)
bc_target_lambda_array = np.array(bc_target_lambda_list)

acc_lambda_decision_array = np.array(acc_lambda_decision_list)
bc_lambda_decision_array = np.array(bc_lambda_decision_list)

acc_target_decision_array = np.array(acc_target_decision_list)
bc_target_decision_array = np.array(bc_target_decision_list)
    
    
acc_target_lambda = np.mean(acc_target_lambda_array)
bc_target_lambda = np.mean(bc_target_lambda_array[~np.isnan(bc_target_lambda_array)])

acc_lambda_decision = np.mean(acc_lambda_decision_array)
bc_lambda_decision = np.mean(bc_lambda_decision_array[~np.isnan(bc_lambda_decision_array)])

acc_target_decision = np.mean(acc_target_decision_array)
bc_target_decision = np.mean(bc_target_decision_array[~np.isnan(bc_target_decision_array)])


print('Accuracy Target Lambda', acc_target_lambda)
print('Binary Crossentropy Target Lambda', bc_target_lambda)
print('Accuracy Lambda Decision', acc_lambda_decision)
print('Binary Crossentropy Lambda Decision', bc_lambda_decision)
print('Accuracy Target Decision', acc_target_decision)
print('Binary Crossentropy Target Decision', bc_target_decision)


# In[ ]:


X_test_lambda


# In[ ]:


np.round(y_test_network).ravel()[:100]


# In[ ]:


np.round(y_test_decision_function).ravel()[:100]


# In[ ]:


acc_lambda_decision_array


# In[ ]:


# TODO BENCHMARK RANDOM GUESS


# In[ ]:




