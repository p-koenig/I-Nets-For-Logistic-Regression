import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder

from livelossplot import PlotLosses

import os
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '' #'true'

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

np.seterr(all="ignore")

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


import seaborn as sns
sns.set_style("darkgrid")

import time
import random

from utilities.utilities import *
from utilities.DHDT import *

from joblib import Parallel, delayed

from itertools import product
from collections.abc import Iterable

from copy import deepcopy

from utilities.utilities import *


def make_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]    
        
def sigmoid_squeeze(x, factor=3):
    x = 1/(1+K.exp(-factor*x))
    return x  

class DHDT(tf.Module):
    
    def __init__(
            self,
            number_of_variables,
        
            depth = 3,
        
            learning_rate = 1e-3,
            optimizer = 'adam',
        
            beta_1 = 100,
            beta_2 = 100,
        
            squeeze_factor = 1,
        
            loss = 'binary_crossentropy',#'mae',
        
            initializer = 'RandomNormal', #GlorotUniform
            initializer_index = 'RandomNormal',
        
            random_seed = 42,
            verbosity = 1):    
                
        self.depth = depth
        
        self.learning_rate = learning_rate
        self.loss = tf.keras.losses.get(loss)
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        self.seed = random_seed
        self.verbosity = verbosity
        self.number_of_variables = number_of_variables
        self.squeeze_factor = squeeze_factor
        
        self.internal_node_num_ = 2 ** self.depth - 1 
        self.leaf_node_num_ = 2 ** self.depth
        
        tf.random.set_seed(self.seed)
                        
        maximum_depth = self.depth
        leaf_node_num_ = 2 ** maximum_depth
        internal_node_num_ = 2 ** maximum_depth - 1
                
        #internal_nodes, leaf_nodes = self.get_shaped_parameters_for_decision_tree(dt_params_activation)   
        
        self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_values')
        
        if initializer_index in ['zeros', 'ones']:
            self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer_index, 'config': {}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_index_array')
        else:   
            self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer_index, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_index_array')            
            
        self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer, 'config': {'seed': self.seed}})(shape=(self.leaf_node_num_,)),
                                      trainable=True,
                                      name='leaf_classes_array')
        
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.optimizer.learning_rate = self.learning_rate
                
        #tf.print('self.split_values', self.split_values)
        #tf.print('self.split_index_array', self.split_index_array)
        #tf.print('self.leaf_classes_array', self.leaf_classes_array)
            
        self.plotlosses = PlotLosses()    
        
    def fit(self, X_train, y_train, batch_size=256, epochs=100, early_stopping_epochs=5, valid_data=None):
                
        minimum_loss_epoch = np.inf
        epochs_without_improvement = 0    
        
        batch_size = min(batch_size, X_train.shape[0])
        
        disable = True if self.verbosity == -1 else False
        for current_epoch in tqdm(range(epochs), desc='epochs', disable=disable):
            tf.random.set_seed(self.seed + current_epoch)
            X_train = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
            tf.random.set_seed(self.seed + current_epoch)
            y_train = tf.random.shuffle(y_train, seed=self.seed + current_epoch)
            
            loss_list = []
            for index, (X_batch, y_batch) in enumerate(zip(make_batch(X_train, batch_size), make_batch(y_train, batch_size))):
                current_loss = self.backward(X_batch, y_batch)
                loss_list.append(float(current_loss))
                
                if self.verbosity > 2:
                    batch_idx = (index+1)*batch_size
                    msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                    print(msg.format(current_epoch, batch_idx, current_loss))                   
                  
            current_loss_epoch = np.mean(loss_list)
            if self.verbosity > 1:    
                msg = "Epoch: {:02d} | Loss: {:.5f} |"
                print(msg.format(current_epoch, current_loss_epoch))              

            
            if self.verbosity == 1:  
                loss_dict = {'loss': current_loss_epoch}

                loss_dict['acc'] = accuracy_score(y_train, np.round(tf.sigmoid(self.forward_hard(X_train))))
                
                if valid_data is not None:
                    if self.loss.__name__  == 'binary_crossentropy':
                        loss_dict['val_loss'] = self.loss(valid_data[1], self.forward(valid_data[0]), from_logits=True)
                    else:
                        loss_dict['val_loss'] = self.loss(valid_data[1], tf.sigmoid(self.forward(valid_data[0])))                   
                    loss_dict['val_acc'] = accuracy_score(valid_data[1], np.round(tf.sigmoid(self.forward_hard(valid_data[0]))))
                self.plotlosses.update(loss_dict)#({'acc': 0.0, 'val_acc': 0.0, 'loss': np.mean(loss_list), 'val_loss': 0.0})
                self.plotlosses.send()            

            if current_loss_epoch < minimum_loss_epoch:
                minimum_loss_epoch = current_loss_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_epochs:
                break
    
    
    
    @tf.function(jit_compile=True)                    
    def forward(self, X):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)               

        paths = [[0,1,3], [0,1,4], [0,2,5], [0,2,6]]

        split_index_array_complete = tfa.activations.sparsemax(self.beta_1 * self.split_index_array)
        #split_values_complete = sigmoid_squeeze(self.split_values, self.squeeze_factor)
        split_values_complete = sigmoid_squeeze(self.split_values-0.5, self.squeeze_factor)
        
        function_values_dhdt = np.zeros(shape=X.shape[0])
        for leaf_index, path in enumerate(paths):
            path_result_left = 1
            path_result_right = 1
            for internal_node_index in path: 
                #split_index = tfa.activations.sparsemax(self.beta_1 * self.split_index_array[internal_node_index])
                #split_values = sigmoid_squeeze(self.split_values[internal_node_index]-0.5, self.squeeze_factor)

                split_index = split_index_array_complete[internal_node_index]
                split_values = split_values_complete[internal_node_index]
                
                internal_node_split_value = tf.reduce_sum(split_index*split_values)
                respective_input_value = tf.reduce_sum(split_index*X, axis=1)

                split_decision = tf.sigmoid(self.beta_2 * (respective_input_value - internal_node_split_value - 0.5))

                path_result_left *= split_decision
                path_result_right *= (1 - split_decision)

            function_values_dhdt += self.leaf_classes_array[leaf_index*2] * path_result_left + self.leaf_classes_array[leaf_index*2+1] * path_result_right
        
        return function_values_dhdt  
           
    
    @tf.function(jit_compile=True)                    
    def forward_hard(self, X):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)               
        
        paths = [[0,1,3], [0,1,4], [0,2,5], [0,2,6]]

        split_index_array_complete = tfa.seq2seq.hardmax(self.split_index_array)
        #split_values_complete = sigmoid_squeeze(self.split_values, self.squeeze_factor)
        split_values_complete = sigmoid_squeeze(self.split_values-0.5, self.squeeze_factor)
        
        function_values_dhdt = np.zeros(shape=X.shape[0])
        for leaf_index, path in enumerate(paths):
            path_result_left = 1
            path_result_right = 1
            for internal_node_index in path: 
                #split_index = tfa.seq2seq.hardmax(self.split_index_array[internal_node_index])
                #split_values = sigmoid_squeeze(self.split_values[internal_node_index]-0.5, self.squeeze_factor)
                
                split_index = split_index_array_complete[internal_node_index]
                split_values = split_values_complete[internal_node_index]
                
                internal_node_split_value = tf.reduce_sum(split_index*split_values)
                respective_input_value = tf.reduce_sum(split_index*X, axis=1)

                split_decision = tf.round(tf.sigmoid(respective_input_value - internal_node_split_value - 0.5))


                path_result_left *= split_decision
                path_result_right *= (1 - split_decision)

            function_values_dhdt += self.leaf_classes_array[leaf_index*2] * path_result_left + self.leaf_classes_array[leaf_index*2+1] * path_result_right

        
        return function_values_dhdt  
               
    def predict(self, X):
        return tf.sigmoid(self.forward_hard(X))
        
    def backward(self, x,y):
        #optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)#tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        with tf.GradientTape(persistent=True) as tape:
            predicted = self.forward(x)
            if self.loss.__name__  == 'binary_crossentropy':
                current_loss = self.loss(y, predicted, from_logits=True)
            else:
                current_loss = self.loss(y, tf.sigmoid(predicted))
        #tf.print('predicted', predicted)
        #tf.print('current_loss', current_loss, summarize=-1)
        grads = tape.gradient(current_loss, self.leaf_classes_array)
        self.optimizer.apply_gradients(zip([grads], [self.leaf_classes_array]))
        #tf.print('grads', grads, summarize=-1)        
        
        grads = tape.gradient(current_loss, self.split_values)
        self.optimizer.apply_gradients(zip([grads], [self.split_values]))
        #tf.print('grads', tf.reshape(grads, (self.internal_node_num_, self.number_of_variables)), summarize=-1)
        grads = tape.gradient(current_loss, self.split_index_array)
        self.optimizer.apply_gradients(zip([grads], [self.split_index_array]))
        #tf.print('grads', tf.reshape(grads, (self.internal_node_num_, self.number_of_variables)), summarize=-1)

        #                          global_step=tf.compat.v1.train.get_or_create_global_step())     
        
        return current_loss
        
    def plot(self, normalizer_list=None, path='./dt_plot.png'):
        from anytree import Node, RenderTree
        from anytree.exporter import DotExporter

        internal_node_num_ = 2 ** self.depth - 1 
        
        #split_values = self.split_values
        split_values = sigmoid_squeeze(self.split_values, self.squeeze_factor)
        split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

        split_index_array = self.split_index_array 
        split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)         

        split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(split_index_list_by_internal_node)#tfa.activations.sparsemax(split_index_list_by_internal_node)

        splits = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))

        
        splits = splits.numpy()
        leaf_classes = tf.sigmoid(self.leaf_classes_array).numpy()


        if normalizer_list is not None: 
            transpose = splits.transpose()
            transpose_normalized = []
            for i, column in enumerate(transpose):
                column_new = column
                if len(column_new[column_new != 0]) != 0:
                    column_new[column_new != 0] = normalizer_list[i].inverse_transform(column[column != 0].reshape(-1, 1)).ravel()
                #column_new = normalizer_list[i].inverse_transform(column.reshape(-1, 1)).ravel()
                transpose_normalized.append(column_new)
            splits = np.array(transpose_normalized).transpose()

        splits_by_layer = []
        for i in range(self.depth+1):
            start = 2**i - 1
            end = 2**(i+1) -1
            splits_by_layer.append(splits[start:end])

        nodes = {
        }
        #tree = Tree()
        for i, splits in enumerate(splits_by_layer):
            for j, split in enumerate(splits):
                if i == 0:
                    current_node_id = int(2**i - 1 + j)
                    name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                    split_variable = np.argmax(np.abs(split))
                    split_value = np.round(split[split_variable], 3)
                    split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)

                    nodes[name] = Node(name=name, display_name=split_description)

                    #tree.create_node(tag=split_description, identifier=name, data=None)            
                else:
                    current_node_id = int(2**i - 1 + j)
                    name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                    parent_node_id = int(np.floor((current_node_id-1)/2))
                    parent_name = 'n' + str(parent_node_id)
                    split_variable = np.argmax(np.abs(split))
                    split_value = np.round(split[split_variable], 3)
                    split_description = 'x' + str(split_variable) + ' <= '  + str(split_value)

                    nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)

                    #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)

        for j, leaf_class in enumerate(leaf_classes):
            i = self.depth
            current_node_id = int(2**i - 1 + j)
            name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
            parent_node_id = int(np.floor((current_node_id-1)/2))
            parent_name = 'n' + str(parent_node_id)
            #split_variable = np.argmax(np.abs(split))
            #split_value = np.round(split[split_variable], 3)
            split_description = str(np.round((leaf_class), 3))#'x' + str(split_variable) + ' <= '  + str(split_value)
            nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)
            #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)        

            DotExporter(nodes['n0'], nodeattrfunc=lambda node: 'label="{}"'.format(node.display_name)).to_picture(path)


        return Image(path)#, nodes#nodes#tree        

        
    