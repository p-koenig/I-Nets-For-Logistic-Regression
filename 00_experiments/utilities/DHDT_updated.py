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

from utilities.utilities_updated import *
from utilities.DHDT_updated import *

from joblib import Parallel, delayed

from itertools import product
from collections.abc import Iterable

from copy import deepcopy



def make_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]    
        
def sigmoid(x, factor=1, shift_horizontal=0.0):
    x = 1/(1+K.exp(-factor*(x-shift_horizontal)))
    return x  

def tanh(x, factor=1, shift_horizontal=0, shift_vertical=0):
    x = (K.exp(factor*(x-shift_horizontal))-K.exp(-factor*(x-shift_horizontal)))/(K.exp(factor*(x-shift_horizontal))+K.exp(-factor*(x-shift_horizontal))) + shift_vertical
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
        
            sparse_activation_1 = 'softmax',
            sparse_activation_2 = 'sigmoid',
        
            activation = 'sigmoid',
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
        
        self.sparse_activation_1 = sparse_activation_1
        self.sparse_activation_2 = sparse_activation_2
        
        self.seed = random_seed
        self.verbosity = verbosity
        self.number_of_variables = number_of_variables
        
        self.activation = activation
        self.squeeze_factor = squeeze_factor
        
        self.internal_node_num_ = 2 ** self.depth - 1 
        self.leaf_node_num_ = 2 ** self.depth
        
        tf.random.set_seed(self.seed)
                        
        #internal_nodes, leaf_nodes = self.get_shaped_parameters_for_decision_tree(dt_params_activation)   
        
        if initializer in ['Zeros', 'Ones', 'zeros', 'ones']:
            self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': initializer, 'config': {}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_values')        
        else:
            self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': initializer, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_values')
        
        if initializer_index in ['Zeros', 'Ones', 'zeros', 'ones']:
            self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer_index, 'config': {}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_index_array')
        else:   
            self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer_index, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),
                                      trainable=True,
                                      name='split_index_array')            
            
        if initializer in ['Zeros', 'Ones', 'zeros', 'ones']:
            self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': initializer, 'config': {}})(shape=(self.leaf_node_num_,)),
                                      trainable=True,
                                      name='leaf_classes_array')            
        else:
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

        
        if self.sparse_activation_1 == 'softmax':
            split_index_array_complete = tf.keras.activations.softmax(self.beta_1 * self.split_index_array)
        elif self.sparse_activation_1 == 'entmax':
            split_index_array_complete = entmax15(self.beta_1 * self.split_index_array)           
        elif self.sparse_activation_1 == 'sparsemax':
            split_index_array_complete = tfa.activations.sparsemax(self.beta_1 * self.split_index_array)
            
        if self.activation == 'sigmoid':
            split_values_complete = sigmoid(self.split_values, factor=self.squeeze_factor, shift_horizontal=0)
        elif self.activation == 'tanh':
            split_values_complete = tanh(self.split_values, factor=self.squeeze_factor, shift_horizontal=0, shift_vertical=0)        
        
        X_by_index = tf.reduce_sum(tf.expand_dims(split_index_array_complete, 1)*X, axis=2)
        
        split_values_by_index = tf.expand_dims(tf.reduce_sum(split_values_complete*split_index_array_complete, axis=1), 1)
               
        if self.sparse_activation_2 == 'sigmoid':
            internal_node_result_complete = tf.sigmoid(self.beta_2 * (X_by_index - split_values_by_index)) 
        elif self.sparse_activation_2 == 'entmax':
            internal_node_result_complete = tf.squeeze(tf.squeeze(entmax15(self.beta_2 * tf.concat([tf.expand_dims((X_by_index-split_values_by_index), 2), tf.expand_dims((-(X_by_index-split_values_by_index)), 2)], 2)))[:,:,:1])
        elif self.sparse_activation_2 == 'sparsemax':
            internal_node_result_complete = tf.squeeze(tf.squeeze(tfa.activations.sparsemax([self.beta_2 * tf.concat([tf.expand_dims((X_by_index-split_values_by_index), 2), tf.expand_dims((-(X_by_index-split_values_by_index)), 2)], 2)]))[:,:,:1]) 
        
        #internal_node_result_complete = tf.cast(tf.greater(X_by_index, split_values_by_index), tf.float32)#tf.sigmoid(self.beta_2 * (X_by_index - split_values_by_index - 0.5)) ##tf.greater?

        #tf.print(internal_node_result_complete, summarize=-1)
        
        begin_idx = 0
        end_idx = 1

        layer_result = internal_node_result_complete[begin_idx:end_idx,:]

        layer_result_combined = tf.reshape(tf.stack([layer_result, (1-layer_result)], axis=1), [2**1, X.shape[0]])

        path_results_complete = layer_result_combined

        begin_idx = end_idx
        end_idx = begin_idx + 2 ** 1

        #print('___________________')
        #print(path_results_complete)
        #print('___________________')
        
        #print('self.depth', self.depth)
        for layer_idx in range(1, self.depth):
            #print('layer_idx', layer_idx)
            layer_result = internal_node_result_complete[begin_idx:end_idx,:]

            layer_result_combined = tf.stack([layer_result, (1-layer_result)], axis=1)
            layer_result_combined = tf.reshape(layer_result_combined, [2**(layer_idx+1),X.shape[0]])

            path_results_complete_reshaped = tf.split(path_results_complete, 2**(layer_idx))
            layer_result_combined_reshaped = tf.split(layer_result_combined, 2**(layer_idx))
            
            path_results_complete = tf.reshape(tf.multiply(path_results_complete_reshaped, layer_result_combined_reshaped),  [2**(layer_idx+1), X.shape[0]])

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

            #print('path_results_complete', path_results_complete)
            #print('___________________')

            
            #tf.print(path_results_complete, summarize=-1)
            
        #function_values_dhdt = tf.reduce_sum(path_results_complete*tf.expand_dims(self.leaf_classes_array, 1), axis=0)
        function_values_dhdt = tf.reduce_sum(tf.transpose(path_results_complete)*self.leaf_classes_array, axis=1)
            
        return function_values_dhdt  
           
    
    @tf.function(jit_compile=True)                    
    def forward_hard(self, X):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)               

        split_index_array_complete = tfa.seq2seq.hardmax(self.split_index_array)

        if self.activation == 'sigmoid':
            split_values_complete = sigmoid(self.split_values, factor=self.squeeze_factor, shift_horizontal=0)
        elif self.activation == 'tanh':
            split_values_complete = tanh(self.split_values, factor=self.squeeze_factor, shift_horizontal=0, shift_vertical=0)        
        
        X_by_index = tf.reduce_sum(tf.expand_dims(split_index_array_complete, 1)*X, axis=2)
        
        split_values_by_index =tf.expand_dims(tf.reduce_sum(split_values_complete*split_index_array_complete, axis=1), 1)
        
        internal_node_result_complete = tf.cast(tf.greater(X_by_index, split_values_by_index), tf.float32) #tf.round(tf.sigmoid(X_by_index - split_values_by_index)) ##tf.greater? ##ADJUSTED
        
        begin_idx = 0
        end_idx = 1

        layer_result = internal_node_result_complete[begin_idx:end_idx,:]

        layer_result_combined = tf.reshape(tf.stack([layer_result, (1-layer_result)], axis=1), [2**1, X.shape[0]])

        path_results_complete = layer_result_combined

        begin_idx = end_idx
        end_idx = begin_idx + 2 ** (0 + 1)

        #print('___________________')
        #print(path_results_complete)
        #print('___________________')
        
        #print('self.depth', self.depth)
        for layer_idx in range(1, self.depth):
            #print('layer_idx', layer_idx)
            layer_result = internal_node_result_complete[begin_idx:end_idx,:]

            layer_result_combined = tf.stack([layer_result, (1-layer_result)], axis=1)
            layer_result_combined = tf.reshape(layer_result_combined, [2**(layer_idx+1),X.shape[0]])

            path_results_complete = tf.reshape(tf.multiply(tf.split(path_results_complete, 2**(layer_idx)), tf.split(layer_result_combined, 2**(layer_idx))),  [2**(layer_idx+1),X.shape[0]])

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

            #print('path_results_complete', path_results_complete)
            #print('___________________')

            
            #tf.print(path_results_complete, summarize=-1)
            
        function_values_dhdt = tf.reduce_sum(path_results_complete*tf.expand_dims(self.leaf_classes_array, 1), axis=0)        
        
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
        if self.verbosity > 3:
            tf.print('grads leaf_classes_array', np.round(grads, 5), summarize=-1)       
            
        grads = tape.gradient(current_loss, self.split_values)
        self.optimizer.apply_gradients(zip([grads], [self.split_values]))
        if self.verbosity > 3:
            tf.print('grads split_values', np.round(grads, 5), summarize=-1)        
            
        grads = tape.gradient(current_loss, self.split_index_array)
        self.optimizer.apply_gradients(zip([grads], [self.split_index_array]))
        if self.verbosity > 3:
            tf.print('grads split_index_array', np.round(grads, 5), summarize=-1)
        
        return current_loss
        
    def plot(self, normalizer_list=None, path='./dt_plot.png'):
        from anytree import Node, RenderTree
        from anytree.exporter import DotExporter

        split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(self.split_index_array)
        #split_values_complete = sigmoid_squeeze(self.split_values, self.squeeze_factor)
        if self.activation == 'sigmoid':
            split_values_list_by_internal_node = sigmoid(self.split_values, factor=self.squeeze_factor, shift_horizontal=0)
        elif self.activation == 'tanh':
            split_values_list_by_internal_node = tanh(self.split_values, factor=self.squeeze_factor, shift_horizontal=0, shift_vertical=0)                
        #tf.print('split_index_list_by_internal_node_max', split_index_list_by_internal_node_max)
        #tf.print('split_values_list_by_internal_node', split_values_list_by_internal_node)
        splits = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))
        #tf.print('splits', splits)
        
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

        
        
def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True) 
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs
    
        return outputs, grad_fn
    
    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
    
    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]
    
    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size