import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from anytree import Node, RenderTree
from anytree.exporter import DotExporter
    
import itertools
from IPython.display import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utilities.utility_functions import *

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from copy import deepcopy

#################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################
###################This is the pytorch implementation on Soft Decision Tree (SDT), appearing in the paper "Distilling a Neural Network Into a Soft Decision Tree". 2017 (https://arxiv.org/abs/1711.09784).######################
#####################################################################################Source: https://github.com/xuyxu/Soft-Decision-Tree ########################################################################################
#################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################
 
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

#def batch(iterable, size):
#    it = iter(iterable)
#    while item := list(itertools.islice(it, size)):
#        yield item
    
    
class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.
    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.
    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            depth=3,
            lamda=1e-3,
            lr=1e-2,
            weight_decaly=5e-4,
            beta=1, #temperature
            decision_sparsity=-1, #number of variables in each split (-1 means all variables)
            criterion=nn.CrossEntropyLoss(),
            maximum_path_probability = True,
            random_seed=42,
            use_cuda=False,
            #device_number = 0,
            verbosity=1): #0=no verbosity, 1= epoch lvl verbosity, 2=batch lvl verbosity, 3=additional prints
        super(SDT, self).__init__()
        
        torch.manual_seed(random_seed)
        torch.set_num_threads(1)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.maximum_path_probability = maximum_path_probability

        self.depth = depth
        self.beta = beta
        self.decision_sparsity = decision_sparsity
        self.lamda = lamda
        
        #self.device = torch.device("cuda" if use_cuda else "cpu")
        
        self.verbosity = verbosity

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim, self.internal_node_num_, bias=True),
            #nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)
        
        if False:
            if self.decision_sparsity != -1 and self.decision_sparsity != self.input_dim:
                vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)

                weights = torch.zeros_like(self.inner_nodes[0].weight)
                for i, idx in enumerate(idx_list):
                    weights[i][idx] = self.inner_nodes[0].weight[i][idx]
                self.inner_nodes[0].weight = torch.nn.Parameter(weights)    
            
                    
        self.criterion = criterion
        #self.device = torch.device("cuda" if use_cuda else "cpu", 0)
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
        self.optimizer = torch.optim.Adam(self.parameters(),
                             lr=lr,
                             weight_decay=weight_decaly)        

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        
        #maximum_path_probability
        
        if self.verbosity > 3:
            print('_mu, _penalty', _mu, _penalty)
            
        if self.maximum_path_probability:
            cond = torch.eq(_mu, torch.max(_mu, axis=1).values.reshape(-1,1))
            _mu = torch.where(cond, _mu, torch.zeros_like(_mu))
            if self.verbosity > 3:
                print('_mu', _mu)
            
        if self.verbosity > 3:
            print('leaf_nodes', self.leaf_nodes)
            print('leaf_nodes', self.leaf_nodes.weight)
            print('_mu', _mu)
        y_pred = self.leaf_nodes(_mu)
        
        if self.verbosity > 3:
            print('y_pred', y_pred)
        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        #X = self._data_augment(X)
        if self.verbosity > 3:
            print('X', X)
            
        if self.decision_sparsity != -1 and self.decision_sparsity != self.input_dim:
            if self.decision_sparsity == 1:
                inner_nodes = deepcopy(self.inner_nodes)
                #vals_list, idx_list = torch.topk(torch.abs(inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)
                #weights = torch.zeros_like(inner_nodes[0].weight)
                #for i, idx in enumerate(idx_list):
                #    weights[i][idx] = inner_nodes[0].weight[i][idx]
                inner_nodes[0].weight = torch.nn.Parameter(inner_nodes[0].weight * nn.Softmax()(1000*torch.abs(inner_nodes[0].weight)))
                path_prob = nn.Sigmoid()(self.beta*inner_nodes(X))            
            
            else:
                inner_nodes = deepcopy(self.inner_nodes)
                vals_list, idx_list = torch.topk(torch.abs(inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)
                weights = torch.zeros_like(inner_nodes[0].weight)
                for i, idx in enumerate(idx_list):
                    weights[i][idx] = inner_nodes[0].weight[i][idx]
                inner_nodes[0].weight = torch.nn.Parameter(weights) 
                path_prob = nn.Sigmoid()(self.beta*inner_nodes(X))
        else:   
            path_prob = nn.Sigmoid()(self.beta*self.inner_nodes(X))
            
        if self.verbosity > 3:
            print('path_prob', path_prob)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        if self.verbosity > 3:
            print('path_prob', path_prob)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        if self.verbosity > 3:
            print('path_prob', path_prob)
        
            
        
        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)
        if self.verbosity > 3:
            print('_mu', _mu)
            print('_penalty', _penalty)
        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            if self.verbosity > 3:
                #print('_penalty loop', _penalty)    
                print('_mu updated loop', _mu) 
                
            _mu = _mu * _path_prob  # update path probabilities

            if self.verbosity > 3:
                #print('_penalty loop', _penalty)    
                print('_mu updated loop', _mu)      
                
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        if self.verbosity > 3:
            print('_mu updated', _mu)
        mu = _mu.view(batch_size, self.leaf_node_num_)
        if self.verbosity > 3:
            print('mu', mu)
        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))

            
    def fit(self, X, y, batch_size=32, epochs=100, early_stopping_epochs=5):
        self.train()
        X, y = torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device)
        
        #self.verbosity = 1
        minimum_loss = np.inf
        epochs_without_improvement = 0
        
        #epochs = 1
        
        if self.verbosity > 0:

            for epoch in tqdm(range(epochs)):

                if epochs_without_improvement < early_stopping_epochs:
                    correct_counter = 0
                    loss_list = []
                    for index, (data, target) in enumerate(zip(batch(X, batch_size), batch(y, batch_size))):

                        #data = torch.stack(data).to(self.device)
                        #target =  torch.stack(target).to(self.device)    

                        output, penalty = self.forward(data, is_training_data=True)
                        loss = self.criterion(output, target.view(-1))

                        loss += penalty

                        #print(self.inner_nodes[0].weight)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()     

                        #print('loss',loss)

                        if False:
                            if self.decision_sparsity != -1 and self.decision_sparsity != self.input_dim:
                                vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)
                                weights = torch.zeros_like(self.inner_nodes[0].weight)
                                for i, idx in enumerate(idx_list):
                                    weights[i][idx] = self.inner_nodes[0].weight[i][idx]
                                self.inner_nodes[0].weight = torch.nn.Parameter(weights) 

                        #print(self.inner_nodes[0].weight)
                        pred = output.data.max(1)[1]
                        correct = pred.eq(target.view(-1).data).sum()
                        batch_idx = (index+1)*batch_size

                        loss_list.append(float(loss))
                        #loss_list.append(float(loss))
                        #print('loss_list', loss_list)

                        correct_counter += correct
                        if self.verbosity > 2:
                            msg = (
                                "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                                " Correct: {:03d}/{:03d}"
                            )
                            print(msg.format(epoch, batch_idx, loss, int(correct), int(data.shape[0])))


                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()                    
                    if self.verbosity > 1:

                        #batch_idx = (index+1)*batch_size

                        #print('loss_list', loss_list)
                        #print('torch.FloatTensor(loss_list)', torch.FloatTensor(loss_list))
                        #print('torch.mean(loss_list)', torch.mean(torch.FloatTensor(loss_list)))
                        #print('float(torch.mean(loss_list))', float(torch.mean(torch.FloatTensor(loss_list))))
                        #print('correct_counter', correct_counter)
                        #print('X.shape[0]', X.shape[0])

                        msg = (
                            "Epoch: {:02d} | Loss: {:.5f} |"
                            " Correct: {:03d}/{:03d}"
                        )
                        print(msg.format(epoch, np.mean(loss_list), int(correct_counter), int(X.shape[0])))   

                    current_loss = np.mean(loss_list)#torch.mean(torch.FloatTensor(loss_list))

                    if current_loss < minimum_loss:
                        minimum_loss = current_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                else:
                    break

        else:
            

            for epoch in range(epochs):

                if epochs_without_improvement < early_stopping_epochs:
                    correct_counter = 0
                    loss_list = []
                    for index, (data, target) in enumerate(zip(batch(X, batch_size), batch(y, batch_size))):

                        #data = torch.stack(data).to(self.device)
                        #target =  torch.stack(target).to(self.device)    

                        output, penalty = self.forward(data, is_training_data=True)
                        loss = self.criterion(output, target.view(-1))

                        loss += penalty

                        #print(self.inner_nodes[0].weight)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()     

                        #print('loss',loss)

                        if False:
                            if self.decision_sparsity != -1 and self.decision_sparsity != self.input_dim:
                                vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)
                                weights = torch.zeros_like(self.inner_nodes[0].weight)
                                for i, idx in enumerate(idx_list):
                                    weights[i][idx] = self.inner_nodes[0].weight[i][idx]
                                self.inner_nodes[0].weight = torch.nn.Parameter(weights) 

                        #print(self.inner_nodes[0].weight)
                        pred = output.data.max(1)[1]
                        correct = pred.eq(target.view(-1).data).sum()
                        batch_idx = (index+1)*batch_size

                        loss_list.append(float(loss))
                        #loss_list.append(float(loss))
                        #print('loss_list', loss_list)

                        correct_counter += correct
                        if self.verbosity > 2:
                            msg = (
                                "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                                " Correct: {:03d}/{:03d}"
                            )
                            print(msg.format(epoch, batch_idx, loss, int(correct), int(data.shape[0])))


                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()                    
                    if self.verbosity > 1:

                        #batch_idx = (index+1)*batch_size

                        #print('loss_list', loss_list)
                        #print('torch.FloatTensor(loss_list)', torch.FloatTensor(loss_list))
                        #print('torch.mean(loss_list)', torch.mean(torch.FloatTensor(loss_list)))
                        #print('float(torch.mean(loss_list))', float(torch.mean(torch.FloatTensor(loss_list))))
                        #print('correct_counter', correct_counter)
                        #print('X.shape[0]', X.shape[0])

                        msg = (
                            "Epoch: {:02d} | Loss: {:.5f} |"
                            " Correct: {:03d}/{:03d}"
                        )
                        print(msg.format(epoch, np.mean(loss_list), int(correct_counter), int(X.shape[0])))   

                    current_loss = np.mean(loss_list)#torch.mean(torch.FloatTensor(loss_list))

                    if current_loss < minimum_loss:
                        minimum_loss = current_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                else:
                    break
            
            
        if self.decision_sparsity != -1 and self.decision_sparsity != self.input_dim:
            vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)
            weights = torch.zeros_like(self.inner_nodes[0].weight)
            for i, idx in enumerate(idx_list):
                weights[i][idx] = self.inner_nodes[0].weight[i][idx]
            self.inner_nodes[0].weight = torch.nn.Parameter(weights) 
            
    def evaluate(self, X, y):
        self.eval()
        
        correct = 0.

        data, target = torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device)#data.to(device), target.to(device)
        
        output = F.softmax(self.forward(data), dim=1)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1).data).sum()

        accuracy = float(correct) / target.shape[0]
        
        if self.verbosity > 1:
            msg = (
                "\nTesting Accuracy: {}/{} ({:.3f}%)\n"
            )
            print(
                msg.format(
                    correct,
                    target.shape[0],
                    100.0 * accuracy,
                )
            )        
        
        return accuracy
    
    def predict_proba(self, X):
        if self.output_dim == 2:
            self.eval()

            data = torch.FloatTensor(X).to(self.device)
            output = F.softmax(self.forward(data), dim=1)
            #print(output[:,1:].data)
            #print(output[:,1:].data.reshape(1,-1))
            if self.verbosity > 3:
                print('output', output)

            pred = output[:,1:].data#output.data.max(1)[1]
            if self.verbosity > 3:
                print('pred', pred)        

            predictions = pred.numpy()



            return predictions       
        
        return None
            
            
    
    def predict(self, X):
        self.eval()
        
        data = torch.FloatTensor(X).to(self.device)
        output = F.softmax(self.forward(data), dim=1)
        if self.verbosity > 3:
            print('output', output)
        
        pred = output.data.max(1)[1]
        if self.verbosity > 3:
            print('pred', pred)        
        
        predictions = pred
        
        if False:
            predictions = np.full(X.shape[0], np.nan)
            for index, data_point in enumerate(X):

                data_point = torch.FloatTensor([data_point]).to(self.device)

                output = F.softmax(self.forward(data_point), dim=1)
                if self.verbosity > 3:
                    print('output', output)

                pred = output.data.max(1)[1]
                if self.verbosity > 3:
                    print('pred', pred)
                predictions[index] = pred
            
            
            
        return predictions
    
    
    
    def plot_tree(self, path='./data/plotting/temp.png'):

        tree_data = []
        for (node_filter, node_bias) in zip(self.inner_nodes[0].weight.detach().numpy(), self.inner_nodes[0].bias.detach().numpy()):
            node_string = 'f=' + str(np.round(node_filter, 3)) + '; b=' + str(np.round(node_bias, 3))
            tree_data.append(node_string)
            
           
        leaf_data = []
        for class_probibility in self.leaf_nodes.weight.detach().numpy().T:
            leaf_string = np.round(class_probibility, 3)#['c' + str(i) + ': ' + str(class_probibility[i]) + '\n' for i in range(class_probibility.shape[0])]
            leaf_data.append(leaf_string)
        #tree_data = list(zip(tree.inner_nodes[0].weight.detach().numpy(), tree.inner_nodes[0].bias.detach().numpy()))

        for layer in range(self.depth):
            if layer == 0:
                variable_name = str(layer) + '-' + str(0)
                locals().update({variable_name: Node(tree_data[sum([i**2 for i in range(1, layer)])])})
                root = Node(tree_data[0])
            else:
                for i in range(2**layer):
                    variable_name = str(layer) + '-' + str(i)
                    parent_name = str(layer-1) + '-' + str(i//2)

                    data_index = sum([2**i for i in range(layer)]) + i
                    data = tree_data[data_index]

                    locals().update({variable_name: Node(data, parent=locals()[parent_name])})
                    
        for leaf_index in range(2**(self.depth)):
            variable_name = str(self.depth) + '-' + str(leaf_index)
            parent_name = str(self.depth-1) + '-' + str(leaf_index//2)

            data = leaf_data[leaf_index]    
            locals().update({variable_name: Node(data, parent=locals()[parent_name])})
            
        DotExporter(locals()['0-0']).to_picture(path)

        return Image(path)
    
    def to_array(self, config=None):
        from utilities.utility_functions import largest_indices        
        
        if config is None or config['i_net']['function_representation_type'] == 1:
            filters = self.inner_nodes[0].weight.detach().numpy()
            biases = self.inner_nodes[0].bias.detach().numpy()

            leaf_probabilities = self.leaf_nodes.weight.detach().numpy().T
            
            return np.hstack([filters.flatten(), biases.flatten(), leaf_probabilities.flatten()])
        
        elif config['i_net']['function_representation_type'] == 2:
            filters = self.inner_nodes[0].weight.detach().numpy()
            coefficients_list = []
            topk_index_filter_list = []
            
            topk_softmax_output_filter_list = []
            
            #print('self.internal_node_num_', self.internal_node_num_)
            for i in range(self.internal_node_num_):
                #print('i', i)
                topk = largest_indices(np.abs(filters[i]), config['function_family']['decision_sparsity'])[0]
                topk_index_filter_list.append(topk)
                #print('topk', topk)
                for top_value_index in topk:
                    #print('top_value_index', top_value_index)
                    zeros = np.zeros_like(filters[i])
                    zeros[top_value_index] = 1#filters[i][top_value_index]
                    topk_softmax_output_filter_list.append(zeros)
                
                coefficients_list.append(filters[i][topk])
            
            coefficients = np.array(coefficients_list)
            topk_softmax_output_filter = np.array(topk_softmax_output_filter_list)
            
            biases = self.inner_nodes[0].bias.detach().numpy()

            leaf_probabilities = self.leaf_nodes.weight.detach().numpy().T
            
            return np.hstack([coefficients.flatten(), topk_softmax_output_filter.flatten(), biases.flatten(), leaf_probabilities.flatten()])

        return None
        
    def initialize_from_parameter_array(self, parameters, reshape=False, config=None):
        from utilities.utility_functions import get_shaped_parameters_for_decision_tree        
        if reshape == True:
            weights, biases, leaf_probabilities  = get_shaped_parameters_for_decision_tree(parameters, config, eager_execution=True)
            weights = weights.numpy()
            #biases = biases.numpy()
            leaf_probabilities = leaf_probabilities.numpy()
        else:    
            weights = parameters[:self.input_dim*self.internal_node_num_]
            weights = weights.reshape(self.internal_node_num_, self.input_dim)

            biases = parameters[self.input_dim*self.internal_node_num_:(self.input_dim+1)*self.internal_node_num_]

            leaf_probabilities = parameters[(self.input_dim+1)*self.internal_node_num_:]
            leaf_probabilities = leaf_probabilities.reshape(self.leaf_node_num_, self.output_dim).T

        
        self.inner_nodes[0].weight = torch.nn.Parameter(torch.FloatTensor(weights))
        self.inner_nodes[0].bias = torch.nn.Parameter(torch.FloatTensor(biases))
        self.leaf_nodes.weight = torch.nn.Parameter(torch.FloatTensor(leaf_probabilities))
        
        
        
        
    
    
    
    
class parameterDT():

    parameters = None
    shaped_parameters = None
    
    config = None
    normalizer_list = None

    def __init__(self, parameter_array, config, normalizer_list=None):
        self.parameters = parameter_array
        self.config = config
        self.normalizer_list = normalizer_list

        self.shaped_parameters = self.get_shaped_parameters(self.parameters)
        
    def predict(self, X_data):
        from utilities.metrics import calculate_function_value_from_vanilla_decision_tree_parameters_wrapper
        
        y_data_predicted, _  = calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(X_data, self.config)(self.parameters)
        return y_data_predicted.numpy()

    def plot(self, path='./data/plotting/temp.png'):
        from anytree import Node, RenderTree
        from anytree.exporter import DotExporter
        
        normalizer_list = self.normalizer_list

        splits, leaf_classes = self.shaped_parameters

        splits = splits.numpy()
        leaf_classes = leaf_classes.numpy()


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
        for i in range(self.config['function_family']['maximum_depth']+1):
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
            i = self.config['function_family']['maximum_depth']
            current_node_id = int(2**i - 1 + j)
            name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
            parent_node_id = int(np.floor((current_node_id-1)/2))
            parent_name = 'n' + str(parent_node_id)
            #split_variable = np.argmax(np.abs(split))
            #split_value = np.round(split[split_variable], 3)
            split_description = str(np.round((1-leaf_class), 3))#'x' + str(split_variable) + ' <= '  + str(split_value)
            nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)
            #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)        

            DotExporter(nodes['n0'], nodeattrfunc=lambda node: 'label="{}"'.format(node.display_name)).to_picture(path)


        return Image(path)#, nodes#nodes#tree        

    
    def get_shaped_parameters(self, flat_parameters, eager_execution=False):
        config = self.config

        input_dim = config['data']['number_of_variables']
        output_dim = config['data']['num_classes']
        internal_node_num_ = 2 ** config['function_family']['maximum_depth'] - 1 
        leaf_node_num_ = 2 ** config['function_family']['maximum_depth']

        if 'i_net' not in config.keys():
            config['i_net'] = {'function_representation_type': 1}

        if config['i_net']['function_representation_type'] == 1:

            splits_coeff = flat_parameters[:config['function_family']['decision_sparsity']*internal_node_num_]
            splits_coeff = tf.clip_by_value(splits_coeff, clip_value_min=config['data']['x_min'], clip_value_max=config['data']['x_max'])
            splits_coeff_list = tf.split(splits_coeff, internal_node_num_)
            splits_index = tf.cast(tf.clip_by_value(tf.round(flat_parameters[config['function_family']['decision_sparsity']*internal_node_num_:(config['function_family']['decision_sparsity']*internal_node_num_)*2]), clip_value_min=0, clip_value_max=config['data']['number_of_variables']-1), tf.int64)
            splits_index_list = tf.split(splits_index, internal_node_num_)

            splits_list = []
            for values_node, indices_node in zip(splits_coeff_list, splits_index_list):
                sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(indices_node, axis=1), values=values_node, dense_shape=[input_dim])
                dense_tensor = tf.sparse.to_dense(sparse_tensor)
                splits_list.append(dense_tensor)             

            splits = tf.stack(splits_list)            


            leaf_classes = flat_parameters[(config['function_family']['decision_sparsity']*internal_node_num_)*2:]  
            leaf_classes = tf.clip_by_value(leaf_classes, clip_value_min=0, clip_value_max=1)
            #tf.print(splits, leaf_classes)

            return splits, leaf_classes

        elif config['i_net']['function_representation_type'] == 2:

            split_values_num_params = internal_node_num_ * config['function_family']['decision_sparsity']
            split_index_num_params = config['data']['number_of_variables'] *  config['function_family']['decision_sparsity'] * internal_node_num_
            leaf_classes_num_params = leaf_node_num_ #* config['data']['num_classes']

            split_values = flat_parameters[:split_values_num_params]
            split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

            split_index_array = flat_parameters[split_values_num_params:split_values_num_params+split_index_num_params]    
            split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)
            split_index_list_by_internal_node_by_decision_sparsity = []
            for tensor in split_index_list_by_internal_node:
                split_tensor = tf.split(tensor, config['function_family']['decision_sparsity'])
                split_index_list_by_internal_node_by_decision_sparsity.append(split_tensor)
            split_index_list_by_internal_node_by_decision_sparsity_argmax = tf.split(tf.argmax(split_index_list_by_internal_node_by_decision_sparsity, axis=2), internal_node_num_)
            split_index_list_by_internal_node_by_decision_sparsity_argmax_new = []
            for tensor in split_index_list_by_internal_node_by_decision_sparsity_argmax:
                tensor_squeeze = tf.squeeze(tensor, axis=0)
                split_index_list_by_internal_node_by_decision_sparsity_argmax_new.append(tensor_squeeze)
            split_index_list_by_internal_node_by_decision_sparsity_argmax = split_index_list_by_internal_node_by_decision_sparsity_argmax_new    
            dense_tensor_list = []
            for indices_node, values_node in zip(split_index_list_by_internal_node_by_decision_sparsity_argmax,  split_values_list_by_internal_node):
                sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(indices_node, axis=1), values=values_node, dense_shape=[input_dim])
                dense_tensor = tf.sparse.to_dense(sparse_tensor)
                dense_tensor_list.append(dense_tensor) 
            splits = tf.stack(dense_tensor_list)

            leaf_classes_array = flat_parameters[split_values_num_params+split_index_num_params:]  
            split_index_list_by_leaf_node = tf.split(leaf_classes_array, leaf_node_num_)
            #leaf_classes_list = []
            #for tensor in split_index_list_by_leaf_node:
                #argmax = tf.argmax(tensor)
                #argsort = tf.argsort(tensor, direction='DESCENDING')
                #leaf_classes_list.append(argsort[0])
                #leaf_classes_list.append(argsort[1])

            leaf_classes = tf.squeeze(tf.stack(split_index_list_by_leaf_node))#tf.stack(leaf_classes_list)

            #tf.print(splits, leaf_classes)
            return splits, leaf_classes

        elif config['i_net']['function_representation_type'] >= 3:
            split_values_num_params = config['data']['number_of_variables'] * internal_node_num_#config['function_family']['decision_sparsity']
            split_index_num_params = config['data']['number_of_variables'] * internal_node_num_
            leaf_classes_num_params = leaf_node_num_ #* config['data']['num_classes']

            split_values = flat_parameters[:split_values_num_params]
            split_values_list_by_internal_node = tf.split(split_values, internal_node_num_)

            split_index_array = flat_parameters[split_values_num_params:split_values_num_params+split_index_num_params]    
            split_index_list_by_internal_node = tf.split(split_index_array, internal_node_num_)         

            split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(split_index_list_by_internal_node)

            splits = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))

            leaf_classes_array = flat_parameters[split_values_num_params+split_index_num_params:]  
            split_index_list_by_leaf_node = tf.split(leaf_classes_array, leaf_node_num_)

            leaf_classes = tf.squeeze(tf.stack(split_index_list_by_leaf_node))

            return splits, leaf_classes

        return None

    def calculate_function_value_from_vanilla_decision_tree_parameters_wrapper(self, random_evaluation_dataset, config):

        random_evaluation_dataset = tf.dtypes.cast(tf.convert_to_tensor(random_evaluation_dataset), tf.float32)       

        maximum_depth = config['function_family']['maximum_depth']
        leaf_node_num_ = 2 ** maximum_depth
        internal_node_num_ = 2 ** maximum_depth - 1

        #@tf.function(jit_compile=True)
        def calculate_function_value_from_vanilla_decision_tree_parameters(function_array):

            from utilities.utility_functions import get_shaped_parameters_for_decision_tree

            #tf.print('function_array', function_array)
            weights, leaf_probabilities = get_shaped_parameters_for_decision_tree(function_array, config)
            #tf.print('weights', weights)
            #tf.print('leaf_probabilities', leaf_probabilities)

            function_values_vanilla_dt = tf.vectorized_map(calculate_function_value_from_vanilla_decision_tree_parameter_single_sample_wrapper(weights, leaf_probabilities, leaf_node_num_, internal_node_num_, maximum_depth, config['data']['number_of_variables']), random_evaluation_dataset)
            #tf.print('function_values_vanilla_dt', function_values_vanilla_dt, summarize=-1)



            #penalty = tf.math.maximum(tf.cast((tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 0)) or tf.math.reduce_all(tf.equal(tf.round(leaf_probabilities), 1))), tf.float32) * 1.25, 1)

            return function_values_vanilla_dt, tf.constant(1.0, dtype=tf.float32)#penalty
        return calculate_function_value_from_vanilla_decision_tree_parameters



    def calculate_function_value_from_vanilla_decision_tree_parameter_single_sample_wrapper(self, weights, leaf_probabilities, leaf_node_num_, internal_node_num_, maximum_depth, number_of_variables):

        weights = tf.cast(weights, tf.float32)
        leaf_probabilities = tf.cast(leaf_probabilities, tf.float32)   

        #@tf.function(jit_compile=True)
        def calculate_function_value_from_vanilla_decision_tree_parameter_single_sample(evaluation_entry):

            evaluation_entry = tf.cast(evaluation_entry, tf.float32)

            weights_split = tf.split(weights, internal_node_num_)
            weights_split_new = [[] for _ in range(maximum_depth)]
            for i, tensor in enumerate(weights_split):
                current_depth = np.ceil(np.log2((i+1)+1)).astype(np.int32)

                weights_split_new[current_depth-1].append(tf.squeeze(tensor, axis=0))

            weights_split = weights_split_new

            #TDOD if multiclass, take index of min and max of leaf_proba to generate classes
            #leaf_probabilities_split = tf.split(leaf_probabilities, leaf_node_num_)
            #leaf_classes_list = []
            #for leaf_probability in leaf_probabilities_split:
            #    leaf_classes = tf.stack([tf.argmax(leaf_probability), tf.argmin(leaf_probability)])
            #    leaf_classes_list.append(leaf_classes)
            #leaf_classes = tf.keras.backend.flatten(tf.stack(leaf_classes_list))

            split_value_list = []

            for i in range(maximum_depth):
                #print('LOOP 1 ', i)
                current_depth = i+1#np.ceil(np.log2((i+1)+1)).astype(np.int32)
                num_nodes_current_layer = 2**current_depth - 1 - (2**(current_depth-1) - 1)
                #print('current_depth', current_depth, 'num_nodes_current_layer', num_nodes_current_layer)
                split_value_list_per_depth = []
                for j in range(num_nodes_current_layer):
                    #tf.print('weights_split[i][j]', weights_split[i][j])
                    #print('LOOP 2 ', j)
                    zero_identifier = tf.not_equal(weights_split[i][j], tf.zeros_like(weights_split[i][j]))
                    #tf.print('zero_identifier', zero_identifier)
                    split_complete = tf.greater(evaluation_entry, weights_split[i][j])
                    #tf.print('split_complete', split_complete, 'evaluation_entry', evaluation_entry, 'weights_split[i][j]', weights_split[i][j])
                    split_value = tf.reduce_any(tf.logical_and(zero_identifier, split_complete))
                    #tf.print('split_value', split_value)
                    split_value_filled = tf.fill( [2**(maximum_depth-current_depth)] , split_value)
                    split_value_neg_filled = tf.fill( [2**(maximum_depth-current_depth)], tf.logical_not(split_value))
                    #tf.print('tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled]))', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
                    #print('LOOP 2 OUTPUT', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
                    split_value_list_per_depth.append(tf.keras.backend.flatten(tf.stack([split_value_neg_filled, split_value_filled])))        
                    #tf.print('tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled]))', tf.keras.backend.flatten(tf.stack([split_value_filled, split_value_neg_filled])))
                #print('LOOP 1 OUTPUT', tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)))
                split_value_list.append(tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)))
                #tf.print('DT SPLITS ENCODED', tf.keras.backend.flatten(tf.stack(split_value_list_per_depth)), summarize=-1)
                    #node_index_in_layer += 1        
            #tf.print(split_value_list)
            #tf.print(tf.stack(split_value_list))
            #tf.print('split_value_list', split_value_list, summarize=-1)
            #tf.print('tf.stack(split_value_list)\n', tf.stack(split_value_list), summarize=-1)
            split_values = tf.cast(tf.reduce_all(tf.stack(split_value_list), axis=0), tf.float32)    
            #tf.print('split_values', split_values, summarize=-1)
            leaf_classes = tf.cast(leaf_probabilities, tf.float32)
            #tf.print('leaf_classes', leaf_classes, summarize=-1)
            final_class_probability = 1-tf.reduce_max(tf.multiply(leaf_classes, split_values))                                                                                                                                            
            return final_class_probability#y_pred

        return calculate_function_value_from_vanilla_decision_tree_parameter_single_sample

   