import numpy as np
from tqdm import tqdm_notebook as tqdm

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
            verbosity=1): #0=no verbosity, 1= epoch lvl verbosity, 2=batch lvl verbosity, 3=additional prints
        super(SDT, self).__init__()
        
        torch.manual_seed(random_seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.maximum_path_probability = maximum_path_probability

        self.depth = depth
        self.beta = beta
        self.decision_sparsity = decision_sparsity
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
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
        
        
        if self.decision_sparsity != -1:
            vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)

            weights = torch.zeros_like(self.inner_nodes[0].weight)
            for i, idx in enumerate(idx_list):
                weights[i][idx] = self.inner_nodes[0].weight[i][idx]
            self.inner_nodes[0].weight = torch.nn.Parameter(weights)    
            
                    
        self.criterion = criterion
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
        self.optimizer = torch.optim.Adam(self.parameters(),
                             lr=lr,
                             weight_decay=weight_decaly)        

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        
        #maximum_path_probability
        
        if self.verbosity >= 3:
            print('_mu, _penalty', _mu, _penalty)
            
        if self.maximum_path_probability:
            cond = torch.eq(_mu, torch.max(_mu, axis=1).values.reshape(-1,1))
            _mu = torch.where(cond, _mu, torch.zeros_like(_mu))
            
        y_pred = self.leaf_nodes(_mu)
        
        if self.verbosity>= 3:
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
        if self.verbosity>= 3:
            print('X', X)
        path_prob = nn.Sigmoid()(self.beta*self.inner_nodes(X))
        if self.verbosity>= 3:
            print('path_prob', path_prob)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        if self.verbosity>= 3:
            print('path_prob', path_prob)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        if self.verbosity>= 3:
            print('path_prob', path_prob)
        
            
        
        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)
        if self.verbosity>= 3:
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

            if self.verbosity>= 3:
                #print('_penalty loop', _penalty)    
                print('_mu updated loop', _mu) 
                
            _mu = _mu * _path_prob  # update path probabilities

            if self.verbosity>= 3:
                #print('_penalty loop', _penalty)    
                print('_mu updated loop', _mu)      
                
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        if self.verbosity>= 3:
            print('_mu updated', _mu)
        mu = _mu.view(batch_size, self.leaf_node_num_)
        if self.verbosity >= 3:
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

            
    def fit(self, X, y, batch_size=32, epochs=100):
        self.train()
        X, y = torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device)
        
        for epoch in tqdm(range(epochs)):
        
            correct_counter = 0
            loss_list = []
            for index, (data, target) in enumerate(zip(batch(X, batch_size), batch(y, batch_size))):

                #print(data)

                data = torch.stack(data).to(self.device)
                target =  torch.stack(target).to(self.device)    

                output, penalty = self.forward(data, is_training_data=True)
                loss = self.criterion(output, target.view(-1))

                loss += penalty

                #print(self.inner_nodes[0].weight)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()     
                

                
                if self.decision_sparsity != -1:
                    vals_list, idx_list = torch.topk(torch.abs(self.inner_nodes[0].weight), k=self.decision_sparsity, dim=1)#output.topk(k)

                    weights = torch.zeros_like(self.inner_nodes[0].weight)
                    for i, idx in enumerate(idx_list):
                        weights[i][idx] = self.inner_nodes[0].weight[i][idx]
                    self.inner_nodes[0].weight = torch.nn.Parameter(weights)                          
                
                #print(self.inner_nodes[0].weight)
                
                if self.verbosity >= 1:
                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    batch_idx = (index+1)*batch_size
                    
                    loss_list.append(float(loss))
                    correct_counter += correct
                    
                    if self.verbosity >= 2:
                        msg = (
                            "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                            " Correct: {:03d}/{:03d}"
                        )
                        print(msg.format(epoch, batch_idx, loss, int(correct), int(data.shape[0])))

            if self.verbosity >= 1:
                pred = output.data.max(1)[1]
                correct = pred.eq(target.view(-1).data).sum()
                #batch_idx = (index+1)*batch_size
                msg = (
                    "Epoch: {:02d} | Loss: {:.5f} |"
                    " Correct: {:03d}/{:03d}"
                )
                print(msg.format(epoch, np.mean(loss_list), int(correct_counter), int(X.shape[0])))   
                
    def evaluate(self, X, y):
        self.eval()
        
        correct = 0.

        data, target = torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device)#data.to(device), target.to(device)
        
        output = F.softmax(self.forward(data), dim=1)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1).data).sum()

        accuracy = float(correct) / target.shape[0]
        
        if self.verbosity >= 1:
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
            if self.verbosity>= 3:
                print('output', output)

            pred = output[:,1:].data#output.data.max(1)[1]
            if self.verbosity>= 3:
                print('pred', pred)        

            predictions = pred.numpy()



            return predictions       
        
        return None
            
            
    
    def predict(self, X):
        self.eval()
        
        data = torch.FloatTensor(X).to(self.device)
        output = F.softmax(self.forward(data), dim=1)
        if self.verbosity>= 3:
            print('output', output)
        
        pred = output.data.max(1)[1]
        if self.verbosity>= 3:
            print('pred', pred)        
        
        predictions = pred
        
        if False:
            predictions = np.full(X.shape[0], np.nan)
            for index, data_point in enumerate(X):

                data_point = torch.FloatTensor([data_point]).to(self.device)

                output = F.softmax(self.forward(data_point), dim=1)
                if self.verbosity>= 3:
                    print('output', output)

                pred = output.data.max(1)[1]
                if self.verbosity>= 3:
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
            
            print('self.internal_node_num_', self.internal_node_num_)
            for i in range(self.internal_node_num_):
                print('i', i)
                topk = largest_indices(np.abs(filters[i]), config['function_family']['decision_sparsity'])[0]
                topk_index_filter_list.append(topk)
                print('topk', topk)
                for top_value_index in topk:
                    print('top_value_index', top_value_index)
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
        
    def initialize_from_parameter_array(self, parameters):
        
        weights = parameters[:self.input_dim*self.internal_node_num_]
        weights = weights.reshape(self.internal_node_num_, self.input_dim)
        
        biases = parameters[self.input_dim*self.internal_node_num_:(self.input_dim+1)*self.internal_node_num_]
        
        leaf_probabilities = parameters[(self.input_dim+1)*self.internal_node_num_:]
        leaf_probabilities = leaf_probabilities.reshape(self.leaf_node_num_, self.output_dim).T

        
        self.inner_nodes[0].weight = torch.nn.Parameter(torch.FloatTensor(weights))
        self.inner_nodes[0].bias = torch.nn.Parameter(torch.FloatTensor(biases))
        self.leaf_nodes.weight = torch.nn.Parameter(torch.FloatTensor(leaf_probabilities))
        
        
        
    