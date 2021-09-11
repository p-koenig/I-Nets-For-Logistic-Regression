

# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
from scipy.special import digamma, gamma
import itertools
import copy

from mpmath import *
from sympy import *
#from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from pysymbolic_adjusted.models.special_functions import *

from tqdm import tqdm, trange, tqdm_notebook, tnrange

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sympy import Integral, Symbol
from sympy.abc import x, y

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from IPython import get_ipython

import tensorflow as tf
import sympy as sym


def is_ipython():
    
    try:
        
        __IPYTHON__
        
        return True
    
    except NameError:
        
        return False


def basis(a, b, c, x, hyper_order=[1, 2, 2, 2], approximation_order=3):
        
    epsilon = 0.001
    
    
    #print('a, b, c', a, b, c)
    
    func_   = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=approximation_order)
    
    #print('END MeijerG')
    
    return func_.evaluate(x + epsilon)

def basis_expression(a, b, c, hyper_order=[1, 2, 2, 2], approximation_order=3):
        
    func_ = MeijerG(theta=[a, a, a, b, c], order=hyper_order, approximation_order=approximation_order)
    
    return func_
    

def basis_grad(a, b, c, x, hyper_order=[1, 2, 2, 2], verbosity=False):
    secure_positive_log = False
    
    if c <= 0 and not secure_positive_log: #if c <= 0, log or div cant be calculated ans thus return nan 
        grad_a = np.empty(x.shape)
        grad_a[:] = np.nan
        grad_b = np.empty(x.shape)
        grad_b[:] = np.nan
        grad_c = np.empty(x.shape)
        grad_c[:] = np.nan       
        
        print('Wrong c Value: ' + str(c))
        
        return grad_a, grad_b, grad_c
    elif c == 0:
        c = c + 1e-4
    
    #print('abc', a, b, c)
    
    K1     = sc.special.digamma(a - b + 1)
    K2     = sc.special.digamma(a - b + 2)
    K3     = sc.special.digamma(a - b + 3)
    K4     = sc.special.digamma(a - b + 4)
    
    #print('K', K1, K2, K3, K4)
    
    G1     = sc.special.gamma(a - b + 1)
    G2     = sc.special.gamma(a - b + 2)
    G3     = sc.special.gamma(a - b + 3)
    G4     = sc.special.gamma(a - b + 4)
    
    #print('G', G1, G2, G3, G4)
    if secure_positive_log:
        nema1  = 6 * ((c * x)**3) * (K4 - np.log(np.abs(c * x))) #np.abs() oder np.sqrt()
        nema2  = 2 * ((c * x)**2) * (-K3 + np.log(np.abs(c * x)))
        nema3  = (c * x) * (K2 - np.log(np.abs(c * x)))
        nema4  = -1 * (K1 - np.log(np.abs(c * x)))
    else:
        nema1  = 6 * ((c * x)**3) * (K4 - np.log(c * x))
        nema2  = 2 * ((c * x)**2) * (-K3 + np.log(c * x))
        nema3  = (c * x) * (K2 - np.log(c * x))
        nema4  = -1 * (K1 - np.log(c * x)) 
    #print('nema', nema1[:5], nema2[:5], nema3[:5], nema4[:5])
    
    nemb1  = -1 * 6 * ((c * x)**3) * K4 
    nemb2  = 2 * ((c * x)**2) * K3 
    nemb3  = -1 * (c * x) * K2 
    nemb4  = K1 

    #print('nemb', nemb1[:5], nemb2[:5], nemb3[:5], nemb4)
    
    nemc1  = -1 * (c**2) * (x**3) * (6 * a + 18)
    nemc2  = (c * (x**2)) * (4 + 2 * a) 
    nemc3  = -1 * x * (1 + a)
    nemc4  = a / c
    
    #print('nemc', nemc1[:5], nemc2[:5], nemc3[:5], nemc4)
    
    grad_a = ((c * x) ** a) * (nema1/G4 + nema2/G3 + nema3/G2 + nema4/G1) 
    grad_b = ((c * x) ** a) * (nemb1/G4 + nemb2/G3 + nemb3/G2 + nemb4/G1) 
    grad_c = ((c * x) ** a) * (nemc1/G4 + nemc2/G3 + nemc3/G2 + nemc4/G1) 

    #print('grad', grad_a[:5], grad_b[:5], grad_c[:5])
    
    return grad_a, grad_b, grad_c



def tune_single_dim(lr, n_iter, x, y, tqdm_mode, mode = "classification", verbosity=False, approximation_order=3, max_param_value=100, early_stopping=None, restarts=0):
    
    
    epsilon   = 0.001
    x         = x + epsilon
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    a         = tf.Variable(2.0)
    b         = tf.Variable(1.0)
    c         = tf.Variable(1.0)
    
    batch_size  = np.min((x.shape[0], 500)) 
      
    global_minimum = np.inf
    
    best_parameters = {'a': np.nan,
                      'b': np.nan,
                      'c': np.nan}
        
    for restart_number in tqdm(range(restarts+1), desc='restart loop'):
        if restart_number > 0:
            a         = tf.Variable(np.random.uniform(1, 2))
            b         = tf.Variable(np.random.uniform(1, 2))
            c         = tf.Variable(np.random.uniform(1, 2))
        
        min_loss = np.inf
        early_stopping_counter = 0
        
        
        
        for u in tqdm_mode(range(n_iter), desc='iter loop'):

            batch_index = np.random.choice(list(range(x.shape[0])), size=batch_size)

            #print('function calls')

            new_grads   = basis_grad(tf.identity(a).numpy(), tf.identity(b).numpy(), tf.identity(c).numpy(), x[batch_index])
            func_true   = basis(tf.identity(a).numpy(), tf.identity(b).numpy(), tf.identity(c).numpy(), x[batch_index], approximation_order=approximation_order)
    
            loss        =  np.mean((func_true - y[batch_index])**2)

            if verbosity:

                print("Iteration: %d \t--- Loss: %.3f" % (u, loss))

            #print('grads', new_grads)
            #print('func', func_true)           
            
            grads_a   = np.mean(2 * new_grads[0] * (func_true - y[batch_index]))
            grads_b   = np.mean(2 * new_grads[1] * (func_true - y[batch_index]))
            grads_c   = np.mean(2 * new_grads[2] * (func_true - y[batch_index]))

            optimizer.apply_gradients(zip([grads_a, grads_b, grads_c], [a,b,c]))
            
            if tf.reduce_any(np.isnan([tf.identity(a).numpy(), tf.identity(b).numpy(), tf.identity(c).numpy()])) or tf.reduce_any(np.isinf([tf.identity(a).numpy(), tf.identity(b).numpy(), tf.identity(c).numpy()])) or tf.reduce_any((np.abs(np.array([tf.identity(a).numpy(), tf.identity(b).numpy(), tf.identity(c).numpy()])) > max_param_value)): #or c_new <= 0
                if verbosity:
                    print('BREAK tune_single_dim')
                    print('func_true', func_true[0])
                    print('y[batch_index]', y[batch_index][0])
                    print('a_new, b_new, c_new', grads_a, grads_b, grads_c)
                    print('best_parameters abc', best_parameters['a'], best_parameters['b'], best_parameters['c'])
                break
                
            #print('a,b,c', a,b,c)
            #a_new = a - lr * grads_a
            #b_new = b - lr * grads_b
            #c_new = c - lr * grads_c           


            #a = a_new
            #b = b_new
            #c = c_new          

            if early_stopping is not None:
                if loss < min_loss:
                    min_loss = loss
                    early_stopping_counter = 0
                    
                    best_parameters['a'] = tf.identity(a).numpy()
                    best_parameters['b'] = tf.identity(b).numpy()
                    best_parameters['c'] = tf.identity(c).numpy()

                else:
                    if early_stopping_counter >= early_stopping:
                        if verbosity:
                            print('Early Stopping requirement reached after ' + str(u) + ' Iterations')                          
                        break
                    else:
                        early_stopping_counter += 1              

            #grads_a   = np.nan_to_num(np.mean(2 * new_grads[0] * (func_true - y[batch_index])).astype(np.float32))
            #grads_b   = np.nan_to_num(np.mean(2 * new_grads[1] * (func_true - y[batch_index])).astype(np.float32))
            #grads_c   = np.nan_to_num(np.mean(2 * new_grads[2] * (func_true - y[batch_index])).astype(np.float32))


            #(grads_a, grads_b, grads_c) = Normalizer().fit_transform([np.nan_to_num([grads_a, grads_b, grads_c])])[0]

            
        if min_loss < global_minimum:
            print('New Global Minimum: ' + str(min_loss))
            global_minimum = min_loss
            a_final = best_parameters['a']
            b_final = best_parameters['b']
            c_final = best_parameters['c']
            
    if verbosity:
        print('return abc', a_final, b_final, c_final)
    return a_final, b_final, c_final 


def compose_features(params, X, approximation_order=3):
    
    #print(params)
    
    X_out = [basis(a=float(params[k, 0]), b=float(params[k, 1]), c=float(params[k, 2]), 
                   x=X[:, k], hyper_order=[1, 2, 2, 2], approximation_order=approximation_order) for k in range(X.shape[1])] 
    
    return np.array(X_out).T
    

class symbolic_metamodel:
    
    def __init__(self, model, X, mode="classification", approximation_order=3, force_polynomial=False, verbosity=False, early_stopping=None, restarts=0):
        
        self.verbosity = verbosity
        
        self.feature_expander = PolynomialFeatures(2, include_bias=False, interaction_only=True)
        self.X                = X
        self.X_new            = self.feature_expander.fit_transform(X) 
        self.X_names          = self.feature_expander.get_feature_names()
        
        self.mode = mode
        #print('self.X.shape', self.X.shape)
        #print('self.X_new.shape', self.X_new.shape)
        
        self.max_param_value = 100000
        self.early_stopping = early_stopping
        self.restarts = restarts
        
        self.approximation_order = approximation_order
        self.force_polynomial = force_polynomial
                
            
        if self.mode == "classification": 

            self.Y                = model.predict_proba(self.X)[:, 1]
            self.Y_r              = np.log(self.Y/(1 - self.Y))        
        else:
            
            self.Y_r              = model.predict(self.X)
        
        
        self.num_basis        = self.X_new.shape[1]
        
        #print(self.num_basis)
        
        self.params_per_basis = 3
        self.total_params     = self.num_basis * self.params_per_basis + 1
        
        a_init                = 1.393628702223735 
        b_init                = 1.020550117939659
        c_init                = 1.491820813243337
        
        self.params           = np.tile(np.array([a_init, b_init, c_init]), [self.num_basis, 1])
        
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            
            self.tqdm_mode = tqdm_notebook
            
        else:
            
            self.tqdm_mode = tqdm
            
    
    def set_equation(self, reset_init_model=False):
         
        #print(self.params, self.X_new[:10], self.approximation_order)
            
        self.X_init           = compose_features(self.params, self.X_new, approximation_order=self.approximation_order)
        
        #print('self.X_init', self.X_init[:10])
        
        if reset_init_model:
            
            self.init_model   = Ridge(alpha=.1, fit_intercept=False, normalize=True) #LinearRegression
            
            self.init_model.fit(self.X_init, self.Y_r)
    
    def get_gradients(self, Y_true, Y_metamodel, batch_index=None):
        
        #print('FUNC: get_gradients', Y_true, Y_metamodel, batch_index)
        
        param_grads = self.params * 0
        epsilon     = 0.001 
        
        #print('self.params.shape[0]', self.params.shape[0])
        
        for k in range(self.params.shape[0]):
            
            a                 = float(self.params[k, 0])
            b                 = float(self.params[k, 1])
            c                 = float(self.params[k, 2])
            
            #print('abc', a, b, c)
            #print('self.X_new[:, k]', self.X_new[:, k])
            
            if batch_index is None:
                grads_vals    = basis_grad(a, b, c, self.X_new[:, k] +  epsilon)
            else:
                grads_vals    = basis_grad(a, b, c, self.X_new[batch_index, k] +  epsilon)
            
            
            
            param_grads[k, :] = np.array(self.loss_grads(Y_true, Y_metamodel, grads_vals))
            
            #print('param_grads[k, :]', param_grads[k, :])
        
        return param_grads
        
    
    def loss(self, Y_true, Y_metamodel):

        loss = np.mean((Y_true - Y_metamodel)**2)
        
        return loss
    
    def loss_grads(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_a = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[0])
        loss_grad_b = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[1])
        loss_grad_c = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x[2])
        
        return loss_grad_a, loss_grad_b, loss_grad_c 
    
    def loss_grad_coeff(self, Y_true, Y_metamodel, param_grads_x):
        
        loss_grad_ = np.mean(2 * (Y_true - Y_metamodel) * param_grads_x)
        
        return loss_grad_
        
    
    def fit(self, num_iter=10, batch_size=100, learning_rate=.01):
        
        print("---- Tuning the basis functions ----")
        for u in self.tqdm_mode(range(self.X.shape[1]), desc='basis function loop'):
    
            self.params[u, :] = tune_single_dim(lr=0.1, 
                                                n_iter=500, 
                                                x=self.X_new[:, u], 
                                                y=self.Y_r, 
                                                mode=self.mode,
                                                tqdm_mode=self.tqdm_mode, 
                                                verbosity=self.verbosity, 
                                                approximation_order=self.approximation_order, 
                                                max_param_value=self.max_param_value, 
                                                early_stopping=self.early_stopping, 
                                                restarts=self.restarts)
        self.set_equation(reset_init_model=True)
        self.exact_expression, self.approx_expression = self.symbolic_expression()         
        self.metamodel_loss = []
        
        print("----  Optimizing the metamodel  ----")
        
        #print(num_iter) 
        
        #print('self.params', self.params)
        min_loss = np.inf
        early_stopping_counter = 0       
        
        best_params = None
        best_coefs = None
        
        optimizer_coef_ = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer_params = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        params_variable       = [[tf.Variable(item) for item in array] for array in self.params] #tf.Variable(self.params)
        coef_variable         = [tf.Variable(item) for item in self.init_model.coef_]#tf.Variable(self.init_model.coef_)
        
      
        for i in self.tqdm_mode(range(num_iter)):
            batch_index = np.random.choice(list(range(self.X_new.shape[0])), size=batch_size)
            
            #print('batch_index', batch_index[:10])
                        
            if np.isnan(self.X_init[batch_index, :]).any() or np.isinf(self.X_init[batch_index, :]).any():
                print('\n\nBREAK X_init', i)
                #self.params  = best_params
                #self.init_model.coef_ = best_coefs                    
                #self.set_equation()  
                #self.exact_expression, self.approx_expression = self.symbolic_expression()           
                
                if self.verbosity:
                    print('\n\nBREAK X_init')
                    print('self.X_init[batch_index, :]', self.X_init[batch_index, :])

                    print('self.params', self.params)
                    print('self.init_model.coef_', self.init_model.coef_)

                    #print('self.exact_expression', self.exact_expression)
                    #print('self.approx_expression', self.approx_expression)
            
                
                break                
            
            curr_func   = self.init_model.predict(np.nan_to_num(self.X_init[batch_index, :]))
            #print('curr_func INIT MODEL', np.round(curr_func, 3)[:5])
                            
            
            curr_func   = calculate_function_values_from_sympy(self.exact_expression, np.nan_to_num(self.X_init[batch_index, :]))
            #print('curr_func calculate_function_values_from_sympy MODEL', np.round(curr_func, 3)[:5])
            
                                    
            #print('self.loss(self.Y_r[batch_index], curr_func)', self.loss(self.Y_r[batch_index], curr_func))
            
            #print('self.Y_r[batch_index]', self.Y_r[batch_index])
            
            if np.isnan(curr_func).any() or np.isinf(curr_func).any():
                print('\n\nBREAK curr_func', i)
                #self.params  = best_params
                #self.init_model.coef_ = best_coefs                   
                #self.set_equation()  
                #self.exact_expression, self.approx_expression = self.symbolic_expression()            
                    
                if self.verbosity:
                    print('\n\nBREAK curr_func')
                    print('curr_func', curr_func)

                    print('self.params', self.params)
                    print('self.init_model.coef_', self.init_model.coef_)

                    #print('self.exact_expression', self.exact_expression)
                    #print('self.approx_expression', self.approx_expression)
                
                
                break
            
            #param_grads  = np.nan_to_num(self.get_gradients(self.Y_r[batch_index], curr_func, batch_index).astype(np.float32))
            param_grads  = self.get_gradients(self.Y_r[batch_index], curr_func, batch_index)
            
            flatten_list = lambda t: [item for sublist in t for item in sublist]      
            
            optimizer_params.apply_gradients(flatten_list(list([list(zip(grads, variables)) for grads, variables in zip(param_grads, params_variable)])))
            #params = self.params - learning_rate * param_grads #np.nan_to_num(self.params - learning_rate * param_grads, nan=0.001)            
            
            
            
            #print('param_grads', param_grads[:10])
            #print('self.params', self.params[:10])
            
            #coef_grads            = np.nan_to_num(np.array([self.loss_grad_coeff(self.Y_r[batch_index], curr_func, self.X_init[batch_index, k]) for k in range(self.X_init.shape[1])]).astype(np.float32)) 
            
            coef_grads            = [self.loss_grad_coeff(self.Y_r[batch_index], curr_func, self.X_init[batch_index, k]) for k in range(self.X_init.shape[1])]
            optimizer_coef_.apply_gradients(zip(coef_grads, coef_variable))
            #coef_ = self.init_model.coef_ - learning_rate * np.array(coef_grads)

             
            #print('coef_grads', coef_grads[:10])
            #print('self.init_model.coef_', self.init_model.coef_[:10])
            
                        
            #print('self.init_model.coef_', self.init_model.coef_[:10])  
            
            #if np.isnan(params).any() or (np.abs(np.array(params)) > self.max_param_value).any() or np.isnan(coef_).any() or np.abs((np.array(coef_) > self.max_param_value)).any(): #or (params[:, 2] <= 0).any()
            if tf.reduce_any(np.isnan(tf.identity(params_variable))) or tf.reduce_any((tf.abs(tf.identity(params_variable)) > self.max_param_value)) or tf.reduce_any(np.isnan(tf.identity(coef_variable))) or tf.reduce_any((np.abs(tf.identity(coef_variable)) > self.max_param_value)): #or (params[:, 2] <= 0).any()
                print('\n\nBREAK Params or Coef', i)
                print('curr_func', curr_func)

                print('param_grads', param_grads)
                print('params', params_variable)
                print('self.params', self.params)
                print('coef_grads', coef_grads)
                print('coef_', coef_variable)
                
                
                #self.params  = best_params
                #self.init_model.coef_ = best_coefs                      
                #self.set_equation()  
                #self.exact_expression, self.approx_expression = self.symbolic_expression()
                    
                if self.verbosity:
                    print('\n\nBREAK Params or Coef')
                    print('curr_func', curr_func)

                    print('param_grads', param_grads)
                    print('params', params_variable)
                    print('self.params', self.params)
                    print('coef_grads', coef_grads)
                    print('coef_', coef_variable)
                    print('self.init_model.coef_', self.init_model.coef_)

                    #print('self.exact_expression', self.exact_expression)
                    #print('self.approx_expression', self.approx_expression)
            
            
                
                break
            else:
                metamodel_loss = self.loss(self.Y_r[batch_index], curr_func)
                
                print("Iteration: %d \t--- Loss: %.3f" % (i, metamodel_loss))
                
                if self.early_stopping is not None:
                    if metamodel_loss < min_loss:
                        min_loss = metamodel_loss
                        early_stopping_counter = 0
                        
                        best_params = tf.identity(params_variable).numpy()
                        best_coefs = tf.identity(coef_variable).numpy()                      
                        #best_params = params
                        #best_coefs = coef_                           
                        
                    else:
                        if early_stopping_counter >= self.early_stopping:
                            if self.verbosity:
                                print('Early Stopping requirement reached after ' + str(i) + ' Iterations')

                            #self.params  = best_params
                            #self.init_model.coef_ = best_coefs                     
                            #self.set_equation()  
                            #self.exact_expression, self.approx_expression = self.symbolic_expression()
                            
                            break
                        else:
                            early_stopping_counter += 1
                    
                self.metamodel_loss.append(metamodel_loss)
                self.params  = tf.identity(params_variable).numpy()
                self.init_model.coef_ = tf.identity(coef_variable).numpy()                
                #self.params  = params
                #self.init_model.coef_ = coef_
                
                self.set_equation()  
                self.exact_expression, self.approx_expression = self.symbolic_expression()            
                #if self.verbosity:
                    #print('self.exact_expression', self.exact_expression)
                    #print('self.approx_expression', self.approx_expression)
        
        self.params  = best_params
        self.init_model.coef_ = best_coefs                     
        self.set_equation()  
        self.exact_expression, self.approx_expression = self.symbolic_expression()        
        
    def evaluate(self, X):
        
        X_modified  = self.feature_expander.fit_transform(X)
        X_modified_ = compose_features(self.params, X_modified, approximation_order=self.approximation_order)
        X_modified_ = np.nan_to_num(X_modified_)
        Y_pred_r    = self.init_model.predict(X_modified_)

        
        if self.force_polynomial:
            return Y_pred_r

        Y_pred      = 1 / (1 + np.exp(-1 * Y_pred_r))
        
        return Y_pred 
    
    def symbolic_expression(self):
    
        dims_ = []

        for u in range(self.num_basis):

            new_symb = self.X_names[u].split(" ")

            if len(new_symb) > 1:
    
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
                S2 = Symbol(new_symb[1].replace("x", "X"), real=True)
        
                dims_.append(S1 * S2)
    
            else:
        
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
    
                dims_.append(S1)
        
        self.dim_symbols = dims_
        
        sym_exact   = 0
        sym_approx  = 0
        x           = symbols('x')

        #print(self.num_basis)
        #print(self.init_model.coef_)
        #print('self.init_model.coef_.shape', self.init_model.coef_.shape)
        
        #if self.init_model.coef_.shape == (1,1):
        #    self.init_model.coef_ = self.init_model.coef_.reshape(1,)
        if len(self.init_model.coef_.shape) >= 2 and self.init_model.coef_.shape[0] == 1:
            self.init_model.coef_ = self.init_model.coef_.reshape(-1,)                
                
        #print('self.init_model.coef_.shape', self.init_model.coef_.shape)
                
        for v in range(self.num_basis):
    
            f_curr      = basis_expression(a=float(self.params[v,0]), 
                                           b=float(self.params[v,1]), 
                                           c=float(self.params[v,2]), 
                                           approximation_order=self.approximation_order)
        
            #print(v)
            #print('self.init_model.coef_', self.init_model.coef_)
            #print(self.init_model.coef_[v])
        
            #print(sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))))
            #print(sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))))
            sym_exact  += sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))).subs(x, dims_[v])
            
            sym_approx += sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))).subs(x, dims_[v])    
        
        if self.force_polynomial:
            return sym_exact, sym_approx
            
        return 1/(1 + exp(-1*sym_exact)), 1/(1 + exp(-1*sym_approx))   
    
    
    def get_gradient_expression(self):
        
        diff_dims  = self.dim_symbols[:self.X.shape[1]]
        gradients_ = [diff(self.approx_expression, diff_dims[k]) for k in range(len(diff_dims))]

        diff_dims  = [str(diff_dims[k]) for k in range(len(diff_dims))]
        evaluator  = [lambdify(diff_dims, gradients_[k], modules=['math']) for k in range(len(gradients_))]
    
        return gradients_, diff_dims, evaluator
    

    def _gradient(self, gradient_expressions, diff_dims, evaluator, x_in):
    
        Dict_syms  = dict.fromkeys(diff_dims)

        for u in range(len(diff_dims)):

            Dict_syms[diff_dims[u]] = x_in[u]
         
        grad_out  = [np.abs(evaluator[k](**Dict_syms)) for k in range(len(evaluator))]
    
        
        return np.array(grad_out)    
    
    
    def get_instancewise_scores(self, X_in):
    
        gr_exp, diff_dims, evaluator = self.get_gradient_expression()
    
        gards_ = [self._gradient(gr_exp, diff_dims, evaluator, X_in[k, :]) for k in range(X_in.shape[0])]
    
        return gards_
    
        

        
def calculate_function_values_from_sympy(function, data_points, variable_names=None):
    function_vars = None
    
    if variable_names is None:
        variable_names = ['X' + str(i) for i in range(data_points.shape[1])]
    
    if function is None:
        return np.array([np.nan for i in range(data_points.shape[0])])
    try:
        if variable_names is None:
            function_vars = function.atoms(Symbol)
            print(function_vars)
        else:
            function_vars = [sym.symbols(variable_name, real=True) for variable_name in variable_names]
        #print('function_vars', function_vars)
        lambda_function = lambdify([function_vars], function, modules=["scipy", "numpy"])
        #print('lambda_function', lambda_function)
        #print('data_points[0]', data_points[0])
        if len(function_vars) >= 1:
            function_values = [lambda_function(data_point) for data_point in data_points]
            
        else:
            function_values = [lambda_function() for i in range(data_points.shape[0])]
    except (NameError, KeyError) as e:
        #print(e)
        function_values = []
        for data_point in data_points:
            function_value = function.evalf(subs={var: data_point[index] for index, var in enumerate(list(function_vars))})
            try:
                function_value = float(function_value)
            except TypeError as te:
                #print('te', te)
                #print('function_value', function_value)
                #print('function', function)
                #print('function_vars', function_vars, type(function_vars))
                function_value = np.inf
            function_values.append(function_value)
    function_values = np.nan_to_num(function_values).ravel()
                
    return function_values