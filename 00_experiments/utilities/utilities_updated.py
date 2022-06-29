import numpy as np
import sklearn
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
import timeit

from xgboost import XGBClassifier
from genetic_tree import GeneticTree



def flatten_list(l):
    
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
                
    flat_l = flatten(l)
    
    return list(flat_l)


def mergeDict(dict1, dict2):
    #Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                newDict[key] = mergeDict(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
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


def normalize_data(X_data, low=-1, high=1):
    normalizer_list = []
    if isinstance(X_data, pd.DataFrame):
        for column_name in X_data:
            scaler = MinMaxScaler(feature_range=(low, high))
            scaler.fit(X_data[column_name].values.reshape(-1, 1))
            X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)
    else:
        for i, column in enumerate(X_data.T):
            scaler = MinMaxScaler(feature_range=(low, high))
            scaler.fit(column.reshape(-1, 1))
            X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)
        
    return X_data, normalizer_list

def split_train_test_valid(X_data, y_data, valid_frac=0.10, test_frac=0.20, seed=42, verbosity=0):
    data_size = X_data.shape[0]
    test_size = int(data_size*test_frac)
    valid_size = int(data_size*valid_frac)
    
    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_with_valid, y_train_with_valid, test_size=valid_size, random_state=seed)

    if verbosity > 0:
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def rebalance_data(X_train, 
                   y_train, 
                   balance_ratio=0.25, 
                   strategy='SMOTE', 
                   seed=42, 
                   verbosity=0):#, strategy='SMOTE'
    true_labels = len(y_train[y_train >= 0.5 ]) 
    false_labels = len(y_train[y_train < 0.5 ]) 

    true_ratio = true_labels/(true_labels+false_labels)
    false_ratio = false_labels/(false_labels+true_labels)
    
    min_ratio = min(true_ratio, false_ratio)
    if verbosity > 0:
        print('True Ratio: ', str(true_labels/(true_labels+false_labels)))    
    if min_ratio <= balance_ratio:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
        from imblearn.combine import SMOTETomek, SMOTEENN
        if strategy == 'SMOTE':
            oversample = SMOTE()
        elif strategy == 'SMOTEN':
            oversample = SMOTEN()                 
        elif strategy == 'BorderlineSMOTE':
            oversample = BorderlineSMOTE()                
        elif strategy == 'KMeansSMOTE':
            oversample = KMeansSMOTE(cluster_balance_threshold=0.1)    
        elif strategy == 'SVMSMOTE':
            oversample = SVMSMOTE()   
        elif strategy == 'SMOTETomek':
            oversample = SMOTETomek()   
        elif strategy == 'SMOTEENN':
            oversample = SMOTEENN()               
        elif strategy == 'ADASYN':
            oversample = ADASYN()
        else:
            oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)

        X_train, y_train = oversample.fit_resample(X_train, y_train)

        true_labels = len(y_train[y_train >= 0.5 ]) 
        false_labels = len(y_train[y_train < 0.5 ]) 
        if verbosity > 0:
            print('True Ratio: ', str(true_labels/(true_labels+false_labels)))    

    return X_train, y_train





def preprocess_data(X_data, 
                    y_data,
                    nominal_features,
                    ordinal_features,
                    random_seed=42,
                    verbosity=0):
    
    start_evaluate_network_complete = time.time()

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    if verbosity > 0:
        print('Original Data Shape (selected): ', X_data.shape)

    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), nominal_features)], remainder='passthrough', sparse_threshold=0)
    transformer.fit(X_data)

    X_data = transformer.transform(X_data)
    X_data = pd.DataFrame(X_data, columns=transformer.get_feature_names())

    for ordinal_feature in ordinal_features:
        X_data[ordinal_feature] = OrdinalEncoder().fit_transform(X_data[ordinal_feature].values.reshape(-1, 1)).flatten()

    X_data = X_data.astype(np.float64)

    if verbosity > 0:
        print('Original Data Shape (encoded): ', X_data.shape)
        print('Original Data Class Distribution: ', y_data[y_data>=0.5].shape[0], ' (true) /', y_data[y_data<0.5].shape[0], ' (false)')

    X_data, normalizer_list = normalize_data(X_data)

    (X_train, 
     y_train, 
     X_valid, 
     y_valid, 
     X_test, 
     y_test) = split_train_test_valid(X_data, 
                                      y_data, 
                                      seed=random_seed,
                                      verbosity=verbosity)    
    
    X_train, y_train = rebalance_data(X_train, 
                                      y_train, 
                                      balance_ratio=0.25, 
                                      strategy='SMOTE',
                                      verbosity=verbosity)    
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), normalizer_list
    
    
def get_preprocessed_dataset(identifier, 
                             random_seed=42, 
                             config=None,
                             verbosity=0):
    
    return_dict = {
        identifier: {}
    }
    
    if identifier == 'Cervical Cancer':
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv', index_col=False)#, names=feature_names

        features_select = [
                            'Age',
                            'Number of sexual partners',
                            'First sexual intercourse',
                            'Num of pregnancies',
                            'Smokes',
                            'Smokes (years)',
                            'Hormonal Contraceptives',
                            'Hormonal Contraceptives (years)',
                            'IUD',
                            'IUD (years)',
                            'STDs',
                            'STDs (number)',
                            'STDs: Number of diagnosis',
                            'STDs: Time since first diagnosis',
                            'STDs: Time since last diagnosis',
                            'Biopsy'
                           ]

        data = data[features_select]

        data['Number of sexual partners'][data['Number of sexual partners'] == '?'] = data['Number of sexual partners'].mode()[0]
        data['First sexual intercourse'][data['First sexual intercourse'] == '?'] = data['First sexual intercourse'].mode()[0]
        data['Num of pregnancies'][data['Num of pregnancies'] == '?'] = data['Num of pregnancies'].mode()[0]
        data['Smokes'][data['Smokes'] == '?'] = data['Smokes'].mode()[0]
        data['Smokes (years)'][data['Smokes (years)'] == '?'] = data['Smokes (years)'].mode()[0]
        data['Hormonal Contraceptives'][data['Hormonal Contraceptives'] == '?'] = data['Hormonal Contraceptives'].mode()[0]
        data['Hormonal Contraceptives (years)'][data['Hormonal Contraceptives (years)'] == '?'] = data['Hormonal Contraceptives (years)'].mode()[0]
        data['IUD'][data['IUD'] == '?'] = data['IUD'].mode()[0]
        data['IUD (years)'][data['IUD (years)'] == '?'] = data['IUD (years)'].mode()[0]
        data['STDs'][data['STDs'] == '?'] = data['STDs'].mode()[0]
        data['STDs (number)'][data['STDs (number)'] == '?'] = data['STDs (number)'].mode()[0]
        data['STDs: Time since first diagnosis'][data['STDs: Time since first diagnosis'] == '?'] = data['STDs: Time since first diagnosis'][data['STDs: Time since first diagnosis'] != '?'].mode()[0]
        data['STDs: Time since last diagnosis'][data['STDs: Time since last diagnosis'] == '?'] = data['STDs: Time since last diagnosis'][data['STDs: Time since last diagnosis'] != '?'].mode()[0]

        nominal_features = [
                            ]
        
        ordinal_features = [
                            ]


        X_data = data.drop(['Biopsy'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Biopsy'].values.reshape(-1, 1)).flatten(), name='Biopsy')

    elif identifier == 'Credit Card':
        data = pd.read_csv('./real_world_datasets/UCI_Credit_Card/UCI_Credit_Card.csv', index_col=False)
        data = data.drop(['ID'], axis = 1)

        nominal_features = [
                            ]

        ordinal_features = [
                            ]

        X_data = data.drop(['default.payment.next.month'], axis = 1)
        y_data = ((data['default.payment.next.month'] < 1) * 1)
      
    elif identifier == 'Absenteeism':

        data = pd.read_csv('real_world_datasets/Absenteeism/absenteeism.csv', delimiter=';')

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

        data = data[features_select]

        nominal_features = [
                            ]

        ordinal_features = [
                            ]

        X_data = data.drop(['Absenteeism time in hours'], axis = 1)
        y_data = ((data['Absenteeism time in hours'] > 4) * 1) #absenteeism_data['Absenteeism time in hours']

    elif identifier == 'Adult':
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

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=feature_names, index_col=False)


        #adult_data['Workclass'][adult_data['Workclass'] != ' Private'] = 'Other'
        #adult_data['Race'][adult_data['Race'] != ' White'] = 'Other'

        #adult_data.head()

        features_select = [
                         "Sex",  #9 
                         "Race",  #8
                         "Workclass",  #1
                         "Age", #0
                         "fnlwgt",  #2
                         #"Education",  #3
                         "Education-Num",  #4
                         "Marital Status", #5
                         #"Occupation",  #6
                         #"Relationship",  #7
                         "Capital Gain",  #10
                         "Capital Loss", #11
                         "Hours per week",  #12
                         #"Country", #13 
                         "capital_gain"
                          ]

        data = data[features_select]

        nominal_features = [
                                  'Race', 
                                  'Workclass', 
                                  #'Education',
                                  "Marital Status",
                                  #"Occupation", 
                                  #"Relationship"
                                ]
        ordinal_features = ['Sex']

        X_data = data.drop(['capital_gain'], axis = 1)
        y_data = ((data['capital_gain'] != ' <=50K') * 1)        
        
    elif identifier == 'Titanic':
        data = pd.read_csv("./real_world_datasets/Titanic/train.csv")

        data['Age'].fillna(data['Age'].mean(), inplace = True)
        data['Fare'].fillna(data['Fare'].mean(), inplace = True)

        data['Embarked'].fillna('S', inplace = True)

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

        data = data[features_select]

        nominal_features = ['Embarked']#[1, 2, 7]
        ordinal_features = ['Sex']

        X_data = data.drop(['Survived'], axis = 1)
        y_data = data['Survived']

    elif identifier == 'Loan House':
        data = pd.read_csv('real_world_datasets/Loan/loan-train.csv', delimiter=',')

        data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
        data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
        data['Married'].fillna(data['Married'].mode()[0], inplace=True)
        data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
        data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        data['Credit_History'].fillna(data['Credit_History'].mean(), inplace=True)

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

        data = data[features_select]

        #loan_data['Dependents'][loan_data['Dependents'] == '3+'] = 4
        #loan_data['Dependents'] = loan_data['Dependents'].astype(int)

        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Rural'] = 0
        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Semiurban'] = 1
        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Urban'] = 2
        #loan_data['Property_Area'] = loan_data['Property_Area'].astype(int)

        nominal_features = [
                                'Dependents',
                                'Property_Area',    
                                ]


        ordinal_features = [
                            'Education',
                            'Gender', 
                            'Married', 
                            'Self_Employed',
                           ]

        X_data = data.drop(['Loan_Status'], axis = 1)
        y_data = ((data['Loan_Status'] == 'Y') * 1)         
        
    elif identifier == 'Loan Credit':

        data = pd.read_csv('real_world_datasets/Credit Loan/train_split.csv', delimiter=',')

        data['emp_title'].fillna(data['emp_title'].mode()[0], inplace=True)
        data['emp_length'].fillna(data['emp_length'].mode()[0], inplace=True)
        #data['desc'].fillna(data['desc'].mode()[0], inplace=True)
        data['title'].fillna(data['title'].mode()[0], inplace=True)
        #data['mths_since_last_delinq'].fillna(data['mths_since_last_delinq'].mode()[0], inplace=True)
        #data['mths_since_last_record'].fillna(data['mths_since_last_record'].mode()[0], inplace=True)
        data['revol_util'].fillna(data['revol_util'].mode()[0], inplace=True)
        data['collections_12_mths_ex_med'].fillna(data['collections_12_mths_ex_med'].mode()[0], inplace=True)
        #data['mths_since_last_major_derog'].fillna(data['mths_since_last_major_derog'].mode()[0], inplace=True)
        #data['verification_status_joint'].fillna(data['verification_status_joint'].mode()[0], inplace=True)
        data['tot_coll_amt'].fillna(data['tot_coll_amt'].mode()[0], inplace=True)
        data['tot_cur_bal'].fillna(data['tot_cur_bal'].mode()[0], inplace=True)
        data['total_rev_hi_lim'].fillna(data['total_rev_hi_lim'].mode()[0], inplace=True)


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

        data = data[features_select]

        nominal_features = [

                                ]
        ordinal_features = [
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

        X_data = data.drop(['loan_status'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['loan_status'].values.reshape(-1, 1)).flatten(), name='loan_status')

    elif identifier == 'Medical Insurance':
        
        data = pd.read_csv('real_world_datasets/Medical Insurance/insurance.csv', delimiter=',')

        features_select = [
                            'age', 
                            'sex', 
                            'bmi', 
                            'children', 
                            'smoker',
                            'region',
                            'charges'
                            ]

        data = data[features_select]

        nominal_features = [
                            'region',
                                ]
        ordinal_features = [
                            'sex',
                            'smoker'
                           ]


        X_data = data.drop(['charges'], axis = 1)
        y_data = ((data['charges'] > 10_000) * 1)

    elif identifier == 'Bank Marketing':

        data = pd.read_csv('real_world_datasets/Bank Marketing/bank-full.csv', delimiter=';') #bank

        features_select = [
                            'age',
                            'job', 
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

        data = data[features_select]

        nominal_features = [
                                'job',
                                'education',
                                #'contact',
                                #'day',
                                #'month',
                                'poutcome',
                                ]
        ordinal_features = [
                            'marital',
                            'default',
                            'housing',
                            'loan',
                           ]


        X_data = data.drop(['y'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['y'].values.reshape(-1, 1)).flatten(), name='y')

    elif identifier == 'Wisconsin Breast Cancer Original':
        
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

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=feature_names, index_col=False)

        data['Clump Thickness'][data['Clump Thickness'] == '?'] = data['Clump Thickness'].mode()[0]
        data['Uniformity of Cell Size'][data['Uniformity of Cell Size'] == '?'] = data['Uniformity of Cell Size'].mode()[0]
        data['Uniformity of Cell Shape'][data['Uniformity of Cell Shape'] == '?'] = data['Uniformity of Cell Shape'].mode()[0]
        data['Marginal Adhesion'][data['Marginal Adhesion'] == '?'] = data['Marginal Adhesion'].mode()[0]
        data['Single Epithelial Cell Size'][data['Single Epithelial Cell Size'] == '?'] = data['Single Epithelial Cell Size'].mode()[0]
        data['Bare Nuclei'][data['Bare Nuclei'] == '?'] = data['Bare Nuclei'].mode()[0]
        data['Bland Chromatin'][data['Bland Chromatin'] == '?'] = data['Bland Chromatin'].mode()[0]
        data['Normal Nucleoli'][data['Normal Nucleoli'] == '?'] = data['Normal Nucleoli'].mode()[0]
        data['Mitoses'][data['Mitoses'] == '?'] = data['Mitoses'].mode()[0]

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

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                           ]


        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='Class')

    elif identifier == 'Wisconsin Diagnostic Breast Cancer':

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
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', names=feature_names, index_col=False)

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

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                           ]


        X_data = data.drop(['Diagnosis'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')
       
    elif identifier == 'Wisconsin Prognostic Breast Cancer':

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
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data', names=feature_names, index_col=False)

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

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            ]

        X_data = data.drop(['Diagnosis'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')

    elif identifier == 'Abalone':
        
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

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', names=feature_names, index_col=False)


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

        data = data[features_select]

        nominal_features = [
                                'Sex',
                                ]
        ordinal_features = [
                           ]

        X_data = data.drop(['Rings'], axis = 1)
        y_data = ((data['Rings'] > 10) * 1)

    elif identifier == 'Car':
        feature_names = [
           'buying',#       v-high, high, med, low
           'maint',#        v-high, high, med, low
           'doors',#        2, 3, 4, 5-more
           'persons',#      2, 4, more
           'lug_boot',#     small, med, big
           'safety',#       low, med, high
           'class',#        unacc, acc, good, v-good
                        ]

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=feature_names, index_col=False)

        features_select = [
                           'buying',#       v-high, high, med, low
                           'maint',#        v-high, high, med, low
                           'doors',#        2, 3, 4, 5-more
                           'persons',#      2, 4, more
                           'lug_boot',#     small, med, big
                           'safety',#       low, med, high
                           'class',#        unacc, acc, good, v-good
                            ]

        data = data[features_select]

        nominal_features = [
                               'buying',#       v-high, high, med, low
                               'maint',#        v-high, high, med, low
                               'doors',#        2, 3, 4, 5-more
                               'persons',#      2, 4, more
                               'lug_boot',#     small, med, big
                               'safety',#       low, med, high
                                ]

        ordinal_features = [
                           ]



        X_data = data.drop(['class'], axis = 1)
        y_data = ((data['class'] != 'unacc') * 1)        

    elif identifier == 'Heart Disease':
        feature_names = [
           'age',#      
           'sex',#   
           'cp',#      
           'trestbps',#
           'chol',#    
           'fbs',#      
           'restecg',# 
           'thalach',#      
           'exang',#   
           'oldpeak',#      
           'slope',#
           'ca',#    
           'thal',#      
           'num',#     
                        ]

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', names=feature_names, index_col=False) #, delimiter=' '

        nominal_features = [
                                ]

        ordinal_features = [
                           ]


        data['age'][data['age'] == '?'] = data['age'].mode()[0]
        data['sex'][data['sex'] == '?'] = data['sex'].mode()[0]
        data['cp'][data['cp'] == '?'] = data['cp'].mode()[0]
        data['trestbps'][data['trestbps'] == '?'] = data['trestbps'].mode()[0]
        data['chol'][data['chol'] == '?'] = data['chol'].mode()[0]
        data['fbs'][data['fbs'] == '?'] = data['fbs'].mode()[0]
        data['restecg'][data['restecg'] == '?'] = data['restecg'].mode()[0]
        data['thalach'][data['thalach'] == '?'] = data['thalach'].mode()[0]
        data['exang'][data['exang'] == '?'] = data['exang'].mode()[0]
        data['oldpeak'][data['oldpeak'] == '?'] = data['oldpeak'].mode()[0]
        data['slope'][data['slope'] == '?'] = data['slope'].mode()[0]
        data['ca'][data['ca'] == '?'] = data['ca'].mode()[0]
        data['thal'][data['thal'] == '?'] = data['thal'].mode()[0]

        X_data = data.drop(['num'], axis = 1)
        y_data = ((data['num'] < 1) * 1)
        
    elif identifier == 'Habermans Survival':

        feature_names = [
           'age',#      
           'year',#   
           'nodes_detected',#      
           'survival',#     
                        ]

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', names=feature_names, index_col=False) #, delimiter=' '


        nominal_features = [
                                ]

        ordinal_features = [
                           ]


        X_data = data.drop(['survival'], axis = 1)
        y_data = ((data['survival'] < 2) * 1)

    elif identifier == 'Heart Failure':
        
        data = pd.read_csv('real_world_datasets/Heart Failure/heart_failure_clinical_records_dataset.csv', delimiter=',')


        nominal_features = [
                                ]
        ordinal_features = [

                           ]


        X_data = data.drop(['DEATH_EVENT'], axis = 1)
        y_data = ((data['DEATH_EVENT'] > 0) * 1)
     
    elif identifier == 'make_classification':
        
        informative = np.random.randint(config['number_of_variables']//2, high=config['number_of_variables']+1) #config['data']['number_of_variables']
        redundant = np.random.randint(0, high=config['number_of_variables']-informative+1) #0
        repeated = config['number_of_variables']-informative-redundant # 0

        n_clusters_per_class =  max(2, np.random.randint(0, high=informative//2+1)) #2

        X_data, y_data = make_classification(n_samples=config['n_samples'], 
                                               n_features=config['number_of_variables'], #The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
                                               n_informative=informative,#config['data']['number_of_variables'], #The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative.
                                               n_redundant=redundant, #The number of redundant features. These features are generated as random linear combinations of the informative features.
                                               n_repeated=repeated, #The number of duplicated features, drawn randomly from the informative and the redundant features.
                                               n_classes=2, 
                                               n_clusters_per_class=n_clusters_per_class, 
                                               flip_y=config['noise'], #The fraction of samples whose class is assigned randomly. 
                                               #class_sep=1.0, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                               #hypercube=False, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                               #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                               #scale=1.0, #Multiply features by the specified value. 
                                               shuffle=True, 
                                               random_state=random_seed)         
        
        
        nominal_features = []
        ordinal_features = []        

    else:
        raise SystemExit('Unknown key: ' + str(identifier))
    
    
    
    ((return_dict['X_train'], return_dict['y_train']),
     (return_dict['X_valid'], return_dict['y_valid']),
     (return_dict['X_test'], return_dict['y_test']),
     return_dict['normalizer_list']) = preprocess_data(X_data, 
                                                       y_data,
                                                       nominal_features,
                                                       ordinal_features,
                                                       random_seed=random_seed,
                                                       verbosity=verbosity)       
        
    return return_dict


def evaluate_dhdt(identifier, 
                  random_seed_data=42, 
                  random_seed_model=42, 
                  config=None,
                  sklearn_params=None,
                  metrics=['accuracy', 'f1'],
                  verbosity=0):

    if verbosity > 0:
        print('________________________________________________________________________________________________________')   
    
    dataset_dict = {}
    model_dict = {}

    scores_dict = {'sklearn': {},
                   'XGB': {},
                   'DHDT': {},
                   'GeneticTree': {}}
    
    dataset_dict = get_preprocessed_dataset(identifier,
                                            random_seed=random_seed_data,
                                            config=config['make_classification'],
                                            verbosity=verbosity)

    
    ##############################################################
    if sklearn_params is None:
        sklearn_params = {'max_depth': 3,
                          'random_state': random_seed_model}
    
    
    model_dict['sklearn'] = DecisionTreeClassifier()
    model_dict['sklearn'].set_params(**sklearn_params)

    
    #model_dict['sklearn'] = DecisionTreeClassifier(max_depth=config['dhdt']['depth'], 
    #                                               random_state=random_seed_model)

    start_sklearn = timeit.default_timer()
    
    model_dict['sklearn'].fit(dataset_dict['X_train'], 
                              dataset_dict['y_train'])    
    
    
    end_sklearn = timeit.default_timer()  
    runtime_sklearn = end_sklearn - start_sklearn

    
    ##############################################################
    model_dict['XGB'] = XGBClassifier(random_state = random_seed_model, 
                                      eval_metric = 'logloss',
                                      n_jobs = 1)
    
    
    start_xgb = timeit.default_timer()
    
    model_dict['XGB'] = model_dict['XGB'].fit(dataset_dict['X_train'], 
                                              dataset_dict['y_train']) 
    
    end_xgb = timeit.default_timer()  
    runtime_xgb = end_xgb - start_xgb
    
    ##############################################################
    model_dict['GeneticTree'] = GeneticTree(max_depth=config['dhdt']['depth'],
                                            random_state = random_seed_model, 
                                            n_jobs = 1)
    
    
    start_gentree = timeit.default_timer()
    
    model_dict['GeneticTree'] = model_dict['GeneticTree'].fit(dataset_dict['X_train'].values, 
                                                              dataset_dict['y_train'].values) 
    
    end_gentree = timeit.default_timer()  
    runtime_gentree = end_gentree - start_gentree 
    
    ##############################################################

    model_dict['DHDT'] = DHDT(dataset_dict['X_train'].shape[1],

                                depth = config['dhdt']['depth'],

                                learning_rate = config['dhdt']['learning_rate'],
                                optimizer = config['dhdt']['optimizer'],
                              
                                initializer = config['dhdt']['initializer'],
                                initializer_index = config['dhdt']['initializer_index'],
                              
                                beta_1 = config['dhdt']['beta_1'],
                                beta_2 = config['dhdt']['beta_2'],
                              
                                sparse_activation_1 = config['dhdt']['sparse_activation_1'],
                                sparse_activation_2 = config['dhdt']['sparse_activation_2'],
                              
                                activation = config['dhdt']['activation'],
                                squeeze_factor = config['dhdt']['squeeze_factor'],

                                loss = config['dhdt']['loss'],#'mae',

                                random_seed = random_seed_model,
                                verbosity = verbosity)        


    start_dhdt = timeit.default_timer()
    
    scores_dict['history'] = model_dict['DHDT'].fit(dataset_dict['X_train'], 
                                                  dataset_dict['y_train'], 
                                                  batch_size=config['dhdt']['batch_size'], 
                                                  epochs=config['dhdt']['epochs'], 
                                                  early_stopping_epochs=config['dhdt']['early_stopping_epochs'], 
                                                  valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))

    end_dhdt = timeit.default_timer()
    runtime_dhdt = end_dhdt - start_dhdt
    
    ##############################################################
    
    #print(runtime_dhdt, runtime_sklearn, runtime_xgb)
    
    dataset_dict['y_test_dhdt'] = model_dict['DHDT'].predict(dataset_dict['X_test'])
    dataset_dict['y_valid_dhdt'] = model_dict['DHDT'].predict(dataset_dict['X_valid'])

    dataset_dict['y_test_sklearn'] = model_dict['sklearn'].predict(dataset_dict['X_test'])
    dataset_dict['y_valid_sklearn'] = model_dict['sklearn'].predict(dataset_dict['X_valid'])   
    
    dataset_dict['y_test_xgb'] = model_dict['XGB'].predict(dataset_dict['X_test'])
    dataset_dict['y_valid_xgb'] = model_dict['XGB'].predict(dataset_dict['X_valid'])     
    
    dataset_dict['y_test_gentree'] = model_dict['GeneticTree'].predict(dataset_dict['X_test'].values)
    dataset_dict['y_valid_gentree'] = model_dict['GeneticTree'].predict(dataset_dict['X_valid'].values)     
        
    for metric in metrics:
        
        if metric in ['accuracy', 'f1']:
            y_test_dhdt = np.round(dataset_dict['y_test_dhdt'])
            y_valid_dhdt = np.round(dataset_dict['y_valid_dhdt'])
            y_test_sklearn = np.round(dataset_dict['y_test_sklearn'])
            y_valid_sklearn = np.round(dataset_dict['y_valid_sklearn'])         
            y_test_xgb = np.round(dataset_dict['y_test_xgb'])
            y_valid_xgb = np.round(dataset_dict['y_valid_xgb'])       
            y_test_gentree = np.round(dataset_dict['y_test_gentree'])
            y_valid_gentree = np.round(dataset_dict['y_valid_gentree'])              
        else:
            y_test_dhdt = dataset_dict['y_test_dhdt']
            y_valid_dhdt = dataset_dict['y_valid_dhdt']
            y_test_sklearn = dataset_dict['y_test_sklearn']
            y_valid_sklearn = dataset_dict['y_valid_sklearn']       
            y_test_xgb = dataset_dict['y_test_xgb']
            y_valid_xgb = dataset_dict['y_valid_xgb']    
            y_test_gentree = dataset_dict['y_test_gentree']
            y_valid_gentree = dataset_dict['y_valid_gentree']   
            
        if metric != 'f1':
            scores_dict['sklearn'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_sklearn)
            scores_dict['XGB'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_xgb)
            scores_dict['DHDT'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_dhdt)
            scores_dict['GeneticTree'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_gentree)

            scores_dict['sklearn'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_sklearn)  
            scores_dict['XGB'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_xgb)   
            scores_dict['DHDT'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_dhdt)
            scores_dict['GeneticTree'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_gentree)
        else:
            scores_dict['sklearn'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_sklearn, average='weighted')
            scores_dict['XGB'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_xgb, average='weighted')
            scores_dict['DHDT'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_dhdt, average='weighted')
            scores_dict['GeneticTree'][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_test'], y_test_gentree, average='weighted')

            scores_dict['sklearn'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_sklearn, average='weighted')   
            scores_dict['XGB'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_xgb, average='weighted')   
            scores_dict['DHDT'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_dhdt, average='weighted')  
            scores_dict['GeneticTree'][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(dataset_dict['y_valid'], y_valid_gentree, average='weighted')  
            
        if verbosity > 0:
            print('Test ' + metric + ' Sklearn (' + identifier + ')', scores_dict['sklearn'][metric + '_test'])
            print('Test ' + metric + ' XGB (' + identifier + ')', scores_dict['XGB'][metric + '_test'])
            print('Test ' + metric + ' DHDT (' + identifier + ')', scores_dict['DHDT'][metric + '_test'])   
            print('Test ' + metric + ' GeneticTree (' + identifier + ')', scores_dict['GeneticTree'][metric + '_test'])   
            print('________________________________________________________________________________________________________')   

    scores_dict['DHDT']['runtime'] = runtime_dhdt
    scores_dict['sklearn']['runtime'] = runtime_sklearn
    scores_dict['XGB']['runtime'] = runtime_xgb     
    scores_dict['GeneticTree']['runtime'] = runtime_gentree  
            
    return identifier, dataset_dict, model_dict, scores_dict
    
    
def evaluate_synthetic_parallel(index,
                               random_seed_data=42, 
                               random_seed_model=42, 
                               config=None,
                               metrics=['accuracy', 'f1'],
                               verbosity=0):

    dataset_dict = {}
    model_dict = {}

    scores_dict = {}
    
    disable = True if verbosity <= 0 else False
    for trial_num in range(config['computation']['trials']):    
        dataset_dict_trial = {}
        model_dict_trial = {}

        scores_dict_trial = {}    

        (identifier,
         dataset_dict_trial[index], 
         model_dict_trial[index], 
         scores_dict_trial[index]) = evaluate_dhdt(identifier='make_classification', 
                                                  random_seed_data=random_seed_data, 
                                                  random_seed_model=random_seed_model+trial_num, 
                                                  config=config,
                                                  metrics=metrics,
                                                  verbosity=verbosity)

        if dataset_dict == {}:
            dataset_dict[index] = dataset_dict_trial[index]
        else:
            dataset_dict[index] = mergeDict(dataset_dict[index], dataset_dict_trial[index])

        if  model_dict == {}:
            model_dict[index] = model_dict_trial[index]
        else:
            model_dict[index] = mergeDict(model_dict[index], model_dict_trial[index])

        if  scores_dict == {}:
            scores_dict[index] = scores_dict_trial[index]
        else:
            scores_dict[index] = mergeDict(scores_dict[index], scores_dict_trial[index])
                
        
    return model_dict, scores_dict, dataset_dict

def evaluate_real_world_parallel(identifier_list, 
                                  random_seed_data=42, 
                                  random_seed_model=42, 
                                  config=None,
                                  sklearn_params = None,
                                  metrics=['accuracy', 'f1'],
                                  verbosity=0):

    dataset_dict = {}
    model_dict = {}

    scores_dict = {}
    
    disable = True if verbosity <= 0 else False
    for identifier in tqdm(identifier_list, desc='dataset loop', disable=disable):

        (identifier,
         dataset_dict[identifier], 
         model_dict[identifier], 
         scores_dict[identifier]) = evaluate_dhdt(identifier, 
                                                  random_seed_data=random_seed_data, 
                                                  random_seed_model=random_seed_model, 
                                                  config=config,
                                                  sklearn_params = sklearn_params,
                                                  metrics=metrics,
                                                  verbosity=verbosity)
        
    return model_dict, scores_dict, dataset_dict


def evaluate_parameter_setting_synthetic(parameter_setting, 
                                         config, 
                                         metrics=['accuracy', 'f1']):
    
    config_parameter_setting = deepcopy(config)
    
    
    for key, value in parameter_setting.items():
        config_parameter_setting['dhdt'][key] = value
    
    
    evaluation_results_synthetic = []
    for index in range(config['make_classification']['num_eval']):
        evaluation_result = evaluate_synthetic_parallel(index = index,
                                                        random_seed_data = config['computation']['random_seed']+index,
                                                        random_seed_model = config['computation']['random_seed'],#+random_seed_model,
                                                        config = config_parameter_setting,
                                                        metrics = metrics,
                                                        verbosity = -1)
        evaluation_results_synthetic.append(evaluation_result)
    
    #parallel_eval_synthetic = Parallel(n_jobs=1, verbose=0, backend='sequential') #loky #sequential multiprocessing
    #evaluation_results_synthetic = parallel_eval_synthetic(delayed(evaluate_synthetic_parallel)(index = index,
    #                                                                                            random_seed_data = config['computation']['random_seed']+index,
    #                                                                                            random_seed_model = config['computation']['random_seed'],#+random_seed_model,
    #                                                                                            config = config_parameter_setting,
    #                                                                                            verbosity = -1) for index in range(config['make_classification']['num_eval']))

    
    for i, synthetic_result in enumerate(evaluation_results_synthetic):
        if i == 0:
            model_dict_synthetic = synthetic_result[0]
            scores_dict_synthetic = synthetic_result[1]
            dataset_dict_synthetic = synthetic_result[2]
        else: 
            model_dict_synthetic = mergeDict(model_dict_synthetic, synthetic_result[0])
            scores_dict_synthetic = mergeDict(scores_dict_synthetic, synthetic_result[1])
            dataset_dict_synthetic = mergeDict(dataset_dict_synthetic, synthetic_result[2])    
    
    del synthetic_result, evaluation_results_synthetic
    
    
    metric_identifer = '_valid'

    index = [i for i in range(config['make_classification']['num_eval'])]
    columns = flatten_list([[[approach + ' ' + metric + '_mean', approach + ' ' + metric + '_max', approach + ' ' + metric + '_std'] for metric in metrics] for approach in ['DHDT', 'sklearn']])


    results_DHDT = None
    results_sklearn = None
    for metric in metrics:
        scores_DHDT = [scores_dict_synthetic[i]['DHDT'][metric + metric_identifer] for i in range(config['make_classification']['num_eval'])]

        scores_sklearn = [scores_dict_synthetic[i]['sklearn'][metric + metric_identifer] for i in range(config['make_classification']['num_eval'])]

        scores_DHDT_mean = np.mean(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else scores_DHDT
        scores_sklearn_mean = np.mean(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else scores_sklearn

        scores_DHDT_max = np.max(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else scores_DHDT
        scores_sklearn_max = np.max(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else scores_sklearn

        scores_DHDT_std = np.std(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])
        scores_sklearn_std = np.std(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])

        results_DHDT_by_metric = np.vstack([scores_DHDT_mean, scores_DHDT_max, scores_DHDT_std])
        results_sklearn_by_metric = np.vstack([scores_sklearn_mean, scores_sklearn_max, scores_sklearn_std])

        if results_DHDT is None and results_sklearn is None:
            results_DHDT = results_DHDT_by_metric
            results_sklearn = results_sklearn_by_metric
        else:
            results_DHDT = np.vstack([results_DHDT, results_DHDT_by_metric])
            results_sklearn = np.vstack([results_sklearn, results_sklearn_by_metric])

    scores_dataframe_synthetic = pd.DataFrame(data=np.vstack([results_DHDT, results_sklearn]).T, index = index, columns = columns)    
    
    del model_dict_synthetic, scores_dict_synthetic, dataset_dict_synthetic
    
    return scores_dataframe_synthetic, parameter_setting
    
    
def evaluate_parameter_setting_real_world(parameter_setting, 
                                          identifier, 
                                          config, 
                                          sklearn_params = None,
                                          metrics=['accuracy', 'f1']):
    
    config_parameter_setting = deepcopy(config)
    
    
    for key, value in parameter_setting.items():
        config_parameter_setting['dhdt'][key] = value
    
    
    evaluation_results_real_world = []
    for i in range(config['computation']['trials']):
        evaluation_result = evaluate_real_world_parallel(identifier_list=[identifier], 
                                                           random_seed_model=config['computation']['random_seed']+i,
                                                           config = config_parameter_setting,
                                                           sklearn_params = sklearn_params,
                                                           metrics = metrics,
                                                           verbosity = -1)
        evaluation_results_real_world.append(evaluation_result)
        
    del evaluation_result

    for i, real_world_result in enumerate(evaluation_results_real_world):
        if i == 0:
            model_dict_real_world = real_world_result[0]
            scores_dict_real_world = real_world_result[1]
            dataset_dict_real_world = real_world_result[2]
        else: 
            model_dict_real_world = mergeDict(model_dict_real_world, real_world_result[0])
            scores_dict_real_world = mergeDict(scores_dict_real_world, real_world_result[1])
            dataset_dict_real_world = mergeDict(dataset_dict_real_world, real_world_result[2])    

    del real_world_result, evaluation_results_real_world

    metric_identifer_list = ['_valid', '_test']

    index = [identifier]
    columns = flatten_list([[[approach + ' ' + metric + '_mean', 
                              approach + ' ' + metric + '_max', 
                              approach + ' ' + metric + '_std', 
                              approach + ' mean runtime'] for metric in metrics] for approach in ['DHDT', 
                                                                                                  'sklearn', 
                                                                                                  'XGB',
                                                                                                  'GeneticTree']])



    
    
    scores_dataframe_real_world_dict = {}
    
    for metric_identifer in metric_identifer_list:
    
        results_DHDT = None
        results_sklearn = None
        results_xgb = None    
        results_gentree = None    
        
        for metric in metrics:
            scores_DHDT = [scores_dict_real_world[identifier]['DHDT'][metric + metric_identifer] for identifier in [identifier]]
            scores_sklearn = [scores_dict_real_world[identifier]['sklearn'][metric + metric_identifer] for identifier in [identifier]]    
            scores_xgb = [scores_dict_real_world[identifier]['XGB'][metric + metric_identifer] for identifier in [identifier]]    
            scores_gentree = [scores_dict_real_world[identifier]['GeneticTree'][metric + metric_identifer] for identifier in [identifier]]    

            scores_DHDT_mean = np.mean(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else scores_DHDT
            scores_sklearn_mean = np.mean(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else scores_sklearn
            scores_xgb_mean = np.mean(scores_xgb, axis=1) if config['computation']['trials'] > 1 else scores_xgb
            scores_gentree_mean = np.mean(scores_gentree, axis=1) if config['computation']['trials'] > 1 else scores_gentree

            scores_DHDT_max = np.max(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else scores_DHDT
            scores_sklearn_max = np.max(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else scores_sklearn
            scores_xgb_max = np.max(scores_xgb, axis=1) if config['computation']['trials'] > 1 else scores_xgb
            scores_gentree_max = np.max(scores_gentree, axis=1) if config['computation']['trials'] > 1 else scores_gentree

            scores_DHDT_std = np.std(scores_DHDT, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])
            scores_sklearn_std = np.std(scores_sklearn, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])
            scores_xgb_std = np.std(scores_xgb, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])
            scores_gentree_std = np.std(scores_gentree, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * config['computation']['trials'])

            runtime_DHDT = [scores_dict_real_world[identifier]['DHDT']['runtime'] for identifier in [identifier]]
            runtime_sklearn = [scores_dict_real_world[identifier]['sklearn']['runtime'] for identifier in [identifier]]
            runtime_xgb = [scores_dict_real_world[identifier]['XGB']['runtime'] for identifier in [identifier]]
            runtime_gentree = [scores_dict_real_world[identifier]['GeneticTree']['runtime'] for identifier in [identifier]]

            #print(runtime_DHDT, runtime_sklearn, runtime_xgb)
            
            mean_runtime_DHDT = np.mean(runtime_DHDT, axis=1) if config['computation']['trials'] > 1 else runtime_DHDT
            mean_runtime_sklearn = np.mean(runtime_sklearn, axis=1) if config['computation']['trials'] > 1 else runtime_sklearn
            mean_runtime_xgb = np.mean(runtime_xgb, axis=1) if config['computation']['trials'] > 1 else runtime_xgb   
            mean_runtime_gentree = np.mean(runtime_gentree, axis=1) if config['computation']['trials'] > 1 else runtime_gentree 

            results_DHDT_by_metric = np.vstack([scores_DHDT_mean, scores_DHDT_max, scores_DHDT_std, mean_runtime_DHDT])
            results_sklearn_by_metric = np.vstack([scores_sklearn_mean, scores_sklearn_max, scores_sklearn_std, mean_runtime_sklearn])
            results_xgb_by_metric = np.vstack([scores_xgb_mean, scores_xgb_max, scores_xgb_std, mean_runtime_xgb])
            results_gentree_by_metric = np.vstack([scores_gentree_mean, scores_gentree_max, scores_gentree_std, mean_runtime_gentree])

            if results_DHDT is None and results_sklearn is None and results_xgb is None and results_gentree is None:
                results_DHDT = results_DHDT_by_metric
                results_sklearn = results_sklearn_by_metric
                results_xgb = results_xgb_by_metric
                results_gentree = results_gentree_by_metric
            else:
                results_DHDT = np.vstack([results_DHDT, results_DHDT_by_metric])
                results_sklearn = np.vstack([results_sklearn, results_sklearn_by_metric])
                results_xgb = np.vstack([results_xgb, results_xgb_by_metric])
                results_gentree = np.vstack([results_gentree, results_gentree_by_metric])
        #print(np.vstack([results_DHDT, results_sklearn, results_xgb]).T.shape)
        #print(index)
        #print(columns)
        scores_dataframe_real_world = pd.DataFrame(data=np.vstack([results_DHDT, results_sklearn, results_xgb, results_gentree]).T, index = index, columns = columns)
        scores_dataframe_real_world_dict[metric_identifer[1:]] = scores_dataframe_real_world
        #display(scores_dataframe_real_world)
        #display(scores_dataframe_real_world[scores_dataframe_real_world.columns[1::3]])    

    del scores_dict_real_world, dataset_dict_real_world

    #return_dict = {
    #                'DHDT score (mean)': np.mean(scores_DHDT_mean), 
    #                'Sklearn score (mean)': np.mean(scores_sklearn_mean), 
    #                'DHDT score (mean of max)': np.mean(scores_DHDT_max), 
    #                'Parameters': parameter_setting
    #              }
    
    #return scores_dataframe_real_world, parameter_setting
    return scores_dataframe_real_world_dict, parameter_setting, model_dict_real_world

    
    
def sleep_minutes(minutes):  
    if minutes > 0:
        for _ in tqdm(range(minutes)):
            time.sleep(60)