import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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
import warnings
import logging

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import seaborn as sns
import time
import random
from collections.abc import Iterable


from utilities.utilities import *


from utilities.DHDT import DHDT

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


def normalize_data(X_data):
    normalizer_list = []
    if isinstance(X_data, pd.DataFrame):
        for column_name in X_data:
            scaler = MinMaxScaler()
            scaler.fit(X_data[column_name].values.reshape(-1, 1))
            X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
            normalizer_list.append(scaler)
    else:
        for i, column in enumerate(X_data.T):
            scaler = MinMaxScaler()
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

    elif identifier == 'Medical Insurance':

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

    elif identifier == 'Wisconsin Diagnositc Breast Cancer':

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
       
    elif identifier == 'Wisconsin Prognositc Breast Cancer':

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

    elif identifier == 'Habermans Survival':
        
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
                                               #flip_y=0.0, #The fraction of samples whose class is assigned randomly. 
                                               #class_sep=1.0, #The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.
                                               #hypercube=False, #If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
                                               #shift=0.0, #Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                                               #scale=1.0, #Multiply features by the specified value. 
                                               shuffle=True, 
                                               random_state=random_seed)         
        
        
        nominal_features = []
        ordinal_features = []        

    
    
    
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
                  verbosity=0):

    if verbosity > 0:
        print('________________________________________________________________________________________________________')   
    
    dataset_dict = {}
    model_dict = {}

    scores_dict = {'sklearn': {},
                   'DHDT': {}}
    
    dataset_dict = get_preprocessed_dataset(identifier,
                                            random_seed=random_seed_data,
                                            config=config,
                                            verbosity=verbosity)

    model_dict['sklearn'] = DecisionTreeClassifier(max_depth=3, 
                                                   random_state=random_seed_model)

    model_dict['sklearn'].fit(dataset_dict['X_train'], 
                              dataset_dict['y_train'])

    scores_dict['sklearn']['accuracy'] = model_dict['sklearn'].score(dataset_dict['X_test'], 
                                                                     dataset_dict['y_test'])



    model_dict['DHDT'] = DHDT(depth=3,
                             number_of_variables = dataset_dict['X_train'].shape[1],
                             learning_rate=1e-3,
                             squeeze_factor = 1,
                             loss='binary_crossentropy',#'binary_crossentropy',
                             optimizer='rmsprop',
                             random_seed=random_seed_model,
                             verbosity=verbosity)

    scores_dict['history'] = model_dict['DHDT'].fit(dataset_dict['X_train'], 
                                                  dataset_dict['y_train'], 
                                                  batch_size=512, 
                                                  epochs=1_000, 
                                                  early_stopping_epochs=50, 
                                                  valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))

    dataset_dict['y_test_dhdt'] = model_dict['DHDT'].predict(dataset_dict['X_test'])
    scores_dict['DHDT']['accuracy'] = accuracy_score(dataset_dict['y_test'], np.round(dataset_dict['y_test_dhdt']))
    
    if verbosity > 0:
        print('Test Accuracy Sklearn (' + identifier + ')', scores_dict['sklearn']['accuracy'])
        print('Test Accuracy DHDT (' + identifier + ')', scores_dict['DHDT']['accuracy'])   
        print('________________________________________________________________________________________________________')   

    return identifier, dataset_dict, model_dict, scores_dict
    
    
def evaluate_synthetic_parallel(index,
                               random_seed_data=42, 
                               random_seed_model=42, 
                               trials = 1,
                               config=None,
                               verbosity=0):

    dataset_dict = {}
    model_dict = {}

    scores_dict = {}
    
    disable = True if verbosity <= 0 else False
    for trial_num in range(trials):    
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
                                                  verbosity=verbosity)
        
    return model_dict, scores_dict, dataset_dict
