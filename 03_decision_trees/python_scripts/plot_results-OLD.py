#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy
import researchpy as rp

from python_scripts.interpretation_net_evaluate import extend_inet_parameter_setting


import seaborn as sns
from copy import deepcopy

import itertools
from collections.abc import Iterable

import traceback

sns.set_style("darkgrid")

pd.set_option('max_columns', 50) 

import warnings
from tabulate import tabulate

import os

warnings.filterwarnings('ignore')


# # Function Definitions

# In[2]:


def flatten_list(l):
    
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
                
    flat_l = flatten(l)
    
    return list(flat_l)


def get_relevant_columns_by_config(config, dataframe):
    try:
        if config['i_net_nas'] == False:
            config.pop('i_net_nas_trials')
    except:
        pass
    
    for key, value in config.items():
        try:
            if isinstance(value, list):
                if isinstance(value[0], str):
                    dataframe_string_query = key + ' == "' + str(value[0]) + '"'
                    for dataframe_string in value[1:]:
                        dataframe_string_query += ' | ' + key + ' == "' + str(dataframe_string) + '"'

                    dataframe = dataframe.query(dataframe_string_query)
                else:
                    dataframe = dataframe[dataframe[key].isin(value)]
                    
            else:
                dataframe = dataframe[dataframe[key] == value]
        except:
            traceback.print_exc()
        
    return dataframe


def plot_results(data_reduced, col, x, y, hue, plot_type=sns.barplot, aspect=1.5, col_wrap=2):
    
    #sns.set(rc={'figure.figsize':(20,10)})
    
    g = sns.FacetGrid(data_reduced, 
                      col=col,
                      ##hue='scores_type', 
                      #height=5, 
                      col_wrap=col_wrap,
                      aspect=aspect,
                      ##legend_out=False,
                     )    
    indexes = np.unique(data_reduced[hue], return_index=True)[1]
    hue_order = [data_reduced[hue].values[index] for index in sorted(indexes)]
        
    g.map(plot_type, 
          x, 
          y, 
          hue,
          hue_order=hue_order,#np.unique(data_reduced[hue]),
          ##figsize=(20,10),
          palette=sns.color_palette(),#'colorblind'
          #order=data_reduced[order_columnname],
          ##order=np.unique(results_summary_reduced_accuracy_plot["scores_type"]),
         )
    g.add_legend(fontsize=12,
               ncol=3,
               bbox_to_anchor=(0.5, -0.025),
               borderaxespad=0)    
    
    return plt.gcf()

def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines)#.replace('NaN', '')

# In[ ]:





# # Prepare Results Data 

# ## Loading Files

# In[3]:

def plot_evaluation_results(timestr, parameter_grid):

    os.makedirs(os.path.dirname("./evaluation_results/" + timestr + "/"), exist_ok=True)
    
    
    #results_complete = pd.read_csv('./results_complete.csv', delimiter=';')
    #results_complete = results_complete[results_complete['i_net_nas'] == True]
    #results_complete_columns = list(results_complete.columns)

    #results_summary = pd.read_csv('./results_summary-TEST.csv', delimiter=';')
    results_summary = pd.read_csv('./results_summary' + timestr + '.csv', delimiter=';')
    #results_summary = results_summary[results_summary['i_net_nas'] == True]
    results_summary_columns = list(results_summary.columns)
    results_summary['function_family_decision_sparsity'][results_summary['data_number_of_variables'] == results_summary['function_family_decision_sparsity']] = -1


    results_summary.head(5)

    for parameter_setting in parameter_grid:

        parameter_setting = extend_inet_parameter_setting(parameter_setting)
        # ### Select columns

        # In[4]:


        colmuns_identifier = [
                          'function_family_maximum_depth',
                          'function_family_decision_sparsity', 
                          'function_family_dt_type',

                          'data_dt_type_train',
                          'data_maximum_depth_train',
                          'data_number_of_variables',
                          'data_noise_injected_level',
                          'data_function_generation_type',
                          'data_categorical_indices',

                          'data_exclude_linearly_seperable', 
                          'data_data_generation_filtering', 
                          'data_fixed_class_probability', 
                          'data_weighted_data_generation', 
                          'data_shift_distrib',

                          'lambda_net_lambda_network_layers',
                          'lambda_net_optimizer_lambda',

                          'i_net_dense_layers',
                          'i_net_dropout',
                          'i_net_learning_rate',
                          'i_net_loss',
                          'i_net_interpretation_dataset_size',
                          'i_net_function_representation_type',
                          'i_net_data_reshape_version',
                          'i_net_nas',
                          'i_net_nas_trials',

                          'evaluation_eval_data_description_eval_data_function_generation_type',
                          'evaluation_eval_data_description_eval_data_noise_injected_level',

                          'evaluation_number_of_random_evaluations_per_distribution',
                         ]


        # In[5]:


        columns_inet = []
        for column in results_summary_columns:
            if 'inet_scores' in column:
                columns_inet.append(column)
        results_summary_inet = results_summary[flatten_list([colmuns_identifier, columns_inet])]

        results_summary_inet.head(10)


        # In[6]:


        columns_inet = []
        for column in results_summary_columns:
            if 'inet_scores' in column:
                columns_inet.append(column)
        results_summary_inet = results_summary[flatten_list([colmuns_identifier, columns_inet])]

        columns_inet_rename = []
        for column in columns_inet:
            column = column.replace('inet_scores_', '')
            columns_inet_rename.append(column)

        results_summary_inet.columns = flatten_list([colmuns_identifier, columns_inet_rename])

        #results_summary_inet.insert(0, 'scores_type', 'inet_scores')
        results_summary_inet.insert(0, 'dt_type', [dt_type + str(decision_sparsity) for dt_type, decision_sparsity in zip(results_summary_inet['function_family_dt_type'].values, results_summary_inet['function_family_decision_sparsity'].values)])
        results_summary_inet.insert(0, 'technique', ['inet' for _ in range(results_summary_inet.shape[0])])


        print(results_summary_inet.shape)
        results_summary_inet.head(20)


        # In[7]:


        columns_dt_distilled = []
        for column in results_summary_columns:
            if 'dt_scores' in column:
                if 'data_random' not in column:
                    columns_dt_distilled.append(column)
        results_summary_dt_distilled = results_summary[flatten_list([colmuns_identifier, columns_dt_distilled])]

        columns_dt_distilled_rename = []
        for column in columns_dt_distilled:
            column = column.replace('dt_scores_','')
            columns_dt_distilled_rename.append(column)

        results_summary_dt_distilled.columns = flatten_list([colmuns_identifier, columns_dt_distilled_rename])

        #results_summary_dt_distilled.insert(0, 'scores_type', 'dt_scores')
        results_summary_dt_distilled.insert(0, 'dt_type', [dt_type + str(decision_sparsity) for dt_type, decision_sparsity in zip(results_summary_dt_distilled['function_family_dt_type'].values, results_summary_dt_distilled['function_family_decision_sparsity'].values)])
        results_summary_dt_distilled.insert(0, 'technique', ['distilled' for _ in range(results_summary_dt_distilled.shape[0])])


        print(results_summary_dt_distilled.shape)
        results_summary_dt_distilled.head(15)


        # In[8]:


        results_summary_inet.shape


        # In[9]:


        results_summary_dt_distilled.shape


        # In[10]:


        results_summary_reduced = pd.concat([
                                             results_summary_inet, 
                                             results_summary_dt_distilled, 
                                            ]).reset_index(drop=True)
        results_summary_reduced_columns = results_summary_reduced.columns
        results_summary_reduced.head(5)


        # ## Considered Results Specification

        # In[11]:


        config = {
            'i_net_dense_layers': [str(parameter_setting['dense_layers'])], #['[1024, 1024, 256, 2048, 2048]'], #
            'i_net_dropout': [str(parameter_setting['dropout'])], #['[0, 0, 0, 0, 0.3]'], #
            
            'i_net_interpretation_dataset_size': [parameter_setting['dataset_size']], #['[0, 0, 0, 0, 0.3]'], #            

            'i_net_loss': 'binary_crossentropy', # 'binary_crossentropy', 'soft_binary_crossentropy'

            'data_exclude_linearly_seperable': True,
            'data_data_generation_filtering':  parameter_setting['data_generation_filtering'], 
            'data_fixed_class_probability':  parameter_setting['fixed_class_probability'], 
            'data_weighted_data_generation':  parameter_setting['weighted_data_generation'], 
            'data_shift_distrib':  parameter_setting['shift_distrib'], 

            #'data_noise_injected_level': 0, 
            #'data_data_noise': 0,

            #'data_number_of_variables': [unique_value], # [10]
            'function_family_maximum_depth': [parameter_setting['maximum_depth']], # [3, 4, 5]

            'evaluation_number_of_random_evaluations_per_distribution': [parameter_setting['number_of_random_evaluations_per_distribution']],
        }


        # In[28]:


        score_names_list = ['valid_accuracy', 'valid_binary_crossentropy', 'valid_f1_score']
        valid_scores_columns = [name for name in results_summary_reduced_columns if 'valid' in name and any([score in name for score in score_names_list])]
        valid_identifier_columns = ['dt_type', 'data_number_of_variables', 'technique']
        valid_columns = flatten_list([valid_identifier_columns, valid_scores_columns])

        valid_scores_df = get_relevant_columns_by_config(config, results_summary_reduced)
        valid_scores_df = valid_scores_df[valid_columns]
        valid_scores_df = valid_scores_df.sort_values(['dt_type', 'data_number_of_variables', 'technique'], ascending=[True, True, True])
        valid_scores_df.head(100)


        # ## Real-World Dataset Selection

        # In[13]:


        distribution_list = ['uniform', 'gamma', 'beta', 'poisson', 'normal']

        if True:
            real_world_datasets = {
                                    'Adult': 28,#65,
                                    'Titanic': 9,
                                    'Absenteeism': 15,
                                    'Loan House': 16,#17,
                                    'Loan Credit': 32,
                                    'Medical Insurance': 9,#6,
                                    'Bank Marketing': 29,#17,
                                    'Cervical Cancer': 15,
                                    'Brest Cancer Wisconsin': 9,
                                    'Wisconsin Diagnostic Breast Cancer': 10,
                                    'Wisconsin Prognostic Breast Cancer': 10,
                                    'Abalone': 10,
                                    #'Car': 21,
                                   }
            real_world_datasets = dict(sorted(real_world_datasets.items(), key=lambda item: item[1]))
        elif False:
            real_world_datasets = {
                                'Adult': 28,#65,
                                'Titanic': 9,
                                    'Absenteeism': 15,
                                    'Loan House': 16,#17,
                                    #'Loan Credit': 32,
                                'Medical Insurance': 9,#6,
                                    'Bank Marketing': 29,#17,
                                'Cervical Cancer': 15,
                                    'Brest Cancer Wisconsin': 9,
                                    'Wisconsin Diagnostic Breast Cancer': 10,
                                    #'Wisconsin Prognostic Breast Cancer': 10,
                                    #'Abalone': 10,
                                    ##'Car': 21,
                                   }
            real_world_datasets = dict(sorted(real_world_datasets.items(), key=lambda item: item[1]))
        else:
            real_world_datasets = {
                                'Adult': 32,#28,
                                'Titanic': 10,#9,
                                    'Absenteeism': 15,
                                    'Loan House': 15,#16,
                                        'Loan Credit': 32,
                                'Medical Insurance': 10,#9,
                                    'Bank Marketing': 32,#29,
                                'Cervical Cancer': 15,
                                    'Brest Cancer Wisconsin': 10,#9,
                                    'Wisconsin Diagnostic Breast Cancer': 10,
                                    'Wisconsin Prognostic Breast Cancer': 10,
                                    'Abalone': 10,
                                    ##'Car': 21,
                                   }
            real_world_datasets = dict(sorted(real_world_datasets.items(), key=lambda item: item[1]))    


        # ## Restructuring & Selecting Data

        # In[14]:


        real_world_dataset_names = list(real_world_datasets.keys())
        score_names_list = ['accuracy']#['accuracy', 'binary_crossentropy', 'f1_score']
        real_world_scores_columns = [name for name in results_summary_reduced_columns if any([score in name for score in score_names_list]) and 'soft' not in name and any([dataset_name in name for dataset_name in real_world_dataset_names])]
        real_world_identifier_columns = ['dt_type', 'data_number_of_variables', 'technique']
        real_world_columns = flatten_list([real_world_identifier_columns, real_world_scores_columns])

        real_world_scores_df = get_relevant_columns_by_config(config, results_summary_reduced)
        real_world_scores_df = real_world_scores_df[real_world_columns]
        real_world_scores_df = real_world_scores_df.sort_values(['dt_type', 'data_number_of_variables', 'technique'], ascending=[True, True, True])
        #real_world_scores_df.head(20)


        # In[15]:


        columns = flatten_list(['dt_type', 'technique', 'enumerator', 'distrib', [[real_world_dataset_name + ' ' + score_name for real_world_dataset_name in real_world_datasets.keys()] for score_name in score_names_list]])
        #print(np.array(columns).shape)
        #columns = np.hstack([columns for i in range(5)])
        #print(np.array(columns).shape)


        number_of_random_evaluations_per_distribution= 0
        for column in real_world_columns:
            column_split = column.split('.')
            value = 0
            try:
                value = int(column_split[-1])
            except:
                pass
            if value > number_of_random_evaluations_per_distribution:
                number_of_random_evaluations_per_distribution = value


        empty_data_distilled = np.array([np.vstack([
                     [flatten_list(['vanilla1', 'distilled', i, distrib, [np.nan for _ in range(len(columns)-4)]]) for i in range(number_of_random_evaluations_per_distribution+1)],
                     #[flatten_list(['SDT1',  'distilled', 0, distrib, [np.nan for _ in range(len(columns)-4)]]) for i in range(number_of_random_evaluations_per_distribution+1)]
                     [flatten_list(['SDT-1',  'distilled', i, distrib, [np.nan for _ in range(len(columns)-4)]]) for i in range(number_of_random_evaluations_per_distribution+1)]
                    ]) for distrib in distribution_list] )

        empty_data_distilled = empty_data_distilled.reshape(empty_data_distilled.shape[0]*empty_data_distilled.shape[1], -1)

        empty_data_inet = np.array([
                            flatten_list(['vanilla1', 'inet', np.nan, 'inet', [np.nan for _ in range(len(columns)-4)]]),
                            #flatten_list(['SDT1', 'inet', 'inet', [np.nan for _ in range(len(columns)-4)]]),
                            flatten_list(['SDT-1', 'inet', np.nan, 'inet', [np.nan for _ in range(len(columns)-4)]]),
                          ])

        empty_data = np.vstack([empty_data_inet, empty_data_distilled])
        empty_data[:,4:] = np.nan_to_num(x=empty_data[:,4:].astype(np.float64), nan=0)

        real_world_scores_df_distrib_adjusted = pd.DataFrame(data=empty_data, columns=columns)


        for real_world_dataset_name, real_world_dataset_variables in real_world_datasets.items():
            #scores_by_variables = real_world_scores_df[real_world_scores_df['data_number_of_variables'] == real_world_dataset_variables]

            if real_world_scores_df[real_world_scores_df['data_number_of_variables'] == real_world_dataset_variables].shape[0] > 1:
                scores_by_variables = real_world_scores_df[real_world_scores_df['data_number_of_variables'] == real_world_dataset_variables]
                for i, row in real_world_scores_df_distrib_adjusted.iterrows():
                    for score_name in score_names_list:
                        relevant_column = None
                        for column_name in real_world_scores_df.columns:
                            if (row['distrib'] in column_name and 
                                real_world_dataset_name in column_name and 
                                score_name in column_name and 
                                '10000' in column_name and 
                                '100000' not in column_name and 
                                'std' not in column_name):
                                try:
                                    row['enumerator'] = int(row['enumerator'])
                                except:
                                    pass

                                if '.' + str(row['enumerator']) in column_name or ('.' not in column_name and row['enumerator'] == 0):
                                    if relevant_column is None:
                                        relevant_column = column_name
                                    else:
                                        print('DOUBLE', relevant_column, column_name)
                                        print("row['enumerator']", row['enumerator'])
                        try:
                            if row['technique'] == 'distilled':
                                scores_by_variables_selected = scores_by_variables[scores_by_variables['dt_type'] == row['dt_type']]
                                scores_by_variables_selected = scores_by_variables_selected[scores_by_variables_selected['technique'] == row['technique']]

                                row[real_world_dataset_name + ' ' + score_name] = np.max(scores_by_variables_selected[relevant_column].values)
                            else:
                                scores_by_variables_selected = scores_by_variables[scores_by_variables['dt_type'] == row['dt_type']]
                                scores_by_variables_selected = scores_by_variables_selected[scores_by_variables_selected['technique'] == row['technique']]
                                relevant_column = score_name + '_' + real_world_dataset_name + '_10000'
                                row[real_world_dataset_name + ' ' + score_name] = np.max(scores_by_variables_selected[relevant_column].values)                            

                        except:
                            pass
                            #print(scores_by_variables_selected[relevant_column])
                            #traceback.print_exc()

        real_world_scores_df_distrib_adjusted.iloc[:,4:] = real_world_scores_df_distrib_adjusted.iloc[:,4:].astype(float)
        real_world_scores_df_distrib_adjusted.head(100)           


        # In[16]:


        data = []

        for row in real_world_scores_df_distrib_adjusted.query('technique == "inet"').values:

            row_inet_identifier = row[:4]
            row_inet_mean = row[4:]
            row_inet_std = np.zeros_like(row_inet_mean)

            row_inet_mean_std = np.dstack([row_inet_mean, row_inet_std]).flatten()

            row_inet = np.hstack([row_inet_identifier, row_inet_mean_std])    
            data.append(row_inet)

        for dt_type in real_world_scores_df_distrib_adjusted['dt_type'].unique():
            rows_selected = real_world_scores_df_distrib_adjusted.query('dt_type == "' + dt_type + '" & technique == "distilled"')
            rows_selected_values = rows_selected.values[:,4:]

            row_identifier = [dt_type, 'distilled', np.nan, 'distilled']

            row_values_mean = np.mean(rows_selected_values.astype(np.float64), axis=0)
            row_values_std = np.std(rows_selected_values.astype(np.float64), axis=0)

            row_values_mean_std = np.dstack([row_values_mean, row_values_std]).flatten()
            #print(rows_selected.values)
            row_mean_std = np.hstack([row_identifier, row_values_mean_std])
            data.append(row_mean_std)

        for distrib in distribution_list:
            for dt_type in real_world_scores_df_distrib_adjusted['dt_type'].unique():
                rows_selected = real_world_scores_df_distrib_adjusted.query('dt_type == "' + dt_type + '" & distrib == "' + distrib + '"')
                rows_selected_values = rows_selected.values[:,4:]
                row_identifier = [dt_type, 'distilled', np.nan, distrib]

                row_values_mean = np.mean(rows_selected_values.astype(np.float64), axis=0)
                row_values_std = np.std(rows_selected_values.astype(np.float64), axis=0)

                row_values_mean_std = np.dstack([row_values_mean, row_values_std]).flatten()

                row_mean_std = np.hstack([row_identifier, row_values_mean_std])
                data.append(row_mean_std)

        columns = flatten_list([list(real_world_scores_df_distrib_adjusted.columns[:4]), [ [column + ' Mean', column + ' STD'] for column in real_world_scores_df_distrib_adjusted.columns[4:]]])

        real_world_scores_df_distrib_adjusted_mean_std = pd.DataFrame(data=data, columns=columns)
        real_world_scores_df_distrib_adjusted_mean_std.iloc[:,4:] = np.round(real_world_scores_df_distrib_adjusted_mean_std.iloc[:,4:].values.astype(np.float64), 4)
        real_world_scores_df_distrib_adjusted_mean_std = real_world_scores_df_distrib_adjusted_mean_std.drop('enumerator', axis=1)


        # In[17]:


        def write_latex_table_top(f):
            f.write('\\begin{table}[htb]' + '\n')
            f.write('\\centering' + '\n')
            f.write('\\resizebox{\columnwidth}{!}{' + '\n')
            f.write('%\\begin{threeparttable}' + '\n')

            f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')


        def write_latex_table_bottom(f, dt_type):
            f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')

            f.write('%\\begin{tablenotes}' + '\n')
            f.write('%\\item[a] \\footnotesize' + '\n')
            f.write('%\\item[b] \\footnotesize' + '\n')
            f.write('%\\end{tablenotes}' + '\n')
            f.write('%\\end{threeparttable}' + '\n')
            f.write('}' + '\n')
            f.write('\\caption{\\textbf{Evaluation Results ' + dt_type +'.}}' + '\n')
            f.write('\\label{tab:eval-results}' + '\n')
            f.write('\\end{table}' + '\n')



        # In[18]:


        real_world_scores_df_distrib_adjusted_mean_std_VANILLA = real_world_scores_df_distrib_adjusted_mean_std.query('dt_type == "vanilla1"')
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.drop('dt_type', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.drop('technique', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA.index = real_world_scores_df_distrib_adjusted_mean_std_VANILLA['distrib']
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.drop('distrib', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.T

        data = []
        for i in range(real_world_scores_df_distrib_adjusted_mean_std_VANILLA.shape[0]//2):
            row_mean = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.iloc[i*2].values
            row_std = real_world_scores_df_distrib_adjusted_mean_std_VANILLA.iloc[i*2+1].values

            row_values_mean_std = np.dstack([row_mean, row_std]).flatten()
            data.append(row_values_mean_std)

        columns = flatten_list([ [column + ' mean', column + ' std']  for column in real_world_scores_df_distrib_adjusted_mean_std_VANILLA.columns])

        index = real_world_scores_df_distrib_adjusted.columns[4:]
        index = [name.replace(' accuracy', '') for name in index]
        index = [index + ' (n='+  str(real_world_datasets[index]) + ')' for index in index]

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended = pd.DataFrame(data=data, columns=columns, index=index)

        summary_row = pd.Series(data=np.dstack([np.mean(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,::2].values, axis=0), np.std(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,::2].values, axis=0)]).flatten(), name='Summary', index=real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.columns)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.append(summary_row)


        ######################

        columns = flatten_list([[column, column]  for column in real_world_scores_df_distrib_adjusted_mean_std_VANILLA.columns])

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.columns = columns
        #real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.index = index


        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex * 100

        #combiner = lambda s1, s2: '$' + np.round(s1, 2).astype(str) + ' \pm ' + np.round(s2, 2).astype(str) + '$'
        #combiner = lambda s1, s2: '$' + np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: ' ' + x if float(x) < 100 else '  ' + x if float(x) < 10 else x)  + ' \pm ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: ' ' + x if float(x) < 10 else x) + '$' 
        #combiner = lambda s1, s2: '$' + np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x)  + ' \pm ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) + '$' 
        combiner = lambda s1, s2: np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).astype(str).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x) + ' $\pm$ ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) 

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.iloc[:,1::2], combiner)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.drop('distilled', axis=1)

        if number_of_random_evaluations_per_distribution == 0:
            for i, row in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.iterrows():
                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                mean_inet = float(row['inet'].split(' ')[0].split('}')[-1])
                if mean_inet == max_value:
                    row['inet'] = '\\bftab' + row['inet']
                    for best_distrib in best_distrib_key_list:
                        row[best_distrib] = '\\bftab' + row[best_distrib]
                elif mean_inet > max_value:
                    row['inet'] = '\\bftab' + row['inet']
                else:
                    for best_distrib in best_distrib_key_list:
                        row[best_distrib] = '\\bftab' + row[best_distrib]

            for i, row in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.iterrows():    
                mean_inet = float(row['inet'].split(' ')[0].split('}')[-1])
                mean_distilled = float(row['distilled'].split(' ')[0].split('}')[-1])
                if mean_inet == mean_distilled:
                    row['inet'] = '\\bftab' + row['inet']
                    row['distilled'] = '\\bftab' + row['distilled']
                elif mean_inet > mean_distilled:
                    row['inet'] = '\\bftab' + row['inet']
                else:
                    row['distilled'] = '\\bftab' + row['distilled']

                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                for best_distrib in best_distrib_key_list:
                    row[best_distrib] = '\\bftab' + row[best_distrib]  
        else:
            threshold = 0.05

            inet_scores = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.loc[:,'inet mean'].values
            distilled_max_scores = np.max(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,::2].iloc[:,2:].values, axis=1)
            best_distrib_index_by_dataset = np.argmax(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,::2].iloc[:,2:].values, axis=1)
            best_distrib_index_by_dataset = best_distrib_index_by_dataset + 2
            best_distrib_name_by_dataset_name = [[dataset_name ,real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.columns[::2][best_distrib_index].split(' ')[0]] for dataset_name, best_distrib_index in zip(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.index, best_distrib_index_by_dataset)][:-1]

            ttest_by_dataset_VANILLA = []
            for dataset_name, best_distrib_name in best_distrib_name_by_dataset_name:
                #print(dataset_name, best_distrib_name)
                considered_columns_distilled = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"' + '&' + 'distrib == "' + best_distrib_name + '"')
                considered_results_distilled = considered_columns_distilled.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values

                considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"')
                considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]

                ttest_statistics, ttest_p_value  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet)

                identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else best_distrib_name

                ttest_by_dataset_VANILLA.append([dataset_name, identifier_best, ttest_p_value])

            for dataset_name, identifier_best, p_value  in ttest_by_dataset_VANILLA:
                if p_value < threshold:
                    real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.loc[dataset_name, identifier_best] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.loc[dataset_name, identifier_best]


            ttest_by_dataset_VANILLA = []
            for dataset_name in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.index[:-1]:
                considered_results_distilled = []
                for distrib in distribution_list:
                    considered_columns_distilled_distrib = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"' + '&' + 'distrib == "' + distrib + '"')
                    considered_results_distilled_distrib = considered_columns_distilled_distrib.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values
                    considered_results_distilled.append(considered_results_distilled_distrib)
                considered_results_distilled = np.hstack(considered_results_distilled)

                considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"')
                considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]    

                ttest_statistics, ttest_p_value  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet)

                identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else 'distilled'

                ttest_by_dataset_VANILLA.append([dataset_name, identifier_best, ttest_p_value, ('mean distilled', np.mean(considered_results_distilled)), ('std distilled', np.std(considered_results_distilled))])    

            for dataset_name, identifier_best, p_value, mean_distilled, std_distilled in ttest_by_dataset_VANILLA:
                if p_value < threshold:
                    real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.loc[dataset_name, identifier_best] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.loc[dataset_name, identifier_best]

            for i, row in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.iterrows():    
                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                for best_distrib in best_distrib_key_list:
                    row[best_distrib] = '\\bftab' + row[best_distrib] 



        with open("./evaluation_results/" + timestr +"/latex_table_vanilla_with_distilled_mean.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla')
        with open("./evaluation_results/" + timestr +"/latex_table_vanilla.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla')


        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended).iloc[:,4:]

        best_distrib = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib.columns[np.argmax(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib.loc['Summary'].values[::2])*2]
        best_distrib = best_distrib.split(' ')[0]

        best_distrib_columns = []
        for column in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib.columns:
            if best_distrib in column:
                best_distrib_columns.append(column)

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib = pd.concat([real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,:2], real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.loc[:,best_distrib_columns]], axis=1)

        #display(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib.head(100))
        #display(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib.head(100))


        combiner = lambda s1, s2: np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).astype(str).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x) + ' $\pm$ ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) 

        columns = [column.split(' ')[0]  for column in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib.columns]
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.columns = columns

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex * 100
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.iloc[:,1::2], combiner)


        columns = [column.split(' ')[0]  for column in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib.columns]
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib)
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.columns = columns

        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex * 100
        real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.iloc[:,1::2], combiner)



        threshold = 0.05

        ttest_by_dataset_VANILLA_less = []
        ttest_by_dataset_VANILLA_greater = []

        for dataset_name in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib.index[:-1]:
            considered_columns_distilled = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"' + '&' + 'distrib == "' + best_distrib + '"')
            considered_results_distilled = considered_columns_distilled.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values

            #display(considered_columns_distilled)
            #display(considered_results_distilled)

            considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'vanilla1' + '"')
            considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]    

            #display(considered_column_inet)
            #display(considered_result_inet)

            ttest_statistics_less, ttest_p_value_less  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet, alternative='less')
            ttest_statistics_greater, ttest_p_value_greater  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet, alternative='greater')


            identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else best_distrib

            ttest_by_dataset_VANILLA_less.append([dataset_name, identifier_best, ttest_p_value_less, ('mean ' + best_distrib, np.mean(considered_results_distilled)), ('std ' + best_distrib, np.std(considered_results_distilled))])    
            ttest_by_dataset_VANILLA_greater.append([dataset_name, identifier_best, ttest_p_value_greater, ('mean ' + best_distrib, np.mean(considered_results_distilled)), ('std ' + best_distrib, np.std(considered_results_distilled))])    

        for ttest_less, ttest_greater in zip(ttest_by_dataset_VANILLA_less, ttest_by_dataset_VANILLA_greater):
            (dataset_name_less, identifier_best_less, p_value_less, mean_distilled_less, std_distilled_less) = ttest_less
            (dataset_name_greater, identifier_best_greater, p_value_greater, mean_distilled_greater, std_distilled_greater) = ttest_greater
            #print(p_value_less, p_value_greater)
            #print(mean_distilled_less, std_distilled_less)

            if p_value_greater < threshold:
                real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.loc[dataset_name_greater, best_distrib] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.loc[dataset_name_greater, best_distrib]
            if p_value_less < threshold:
                real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.loc[dataset_name_less, 'inet'] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.loc[dataset_name_less, 'inet']

        for i, row in real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.iterrows():    
            distrib_mean = {}
            for distrib in distribution_list:
                distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
            #best_distrib = max(distrib_mean, key=distrib_mean.get)
            max_value = max(distrib_mean.values())
            best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

            for best_distrib in best_distrib_key_list:
                row[best_distrib] = '\\bftab' + row[best_distrib] 




        with open("./evaluation_results/latex_table_vanilla_split.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla distrib comparison')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla distrib inet')    

        #display(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex)
        #display(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex)


            

        # In[19]:


        real_world_scores_df_distrib_adjusted_mean_std_SDT = real_world_scores_df_distrib_adjusted_mean_std.query('dt_type == "SDT-1"')
        real_world_scores_df_distrib_adjusted_mean_std_SDT = real_world_scores_df_distrib_adjusted_mean_std_SDT.drop('dt_type', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_SDT = real_world_scores_df_distrib_adjusted_mean_std_SDT.drop('technique', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_SDT.index = real_world_scores_df_distrib_adjusted_mean_std_SDT['distrib']
        real_world_scores_df_distrib_adjusted_mean_std_SDT = real_world_scores_df_distrib_adjusted_mean_std_SDT.drop('distrib', axis=1)
        real_world_scores_df_distrib_adjusted_mean_std_SDT = real_world_scores_df_distrib_adjusted_mean_std_SDT.T

        data = []
        for i in range(real_world_scores_df_distrib_adjusted_mean_std_SDT.shape[0]//2):
            row_mean = real_world_scores_df_distrib_adjusted_mean_std_SDT.iloc[i*2].values
            row_std = real_world_scores_df_distrib_adjusted_mean_std_SDT.iloc[i*2+1].values

            row_values_mean_std = np.dstack([row_mean, row_std]).flatten()
            data.append(row_values_mean_std)

        columns = flatten_list([ [column + ' mean', column + ' std']  for column in real_world_scores_df_distrib_adjusted_mean_std_SDT.columns])

        index = real_world_scores_df_distrib_adjusted.columns[4:]
        index = [name.replace(' accuracy', '') for name in index]
        index = [index + ' (n='+  str(real_world_datasets[index]) + ')' for index in index]

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended = pd.DataFrame(data=data, columns=columns, index=index)

        summary_row = pd.Series(data=np.dstack([np.mean(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,::2].values, axis=0), np.std(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,::2].values, axis=0)]).flatten(), name='Summary', index=real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.columns)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.append(summary_row)


        ######################

        columns = flatten_list([[column, column]  for column in real_world_scores_df_distrib_adjusted_mean_std_SDT.columns])

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.columns = columns


        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex * 100

        #combiner = lambda s1, s2: '$' + np.round(s1, 2).astype(str) + ' \pm ' + np.round(s2, 2).astype(str) + '$'
        #combiner = lambda s1, s2: '$' + np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: ' ' + x if float(x) < 100 else '  ' + x if float(x) < 10 else x)  + ' \pm ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: ' ' + x if float(x) < 10 else x) + '$' 
        #combiner = lambda s1, s2: '$' + np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x)  + ' \pm ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) + '$' 
        combiner = lambda s1, s2: np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).astype(str).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x) + ' $\pm$ ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) 

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.iloc[:,1::2], combiner)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.drop('distilled', axis=1)

        if number_of_random_evaluations_per_distribution == 0:
            for i, row in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.iterrows():
                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                mean_inet = float(row['inet'].split(' ')[0].split('}')[-1])
                if mean_inet == max_value:
                    row['inet'] = '\\bftab' + row['inet']
                    for best_distrib in best_distrib_key_list:
                        row[best_distrib] = '\\bftab' + row[best_distrib]
                elif mean_inet > max_value:
                    row['inet'] = '\\bftab' + row['inet']
                else:
                    for best_distrib in best_distrib_key_list:
                        row[best_distrib] = '\\bftab' + row[best_distrib]

            for i, row in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.iterrows():    
                mean_inet = float(row['inet'].split(' ')[0].split('}')[-1])
                mean_distilled = float(row['distilled'].split(' ')[0].split('}')[-1])
                if mean_inet == mean_distilled:
                    row['inet'] = '\\bftab' + row['inet']
                    row['distilled'] = '\\bftab' + row['distilled']
                elif mean_inet > mean_distilled:
                    row['inet'] = '\\bftab' + row['inet']
                else:
                    row['distilled'] = '\\bftab' + row['distilled']

                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                for best_distrib in best_distrib_key_list:
                    row[best_distrib] = '\\bftab' + row[best_distrib] 
        else:
            threshold = 0.05

            inet_scores = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.loc[:,'inet mean'].values
            distilled_max_scores = np.max(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,::2].iloc[:,2:].values, axis=1)
            best_distrib_index_by_dataset = np.argmax(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,::2].iloc[:,2:].values, axis=1)
            best_distrib_index_by_dataset = best_distrib_index_by_dataset + 2
            best_distrib_name_by_dataset_name = [[dataset_name ,real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.columns[::2][best_distrib_index].split(' ')[0]] for dataset_name, best_distrib_index in zip(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.index, best_distrib_index_by_dataset)][:-1]

            ttest_by_dataset_SDT = []
            for dataset_name, best_distrib_name in best_distrib_name_by_dataset_name:
                #print(dataset_name, best_distrib_name)
                considered_columns_distilled = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"' + '&' + 'distrib == "' + best_distrib_name + '"')
                considered_results_distilled = considered_columns_distilled.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values

                considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"')
                considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]

                ttest_statistics, ttest_p_value  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet)

                identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else best_distrib_name

                ttest_by_dataset_SDT.append([dataset_name, identifier_best, ttest_p_value])

            for dataset_name, identifier_best, p_value  in ttest_by_dataset_SDT:
                if p_value < threshold:
                    real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.loc[dataset_name, identifier_best] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.loc[dataset_name, identifier_best]


            ttest_by_dataset_SDT = []
            for dataset_name in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.index[:-1]:
                considered_results_distilled = []
                for distrib in distribution_list:
                    considered_columns_distilled_distrib = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"' + '&' + 'distrib == "' + distrib + '"')
                    considered_results_distilled_distrib = considered_columns_distilled_distrib.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values
                    considered_results_distilled.append(considered_results_distilled_distrib)
                considered_results_distilled = np.hstack(considered_results_distilled)

                considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"')
                considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]    

                ttest_statistics, ttest_p_value  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet)

                identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else 'distilled'

                ttest_by_dataset_SDT.append([dataset_name, identifier_best, ttest_p_value, ('mean distilled', np.mean(considered_results_distilled)), ('std distilled', np.std(considered_results_distilled))])    

            for dataset_name, identifier_best, p_value, mean_distilled, std_distilled in ttest_by_dataset_SDT:
                if p_value < threshold:
                    real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.loc[dataset_name, identifier_best] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.loc[dataset_name, identifier_best]

            for i, row in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.iterrows():    
                distrib_mean = {}
                for distrib in distribution_list:
                    distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
                #best_distrib = max(distrib_mean, key=distrib_mean.get)
                max_value = max(distrib_mean.values())
                best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

                for best_distrib in best_distrib_key_list:
                    row[best_distrib] = '\\bftab' + row[best_distrib] 

        with open("./evaluation_results/" + timestr +"/latex_table_sdt_with_distilled_mean.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT')

        with open("./evaluation_results/" + timestr +"/latex_table_sdt.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT')


        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended).iloc[:,4:]

        best_distrib = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib.columns[np.argmax(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib.loc['Summary'].values[::2])*2]
        best_distrib = best_distrib.split(' ')[0]

        best_distrib_columns = []
        for column in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib.columns:
            if best_distrib in column:
                best_distrib_columns.append(column)

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib = pd.concat([real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,:2], real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.loc[:,best_distrib_columns]], axis=1)

        #display(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib.head(100))
        #display(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib.head(100))


        combiner = lambda s1, s2: np.round(s1, 2).apply(lambda x: '{:.2f}'.format(x)).astype(str).apply(lambda x: '\phantom{0}' + x if float(x) < 100 else '\phantom{00}' + x if float(x) < 10 else x) + ' $\pm$ ' + np.round(s2, 2).apply(lambda x: '{:.2f}'.format(x)).apply(lambda x: '\phantom{0}' + x if float(x) < 10 else x) 

        columns = [column.split(' ')[0]  for column in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib.columns]
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.columns = columns

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex * 100
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.iloc[:,1::2], combiner)


        columns = [column.split(' ')[0]  for column in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib.columns]
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex = deepcopy(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib)
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.columns = columns

        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex * 100
        real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.iloc[:,::2].combine(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.iloc[:,1::2], combiner)



        threshold = 0.05

        ttest_by_dataset_SDT_less = []
        ttest_by_dataset_SDT_greater = []
        for dataset_name in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib.index[:-1]:
            considered_columns_distilled = real_world_scores_df_distrib_adjusted.query('technique == "' + 'distilled' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"' + '&' + 'distrib == "' + best_distrib + '"')
            considered_results_distilled = considered_columns_distilled.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values


            considered_column_inet = real_world_scores_df_distrib_adjusted.query('technique == "' + 'inet' + '"' + '&' + 'dt_type == "' + 'SDT-1' + '"')
            considered_result_inet = considered_column_inet.loc[:, ' '.join(dataset_name.split(' ')[:-1]) + ' accuracy'].values[0]    

            ttest_statistics_less, ttest_p_value_less  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet, alternative='less')
            ttest_statistics_greater, ttest_p_value_greater  = scipy.stats.ttest_1samp(considered_results_distilled, considered_result_inet, alternative='greater')


            identifier_best = 'inet' if considered_result_inet > np.mean(considered_results_distilled) else best_distrib

            ttest_by_dataset_SDT_less.append([dataset_name, identifier_best, ttest_p_value_less, ('mean ' + best_distrib, np.mean(considered_results_distilled)), ('std ' + best_distrib, np.std(considered_results_distilled))])    
            ttest_by_dataset_SDT_greater.append([dataset_name, identifier_best, ttest_p_value_greater, ('mean ' + best_distrib, np.mean(considered_results_distilled)), ('std ' + best_distrib, np.std(considered_results_distilled))])    

        for ttest_less, ttest_greater in zip(ttest_by_dataset_SDT_less, ttest_by_dataset_SDT_greater):
            (dataset_name_less, identifier_best_less, p_value_less, mean_distilled_less, std_distilled_less) = ttest_less
            (dataset_name_greater, identifier_best_greater, p_value_greater, mean_distilled_greater, std_distilled_greater) = ttest_greater
            #print(p_value_less, p_value_greater)
            #print(mean_distilled_less, std_distilled_less)

            if p_value_greater < threshold:
                real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.loc[dataset_name_greater, best_distrib] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.loc[dataset_name_greater, best_distrib]
            if p_value_less < threshold:
                real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.loc[dataset_name_less, 'inet'] = '\\bftab' + real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.loc[dataset_name_less, 'inet']

        for i, row in real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.iterrows():    
            distrib_mean = {}
            for distrib in distribution_list:
                distrib_mean[distrib] = float(row[distrib].split(' ')[0].split('}')[-1])
            #best_distrib = max(distrib_mean, key=distrib_mean.get)
            max_value = max(distrib_mean.values())
            best_distrib_key_list = [key for key, value in distrib_mean.items() if value == max_value]

            for best_distrib in best_distrib_key_list:
                row[best_distrib] = '\\bftab' + row[best_distrib] 




        with open("./evaluation_results/latex_table_sdt_split.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib comparison')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib inet')    




        with open("./evaluation_results/latex_table_sdt_split.tex", "w") as f:
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib comparison')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib inet')                


                # In[20]:


        with open("./evaluation_results/" + timestr +"/latex_tables_complete.tex", "w") as f:
            f.write('\\newpage \n')
            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex_with_distilled_mean.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex_with_distilled_mean.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT')    
            f.write('\\newpage \n')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT')
            f.write('\\newpage \n')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_only_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla distrib comparison')

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_only_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib comparison')   

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended_with_best_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'vanilla distrib inet')     

            write_latex_table_top(f)
            f.write(add_hline(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended_with_best_distrib_latex.to_latex(index=True, bold_rows=True, escape=False), 1))
            write_latex_table_bottom(f, 'SDT distrib inet')        

            f.write('\\newpage \n')



        print('Vanilla DT Results:\n')
        print(tabulate(real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.iloc[:,::2].round(decimals=2), headers='keys', tablefmt='psql'))


        print('SDT Results:\n')
        print(tabulate(real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.iloc[:,::2].round(decimals=2), headers='keys', tablefmt='psql'))
        # In[21]:


        real_world_scores_df_distrib_adjusted_plotting = real_world_scores_df_distrib_adjusted.melt(id_vars=["dt_type", "technique", "enumerator", "distrib"], 
                                                                                    var_name="score_name", 
                                                                                    value_name="value")

        real_world_scores_df_distrib_adjusted_plotting = real_world_scores_df_distrib_adjusted_plotting[real_world_scores_df_distrib_adjusted_plotting['value'].notna()]
        real_world_scores_df_distrib_adjusted_plotting['distrib'] = pd.Categorical(real_world_scores_df_distrib_adjusted_plotting['distrib'], flatten_list(['inet', distribution_list]))
        real_world_scores_df_distrib_adjusted_plotting.sort_values('distrib')

        #real_world_scores_df_distrib_adjusted_plotting.head(10)


        # # Tables

        # In[22]:

        inet_scores = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.loc[:,'inet mean']
        distilled_scores = real_world_scores_df_distrib_adjusted_mean_std_VANILLA_extended.loc[:,'distilled mean']

        ttest_equal_var = scipy.stats.ttest_ind(inet_scores, distilled_scores)
        print(ttest_equal_var)

        ttest = scipy.stats.ttest_ind(inet_scores, distilled_scores, equal_var=False)
        print(ttest)

        #ttest_summary, ttest_results = rp.ttest(inet_scores, distilled_scores)
        #print(ttest_results)
        #print(ttest_summary)


        # In[23]:

        inet_scores = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.loc[:,'inet mean']
        distilled_scores = real_world_scores_df_distrib_adjusted_mean_std_SDT_extended.loc[:,'distilled mean']

        ttest_equal_var = scipy.stats.ttest_ind(inet_scores, distilled_scores)
        print(ttest_equal_var)

        ttest = scipy.stats.ttest_ind(inet_scores, distilled_scores, equal_var=False)
        print(ttest)

        #ttest_summary, ttest_results = rp.ttest(inet_scores, distilled_scores)
        #print(ttest_results)
        #print(ttest_summary)


        # # Plots

        # In[24]:


        plot = plot_results(data_reduced=real_world_scores_df_distrib_adjusted_plotting, 
                            col = 'score_name', 
                            x = 'dt_type', 
                            y = 'value', 
                            hue = 'technique', 
                            plot_type = sns.barplot, 
                            aspect = 2.5, 
                            col_wrap = 3)

        plt.savefig("./evaluation_results/" + timestr +"/real_workd_complete_by_technique_barplot.pdf", bbox_inches = 'tight', pad_inches = 0)


        # In[25]:


        plot = plot_results(data_reduced=real_world_scores_df_distrib_adjusted_plotting, 
                            col = 'score_name', 
                            x = 'dt_type', 
                            y = 'value', 
                            hue = 'technique', 
                            plot_type = sns.boxplot, 
                            aspect = 2.5, 
                            col_wrap = 3)

        plt.savefig("./evaluation_results/" + timestr +"/real_workd_complete_by_technique_boxplot.pdf", bbox_inches = 'tight', pad_inches = 0)


        # In[26]:


        plot = plot_results(data_reduced=real_world_scores_df_distrib_adjusted_plotting, 
                            col = 'score_name', 
                            x = 'dt_type', 
                            y = 'value', 
                            hue = 'distrib', 
                            plot_type = sns.barplot, 
                            aspect = 2.5, 
                            col_wrap = 3)

        plt.savefig("./evaluation_results/" + timestr +"/real_workd_complete_by_technique_by_distrib_barplot.pdf", bbox_inches = 'tight', pad_inches = 0)


        # In[27]:


        plot = plot_results(data_reduced=real_world_scores_df_distrib_adjusted_plotting, 
                            col = 'score_name', 
                            x = 'dt_type', 
                            y = 'value', 
                            hue = 'distrib', 
                            plot_type = sns.boxplot, 
                            aspect = 2.5, 
                            col_wrap = 3)

        plt.savefig("./evaluation_results/" + timestr +"/real_workd_complete_by_technique_by_distrib_boxplot.pdf", bbox_inches = 'tight', pad_inches = 0)


        # In[ ]:





        # In[ ]:





        # In[ ]:





        # In[ ]:





        # In[ ]:





        # In[ ]:





        # In[ ]:





        # In[ ]:




