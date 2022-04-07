from python_scripts.interpretation_net_evaluate import run_evaluation, extend_inet_parameter_setting
from python_scripts.plot_results import plot_evaluation_results

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from contextlib import redirect_stdout
import socket
from tqdm import tqdm
import time

import os
                
def sleep_minutes_function(minutes):   
    for _ in tqdm(range(minutes)):
        time.sleep(60)

                
def main(): 
    
    sleep_minutes = 0

        
    timestr = time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(os.path.dirname('./running_evaluations/'), exist_ok=True)
    with open('./running_evaluations/temp-' + socket.gethostname() + '_' + timestr + '.txt', 'a+') as f:
        with redirect_stdout(f):  
            
            if sleep_minutes > 0:
                sleep_minutes_function(sleep_minutes)
        
            evaluation_grid = {
                
                'n_jobs': [6],   
                'force_evaluate_real_world': [False],
                'number_of_random_evaluations_per_distribution': [10],
                
                'dt_setting': [1, 2], # 1=vanilla; 2=SDT ------- 'dt_type', 'decision_sparsity', 'function_representation_type'                
                'inet_setting': [1], 
                'dataset_size': [10000], #10000, 50000
                
                'maximum_depth': [3],
                'number_of_variables':[
                                               9, 
                                               10, 
                                               15, 
                                               16, 
                                               28,
                                               29,
                                               32,
                                              ],      
                
                
                'function_generation_type': ['distribution'],# 'make_classification_distribution', 'make_classification_distribution_trained', 'distribution', 'distribution_trained', 'make_classification', 'make_classification_trained', 'random_decision_tree', 'random_decision_tree_trained'
                
                'distrib_by_feature': [True],
                'distribution_list': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],#[['uniform', 'normal', 'gamma', 'beta', 'poisson']],#['uniform', 'normal', 'gamma', 'exponential', 'beta', 'binomial', 'poisson'], 
                'distribution_list_eval': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],#[['uniform', 'normal', 'gamma', 'beta', 'poisson']],
                'distrib_param_max': [5],
                
                'data_generation_filtering':  [True], 
                'fixed_class_probability':  [False], 
                'weighted_data_generation':  [False], 
                'shift_distrib':  [False],                 
                                
                          
            }


            parameter_grid = list(ParameterGrid(evaluation_grid))

            for parameter_setting in parameter_grid:
                print(parameter_setting)
                
            print('Possible Evaluations: ', len(parameter_grid))

            timestr = time.strftime("%Y%m%d-%H%M%S")
                
            Parallel(n_jobs=5, backend='loky', verbose=10000)(delayed(run_evaluation)(enumerator, timestr, parameter_setting) for enumerator, parameter_setting in enumerate(parameter_grid))

            #for parameter_setting in parameter_grid:
                #run_evaluation.remote(parameter_setting)
            print('COMPUTATION FINISHED')    
                        
            for i in range(len(parameter_grid)):
                del parameter_grid[i]['number_of_variables']
                del parameter_grid[i]['dt_setting']
                #print(parameter_grid[i])
            
            parameter_grid = [ii for n,ii in enumerate(parameter_grid) if ii not in parameter_grid[:n]]
            #parameter_grid = list(set(parameter_grid))
            
            print('Possible Evaluations Types: ', len(parameter_grid)) 
                
            #inet_setting_parameters = [extend_inet_parameter_setting({'inet_setting': inet_setting_value}) for inet_setting_value in evaluation_grid['inet_setting']]
                        
            #structure_list = [inet_setting['dense_layers'] for inet_setting in inet_setting_parameters]
            #dropout_list = [inet_setting['dropout'] for inet_setting in inet_setting_parameters]
                
            plot_evaluation_results(timestr=timestr, parameter_grid=parameter_grid)            
            
            
            
if __name__ == "__main__": 

        
	main()
