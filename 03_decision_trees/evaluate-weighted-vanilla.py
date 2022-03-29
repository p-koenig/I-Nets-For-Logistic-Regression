from interpretation_net_evaluate import run_evaluation
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from contextlib import redirect_stdout
import socket
from tqdm import tqdm
import time
                
def sleep_minutes_function(minutes):   
    for _ in tqdm(range(minutes)):
        time.sleep(60)

                
def main(): 
    
    sleep_minutes = 0

        
    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open('./temp-' + socket.gethostname() + '_' + timestr + '.txt', 'a+') as f:
        with redirect_stdout(f):  
            
            if sleep_minutes > 0:
                sleep_minutes_function(sleep_minutes)
        
            evaluation_grid = {
                
                'n_jobs': [7],   
                'force_evaluate_real_world': [True],
                'number_of_random_evaluations_per_distribution': [10],
                
                'dt_setting': [1], # 1=vanilla; 2=SDT ------- 'dt_type', 'decision_sparsity', 'function_representation_type'                
                'inet_setting': [6], 
                'dataset_size': [10000], #50000
                
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
                
                'distribution_list': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],#['uniform', 'normal', 'gamma', 'exponential', 'beta', 'binomial', 'poisson'], 
                'distribution_list_eval': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],
                'distrib_param_max': [5],
                
                'data_generation_filtering':  [False], 
                'fixed_class_probability':  [True], 
                'weighted_data_generation':  [True], 
                'shift_distrib':  [False],                 
                                
                          
            }


            parameter_grid = ParameterGrid(evaluation_grid)

            for parameter_setting in parameter_grid:
                print(parameter_setting)

            timestr = time.strftime("%Y%m%d-%H%M%S")
                
            Parallel(n_jobs=7, backend='loky', verbose=10000)(delayed(run_evaluation)(enumerator, timestr, parameter_setting) for enumerator, parameter_setting in enumerate(parameter_grid))

            #for parameter_setting in parameter_grid:
                #run_evaluation.remote(parameter_setting)
            print('FINISHED')    
            
if __name__ == "__main__": 

        
	main()
