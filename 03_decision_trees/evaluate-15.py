from interpretation_net_evaluate import run_evaluation
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from contextlib import redirect_stdout
import socket
from tqdm import tqdm
import time

                #'dense_layers': [1024, 1024, 256, 2048, 2048],
                #'dense_layers': [1792, 512, 512],
                #'dense_layers': [704], #SDT-1 n=15
                #'dense_layers': [512, 512, 512], #vanilla n=15 BEST for 15 and 32 on real-world
                #'dense_layers': [512, 512, 512], #SDT-1 n=32
                #'dense_layers': [512, 512, 512], #vanilla n=32        

                #'dropout': [0, 0, 0, 0, 0.3],#[0.3, 0.3, 0.3, 0.3, 0.3],
                #'dropout': [0, 0, 0.5],
                #'dropout': [0], #SDT-1 n=15
                #'dropout': [0.3, 0, 0], #vanilla n=15 BEST for 15 and 32 on real-world
                #'dropout': [0, 0, 0], #SDT-1 n=32
                #'dropout': [0.5, 0, 0], #vanilla n=32

                #'hidden_activation': 'relu',
                #'hidden_activation': 'sigmoid',
                #'hidden_activation': ['sigmoid'], #SDT-1 n=15
                #'hidden_activation': ['sigmoid', 'tanh', 'sigmoid'], #vanilla n=15 BEST for 15 and 32 on real-world
                #'hidden_activation': ['sigmoid', 'tanh', 'tanh'], #SDT-1 n=32
                #'hidden_activation': ['sigmoid', 'tanh', 'sigmoid'], #vanilla n=32

                #'optimizer': 'rmsprop', 
                #'optimizer': 'adam', 
                #'optimizer': 'adam', #SDT-1 n=15
                #'optimizer': 'adam', #vanilla n=15  BEST for 15 and 32 on real-world
                #'optimizer': 'adam', #SDT-1 n=32
                #'optimizer': 'adam', #vanilla n=32

                #'learning_rate': 0.001,
                #'learning_rate': 0.001,
                #'learning_rate': 0.001, #SDT-1 n=15
                #'learning_rate': 0.001, #vanilla n=15 BEST for 15 and 32 on real-world
                #'learning_rate': 0.001, #SDT-1 n=32
                #'learning_rate': 0.001, #vanilla n=32

                
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
                
                'n_jobs': [5],   
                'force_evaluate_real_world': [True],
                'number_of_random_evaluations_per_distribution': [1],
                
                'dt_setting': [1, 2], # 1=vanilla; 2=SDT ------- 'dt_type', 'decision_sparsity', 'function_representation_type'                
                'inet_setting': [1], 
                'dataset_size': [10000], #50000
                
                'maximum_depth': [3],
                'number_of_variables':[
                                               #9, 
                                               #10, 
                                               15, 
                                               #16, 
                                               #28,
                                               #29,
                                               #32,
                                              ],      
                
                
                'function_generation_type': ['distribution'],# 'make_classification_distribution', 'make_classification_distribution_trained', 'distribution', 'distribution_trained', 'make_classification', 'make_classification_trained', 'random_decision_tree', 'random_decision_tree_trained'
                
                'distribution_list': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],#['uniform', 'normal', 'gamma', 'exponential', 'beta', 'binomial', 'poisson'], 
                'distribution_list_eval': [['uniform']],#[['uniform', 'normal', 'gamma', 'beta', 'poisson']],
                'distrib_param_max': [5],
                
                'data_generation_filtering':  [True, False], 
                'fixed_class_probability':  [True, False], 
                'weighted_data_generation':  [True, False], 
                'shift_distrib':  [True, False],                 
                                
                          
            }


            parameter_grid = ParameterGrid(evaluation_grid)

            for parameter_setting in parameter_grid:
                print(parameter_setting)

            timestr = time.strftime("%Y%m%d-%H%M%S")
                
            Parallel(n_jobs=6, backend='loky', verbose=10000)(delayed(run_evaluation)(enumerator, timestr, parameter_setting) for enumerator, parameter_setting in enumerate(parameter_grid))

            #for parameter_setting in parameter_grid:
                #run_evaluation.remote(parameter_setting)
            print('FINISHED')    
            
if __name__ == "__main__": 
      
	main()
