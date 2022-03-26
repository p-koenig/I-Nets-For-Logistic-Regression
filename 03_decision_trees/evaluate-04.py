from interpretation_net_evaluate import run_evaluation
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from contextlib import redirect_stdout
import socket

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

def main(): 

        
    with open('./temp' + socket.gethostname() + '.txt', 'a+') as f:
        with redirect_stdout(f):  
            evaluation_grid = {
                
                'dt_setting': [1, 2], # 1=vanilla; 2=SDT ------- 'dt_type', 'decision_sparsity', 'function_representation_type'
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
                
                'dataset_size': [10000],
                
                'function_generation_type': ['distribution'],# 'make_classification_distribution', 'make_classification_distribution_trained', 'distribution', 'distribution_trained', 'make_classification', 'make_classification_trained', 'random_decision_tree', 'random_decision_tree_trained'
                'distribution_list': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],#['uniform', 'normal', 'gamma', 'exponential', 'beta', 'binomial', 'poisson'], 
                'distribution_list_eval': [['uniform', 'normal', 'gamma', 'beta', 'poisson']],
                'distrib_param_max': [5],
                
                'inet_setting': [2]           
                
            }


            parameter_grid = ParameterGrid(evaluation_grid)

            for parameter_setting in parameter_grid:
                print(parameter_setting)

            Parallel(n_jobs=7, backend='loky', verbose=1000)(delayed(run_evaluation)(parameter_setting) for parameter_setting in parameter_grid)  

            #for parameter_setting in parameter_grid:
                #run_evaluation.remote(parameter_setting)

    
if __name__ == "__main__": 
	main()