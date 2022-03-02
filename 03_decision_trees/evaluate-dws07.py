from interpretation_net_evaluate import run_evaluation
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from contextlib import redirect_stdout



def main(): 

        
    with open('./temp07.txt', 'a+') as f:
        with redirect_stdout(f):  
            evaluation_grid = {
                'depth_experiment': [5],

                'dt_setting_experiment': [1,2,3],#[1,2,3],
                'function_generation_type_experiment': ['make_classification_trained'], #'random_decision_tree_trained'
                #'dt_type_experiment': [],#'vanilla'
                #'decision_sparsity_experiment': [],#1
                'variable_number_experiment': [
                                               6, 
                                               9, 
                                               10, 
                                               #15, 
                                               17, 
                                               21,
                                               #32,
                                               #65
                                              ],

                'i_net_structure_experiment': [[1024, 1024, 256, 2048, 2048]],
                'i_net_dropout_experiment': [[0.3, 0.3, 0.3, 0.3, 0.3]],
                'i_net_learning_rate_experiment': [0.0001],
                'i_net_loss_experiment': ['binary_crossentropy'],
            }


            parameter_grid = ParameterGrid(evaluation_grid)

            for parameter_setting in parameter_grid:
                print(parameter_setting)

            Parallel(n_jobs=8, backend='loky', verbose=1000)(delayed(run_evaluation)(parameter_setting) for parameter_setting in parameter_grid)  

            #for parameter_setting in parameter_grid:
                #run_evaluation.remote(parameter_setting)

    
if __name__ == "__main__": 
	main()
