



'''
config = {
    'data': {
        'degree': 3,
        'number_of_variables': 4,
        'sparsity': None,
        'x_max': 1,
        'x_min': -1,
        'x_step': 0.01,
        'a_max': 10,
        'a_min': -10,
        'a_step': 0.001,
        'lambda_dataset_size': 1000,
        'interpretation_dataset_load_size': 100000,
        'interpretation_dataset_size': 10000,
    },
    'lambda_net': {
        'epochs': 200,
        'batch_size': 64,
        'lambda_network_layers': [5*sparsity],
        'optimizer': 'adam',
        'number_of_lambda_weights': None,
    },
    'i_net': {
        
    },
    'evaluation': {
        
    }
}


config['data']['sparsity'] = nCr(config['data']['n']+config['data']['d'], config['data']['d'])

layers_with_input_output = list(flatten([[config['data']['n']], config['data']['lambda_network_layers'], [1]]))
number_of_lambda_weights = 0
for i in range(len(layers_with_input_output)-1):
    number_of_lambda_weights += (layers_with_input_output[i]+1)*layers_with_input_output[i+1]
    
config['lambda_net']['number_of_lambda_weights'] = nCr(config['data']['n']+config['data']['d'], config['data']['d'])

'''