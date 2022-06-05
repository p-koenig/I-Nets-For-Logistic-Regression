### LOGISTIC REGRESSION

def data_path_LR(config):
    return  'data_LR/nda' + str(config['data']['n_datasets']) + '_nsa' + str(config['data']['n_samples']) + '_nfe' + str(config['data']['n_features']) + '_nin' + str(config['data']['n_informative']) + '_nta' + str(config['data']['n_targets']) + '_ncc' + str(config['data']['n_clusters_per_class']) + '_sep' + str(config['data']['class_sep']) + '_noi' + str(config['data']['noise']) + '_shu' + str(config['data']['shuffle']) + '_ran' + str(config['data']['random_state']) 

def lambda_path_LR(config):
    return  data_path_LR(config) + '/' + 'tsi' + str(config['lambda']['data_prep']['train_test_val_split']['test_size']) + '_vsi' + str(config['lambda']['data_prep']['train_test_val_split']['val_size']) + '_ran' + str(config['lambda']['data_prep']['train_test_val_split']['random_state']) + '_shu' + str(config['lambda']['data_prep']['train_test_val_split']['shuffle']) + '_str' + str(config['lambda']['data_prep']['train_test_val_split']['stratify']) + '_bat' + str(config['lambda']['model_fit']['batch_size']) + '_epo' + str(config['lambda']['model_fit']['epochs']) + '_shu' +  str(config['lambda']['model_fit']['shuffle']) + '_cla' + str(config['lambda']['model_fit']['class_weight']) + '_sam' + str(config['lambda']['model_fit']['sample_weight']) + '_ini' + str(config['lambda']['model_fit']['initial_epoch']) + '_ste' + str(config['lambda']['model_fit']['steps_per_epoch']) + '_vst' + str(config['lambda']['model_fit']['validation_steps']) + '_vbs' + str(config['lambda']['model_fit']['validation_batch_size']) + '_vfr' + str(config['lambda']['model_fit']['validation_freq'])


def inet_path_LR(config):
    return lambda_path_LR(config) + '/' 'tsi' + str(config['inets']['data_prep']['train_test_val_split']['test_size']) + '_vsi' + str(config['inets']['data_prep']['train_test_val_split']['val_size']) + '_ran' + str(config['inets']['data_prep']['train_test_val_split']['random_state']) + '_shu' + str(config['inets']['data_prep']['train_test_val_split']['shuffle']) + '_str' + str(config['inets']['data_prep']['train_test_val_split']['stratify']) + '_bat' + str(config['inets']['model_fit']['batch_size']) + '_epo' + str(config['inets']['model_fit']['epochs']) + '_shu' +  str(config['inets']['model_fit']['shuffle']) + '_cla' + str(config['inets']['model_fit']['class_weight']) + '_sam' + str(config['inets']['model_fit']['sample_weight']) + '_ini' + str(config['inets']['model_fit']['initial_epoch']) + '_ste' + str(config['inets']['model_fit']['steps_per_epoch']) + '_vst' + str(config['inets']['model_fit']['validation_steps']) + '_vbs' + str(config['inets']['model_fit']['validation_batch_size']) + '_vfr' + str(config['inets']['model_fit']['validation_freq'])

### DECISION TREES

def data_path_DT(config):
    return  'data_DT/mDe' + str(config['function_family']['maximum_depth']) + '_bet' + str(config['function_family']['maximum_depth']) + '_dsp' + str(config['function_family']['decision_sparsity']) + '_bet' + str(config['function_family']['maximum_depth']) + '_fgr' + str(config['function_family']['fully_grown']) + '_dtt' + str(config['function_family']['dt_type']) + '_nda' + str(config['data']['n_datasets']) + '_nsa' + str(config['data']['n_samples']) + '_nfe' + str(config['data']['n_features']) + '_nta' + str(config['data']['n_targets']) + '_noi' + str(config['data']['noise'])

def lambda_path_DT(config):
    return data_path_DT(config) + '/' + 'epo' + str(config['lambda_net']['epochs_lambda'])\
           + '__esl' + str(config['lambda_net']['early_stopping_lambda'])\
           + '_esmdl' + str(config['lambda_net']['early_stopping_min_delta_lambda'])\
           + '_bla' + str(config['lambda_net']['batch_lambda'])\
           + '_dla' + str(config['lambda_net']['dropout_lambda'])\
           + '_lnl' + str(config['lambda_net']['lambda_network_layers'])\
           + '_ubl' + str(config['lambda_net']['use_batchnorm_lambda'])\
           + '_ola' + str(config['lambda_net']['optimizer_lambda'])\
           + '_lla' + str(config['lambda_net']['loss_lambda'])\
           + '_nil' + str(config['lambda_net']['number_initializations_lambda'])\
           + '_ntl' + str(config['lambda_net']['number_of_trained_lambda_nets'])

def inet_path_DT(config):
    return lambda_path_DT(config) + '/model/'