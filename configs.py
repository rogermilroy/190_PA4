cfg = {}
cfg['model'] = 'lstm' # input model
cfg['input_dim'] = 203 # input dimension to LSTM
cfg['hidden_dim'] = 100 # hidden dimension for LSTM
cfg['output_dim'] = 98 # output dimension of the model
cfg['layers'] = 2 # number of layers of LSTM
cfg['dropout'] = 0 # dropout rate between two layers of LSTM; useful only when layers > 1; between 0 and 1
cfg['bidirectional'] = False # True or False; True means using a bidirectional LSTM
cfg['batch_size'] = 128 # batch size of input
cfg['learning_rate'] = 0.05 # learning rate to be used
cfg['L2_penalty'] = 0 # weighting constant for L2 regularization term; this is a parameter when you define optimizer
cfg['gen_temp'] = 1. # temperature to use while generating reviews
cfg['max_len'] = 2000 # maximum character length of the generated reviews
cfg['epochs'] = 4 # number of epochs for which the model is trained
cfg['cuda'] = True #True or False depending whether you want to run your model on a GPU or not. If you set this to True, make sure to start a GPU pod on ieng6 server
cfg['train'] = True # True or False; True denotes that the model is bein deployed in training mode, False means the model is not being used to generate reviews

cfg['training_losses_dir'] = "./outputs/training_losses"
cfg['validation_losses_dir'] = "./outputs/validation_losses"
cfg['bleu_scores_dir'] = "./outputs/bleu_scores"

cfg['params_dir'] = "./outputs/parameters.pt"

gen_cfg = {}
gen_cfg['model'] = 'lstm' # input model
gen_cfg['input_dim'] = 203 # input dimension to LSTM
gen_cfg['hidden_dim'] = 100 # hidden dimension for LSTM
gen_cfg['output_dim'] = 98 # output dimension of the model
gen_cfg['layers'] = 2 # number of layers of LSTM
gen_cfg['dropout'] = 0 # dropout rate between two layers of LSTM; useful only when layers > 1;
# between 0 and 1
gen_cfg['bidirectional'] = False # True or False; True means using a bidirectional LSTM
gen_cfg['batch_size'] = 512 # batch size of input
gen_cfg['learning_rate'] = 0.05 # learning rate to be used
gen_cfg['L2_penalty'] = 0 # weighting constant for L2 regularization term; this is a parameter when
# you define optimizer
gen_cfg['gen_temp'] = 0.4 # temperature to use while generating reviews
gen_cfg['max_len'] = 2000 # maximum character length of the generated reviews
gen_cfg['epochs'] = 4 # number of epochs for which the model is trained
gen_cfg['cuda'] = True #True or False depending whether you want to run your model on a GPU or not.
# If you set this to True, make sure to start a GPU pod on ieng6 server
gen_cfg['train'] = False # True or False;
gen_cfg['training_losses_dir'] = "./outputs/training_losses"
gen_cfg['validation_losses_dir'] = "./outputs/validation_losses"
gen_cfg['bleu_scores_dir'] = "./outputs/bleu_scores"

gen_cfg['params_dir'] = "./outputs/parameters.pt"

