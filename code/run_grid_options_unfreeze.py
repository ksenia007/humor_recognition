from dataset import *
from train import *
from models import *
import torch.optim as optim
import pickle
import uuid
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# DATASET OPTIONS, FIXED PARAMS

train_function = train_weighted
model_function = HumorCLS
dataset_function = BasicWeighted

model_name = 'BERT_model_CLS_'


use_cuda = True
epochs = 3
freeze_bert = False

batch_size = 32
maxlen = 30
lr = 5e-5
step_size = 100
folder_data = 'data/training_datasets/'
model_save_loc = 'grid/data_options_unfreeze/combined_datasets/'
datafile_opt = ['weighted_v3']
weight_avail = [0]
#model_save_loc = 'grid/data_options_unfreeze/just_datasets/'
#datafile_opt = ['short']
#weight_avail = [0]
weights = [1]

res = {}

for idata, dataf in enumerate(datafile_opt):
	for weight in weights:
		if weight!=1 and weight_avail[idata]==0: continue
		filename_model = str(uuid.uuid4())

		train_set = dataset_function(filename = folder_data+dataf+'_train.csv', maxlen = maxlen, weight=weight)
		val_set = dataset_function(filename = folder_data+dataf+'_validation.csv', maxlen = maxlen, weight=weight)

		train_loader = DataLoader(train_set, batch_size = batch_size)
		val_loader = DataLoader(val_set, batch_size = batch_size)

		if use_cuda:
			net = model_function(freeze_bert = freeze_bert).cuda()

		criterion = nn.BCELoss()

		opti = optim.Adam(net.parameters(), lr = lr)
		scheduler = torch.optim.lr_scheduler.StepLR(opti, step_size=step_size)

		print('Start training, file', dataf, ' weight ', weight, ' model name: ', filename_model)

		model_name_var = model_save_loc+model_name+filename_model+'.pt'

		net, train_error_all, val_error_all = train_function(net, criterion, opti, scheduler, 
		                                            train_loader, val_loader, epochs=epochs, 
		                                            use_cuda=use_cuda, save_model=False, model_name=model_name_var)

		
		res[model_name_var] = {
		'train_error_all': train_error_all,
		'val_error_all': val_error_all,
		'parameters': {
			'batch_size': batch_size,
			'maxlen': maxlen,
			'lr': lr,
			'step': step_size,
			'datafile': dataf,
			'weight': weight,
			'freeze_bert': freeze_bert
		}
		}

		pickle.dump(res, open(model_save_loc+'metadata_weighted_v3.pickle', 'wb'))






