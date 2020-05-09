from dataset import *
from train import *
from models import *
import torch.optim as optim
import pickle
import uuid



# train_filename = 'data/basic_weighted_v1_train.csv'
# valid_filename = 'data/basic_weighted_v1_validation.csv'
# model_name = 'BERT_model_basic_weighted_CLS_'
# model_save_loc = 'grid/weighted/'
# train_function = train_weighted
# model_function = HumorCLS
# dataset_function = BasicWeighted


train_function = train
model_function = HumorCLS
dataset_function = Basic

train_filename = 'data/basic_full_dataset_v1_train.csv'
valid_filename = 'data/basic_full_dataset_v1_validation.csv'
model_name = 'BERT_model_basic_CLS_'
model_save_loc = 'grid/basic/'

use_cuda = True
epochs = 7


batch_size_opt = [64, 128, 256]
lr_opt = [0.001, 0.0001, 0.00001]
maxlen_opt = [25, 30, 35]
step_size_opt = [5, 10]


res = {}

for batch_size in batch_size_opt:
	for maxlen in maxlen_opt:
		for lr in lr_opt:
			for step_size in step_size_opt:

				filename = str(uuid.uuid4())

				#Creating instances of training and validation set
				train_set = dataset_function(filename = train_filename, maxlen = maxlen)
				val_set = dataset_function(filename = valid_filename, maxlen = maxlen)

				#Creating intsances of training and validation dataloaders
				train_loader = DataLoader(train_set, batch_size = batch_size)
				val_loader = DataLoader(val_set, batch_size = batch_size)

				if use_cuda:
				    net = model_function(freeze_bert = True).cuda()

				criterion = nn.BCELoss()

				opti = optim.Adam(net.parameters(), lr = lr)

				scheduler = torch.optim.lr_scheduler.StepLR(opti, step_size=step_size)

				print('Start training, with batch size ', batch_size, ' maxlen ', maxlen)
				print('lr ', lr, 'stepSize', step_size, ' epochs ', epochs)
				model_name_var = model_save_loc+model_name+filename+'.pt'

				net, train_error_all, val_error_all = train_function(net, criterion, opti, scheduler, 
				                                            train_loader, val_loader, epochs=epochs, 
				                                            use_cuda=use_cuda, save_model=True, model_name=model_name_var)

				
				res[model_name_var] = {
				'train_error_all': train_error_all,
				'val_error_all': val_error_all,
				'parameters': {
					'batch_size': batch_size,
					'maxlen': maxlen,
					'lr': lr,
					'step': step_size
				},
				'train_filename': train_filename
				}

				pickle.dump(res, open(model_save_loc+'metadata.pt', 'wb'))






