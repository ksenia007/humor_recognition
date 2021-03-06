from dataset import *
from train import *
from models import *
import torch.optim as optim
import pickle


train_filename = 'data/basic_weighted_v1_train.csv'
valid_filename = 'data/basic_weighted_v1_validation.csv'
model_name = 'BERT_model_basic_weighted_CLS_v2.pt'

train_function = train_weighted
model_function = HumorCLS
dataset_function = BasicWeighted


batch_size = 64
maxlen = 30
epochs = 36
lr = 0.001
use_cuda = True
step_size = 10


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

net, train_error_all, val_error_all = train_function(net, criterion, opti, scheduler, 
                                            train_loader, val_loader, epochs=epochs, 
                                            use_cuda=use_cuda, save_model=True, model_name=model_name)

print("Finished training, best RMSE on the validation was ", np.sqrt(np.max(np.array(val_error_all.cpu()))))
