from dataset import *
from train import *
from models import *
import torch.optim as optim
import pickle


train_filename = 'data/humicroedit_unpaired_train.csv'
valid_filename = 'data/humicroedit_unpaired_valid.csv'
model_name = 'BERT_model_humicroedit_unpaired.pt'



batch_size = 64
maxlen = 30
epochs = 36
lr = 0.01
use_cuda = True
step_size = 4


#Creating instances of training and validation set
train_set = HumicroeditBasic(filename = train_filename, maxlen = maxlen)
val_set = HumicroeditBasic(filename = valid_filename, maxlen = maxlen)

#Creating intsances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size = batch_size)
val_loader = DataLoader(val_set, batch_size = batch_size)

if use_cuda:
    net = HumorRegressorBase(freeze_bert = False).cuda()

criterion = nn.MSELoss(reduction='none')

opti = optim.Adam(net.parameters(), lr = lr)

scheduler = torch.optim.lr_scheduler.StepLR(opti, step_size=step_size)

print('Start training, with batch size ', batch_size, ' maxlen ', maxlen)
print('lr ', lr, 'stepSize', step_size, ' epochs ', epochs)

net, train_error_all, val_error_all = train(net, criterion, opti, scheduler, 
                                            train_loader, val_loader, epochs=epochs, 
                                            use_cuda=use_cuda, save_model=True, model_name=model_name)

print("Finished training, best RMSE on the validation was ", np.sqrt(np.max(np.array(val_error_all))))
