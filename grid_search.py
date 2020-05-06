from helper_functions import *
from dataset import *
from train import *
from model import *
import torch.optim as optim
import pickle



# batch_size = 64
# maxlen = 50
# doRandom_train = True
# model_name = 'model_base'
# proportionRandom = 0.5
# epochs = 13
# lr = 2e-3
# use_cuda = True
# step_size = 3


batch_sizes = [16, 32]
maxlens = [50]
doRandom_train = True
model_name = 'model_base'
props = [0, 0.1, 0.3, 0.2]
epochs = 9
lrs = [1e-4, 1e-3, 1e-5]
use_cuda = True
steps = [3, 5, 7, 10]

res = {}
count=0

for batch_size in batch_sizes:
    for maxlen in maxlens:
        for proportionRandom in props:
            for lr in lrs:
                for step_size in steps:

                    #Creating instances of training and validation set
                    train_set = HumorDatasetWeightsAndRandom(filename = 'data/task-1/train_split_std.csv', 
                                                            doRandom = doRandom_train, proportionRandom = proportionRandom,
                                                            maxlen = maxlen)
                    val_set = HumorDatasetWeightsAndRandom(filename = 'data/task-1/val_split_std.csv', 
                                                        doRandom = False, maxlen = maxlen)
                    #Creating intsances of training and validation dataloaders
                    train_loader = DataLoader(train_set, batch_size = batch_size)
                    val_loader = DataLoader(val_set, batch_size = batch_size)

                    if use_cuda:
                        net = HumorRegressorBase(freeze_bert = True).cuda()
                    criterion = nn.MSELoss(reduction='none')

                    opti = optim.Adam(net.parameters(), lr = lr)
                    scheduler = torch.optim.lr_scheduler.StepLR(opti, step_size=step_size)

                    print('Start training, with batch size ', batch_size, ' maxlen ', maxlen, ' random sample ', doRandom_train)
                    print('Proportion of random ', proportionRandom, ', lr ', lr, 'stepSize', step_size, ' Epochs ', epochs)

                    net, train_error_all, val_error_all = train(net, criterion, opti, scheduler, 
                                                                train_loader, val_loader, epochs=epochs, use_cuda=use_cuda)

                    print("Finished training, best RMSE on the validation was ", np.sqrt(np.max(val_error_all)))

                    res[count] = {
                        'batch_size': batch_size, 'maxlen': maxlen, 'propRandom': proportionRandom,
                        'lr': lr, 'step_size': step_size,
                        'training_error': train_error_all,
                        'val_error': val_error_all
                    }

                    pickle.dump(res, open( "performace_base_BERT_2_temp.p", "wb" ) )

                    count+=1

pickle.dump(res, open( "performace_base_BERT_2.p", "wb" ) )