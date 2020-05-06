from helper_functions import *
from dataset import *

def train(net, criterion, opti, scheduler, train_loader, val_loader, epochs=2, use_cuda = False, save_model=False, model_name=''):
    train_error_all = list()
    val_error_all = list()

    best_val_loss = 100

    for ep in range(epochs):
        loss_cum=0
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            
            if use_cuda:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            opti.zero_grad()  
            
            preds = net(seq, attn_masks)

            #Computing loss
            loss = torch.mean(criterion(preds.squeeze(-1), labels.float()))
            loss_cum += loss.item()
            loss.backward()
            opti.step()

        train_error_all.append(loss_cum/(it+1))
        print('Epoch ', ep, ' Loss: ', (loss_cum/(it+1)))

        #approximate the validation loss: do 2 batches of validation
        with torch.no_grad():
            loss_val = 0
            for it, (seq, attn_masks, labels) in enumerate(val_loader):
                if use_cuda:
                    seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
                preds = net(seq, attn_masks)
                loss = torch.mean(criterion(preds.squeeze(-1), labels.float()))
                loss_val += loss
        
        if best_val_loss>(loss_val/(it+1)):
            #update and save
            best_val_loss = loss_val/(it+1)
            print('Updating best loss on validation, ', best_val_loss)
            torch.save(net.state_dict(), model_name)

        val_error_all.append(loss_val/(it+1))
        print('Loss on validation: ', loss_val/(it+1))

        scheduler.step()
    
    return net, train_error_all, val_error_all