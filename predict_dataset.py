from helper_functions import *
from dataset import *
from models import *
import torch.optim as optim
import pickle

from scipy.stats import spearmanr
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


def predict_dataset(data_loader, net, use_cuda=True):

	truth = np.zeros(1)
	preds = np.zeros(1)

	with torch.no_grad():
		for it, (seq, attn_masks, labels) in enumerate(data_loader):
			if use_cuda:
				seq, attn_masks = seq.cuda(), attn_masks.cuda()
			pred_temp = net(seq, attn_masks)
			preds = np.append(preds, pred_temp.detach().cpu().numpy())
			truth = np.append(truth, labels)
			if (it + 1) % 100 == 0: 
				print('Iteration ', it)
	return truth[1:], preds[1:]

def calculate_metrics(truth, preds):

	fpr, tpr, thresholds = roc_curve(y_true=(truth>=.5), y_score=preds)
	roc_auc = auc(fpr, tpr)

	res = {
		'roc_auc' : roc_auc,
		'fpr' : fpr,
		'tpr' : tpr,
		'thresholds' : thresholds,
		'spearman' : spearmanr(truth, preds)
		}
	return res

# accuracy precision recall spearman
def evaluate_model(weights_location, model_name, dataset_name, data_set_params, data_loader_params, save_location='', use_cuda=True):
	
	data_set = dataset_name(**data_set_params)
	data_loader = DataLoader(data_set, **data_loader_params)
	

	net = model_name()
	if use_cuda: map_location = 'cuda'	
	else: map_location = 'cpu'	
	state_dict = torch.load(weights_location, map_location=map_location)
	net.load_state_dict(state_dict)
	if use_cuda: net = net.cuda()

	truth, preds = predict_dataset(data_loader, net)
	results_dict = {
		'model_name': model_name,
		'dataset_name': dataset_name,
		'weights_location': weights_location,
		'truth': truth,
		'preds' : preds,
		'metrics' : calculate_metrics(truth, preds)
	}

	pickle.dump(results_dict, open(save_location, 'wb'))

	print(calculate_metrics(truth, preds))



dataset_file = 'data/humicroedit_unpaired_valid.csv'
maxlen = 25
batch_size = 32
weights_location = 'BERT_model_humicroedit_unpaired.pt'
save_loc = 'output/evaluation_for_' + weights_location
model_name = HumorRegressorBase
dataset_name = HumicroeditBasic


data_set_params = {'filename': dataset_file, 'maxlen': maxlen}
data_loader_params = {'batch_size': batch_size}

evaluate_model(weights_location, model_name, dataset_name, data_set_params, data_loader_params, save_location=save_loc, use_cuda=True)