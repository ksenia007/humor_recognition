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


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def predict_dataset(data_loader, net, use_cuda=True):

	truth = np.zeros(1)
	preds = np.zeros(1)

	with torch.no_grad():
		for it, (seq, attn_masks, labels, _) in enumerate(data_loader):
			if use_cuda:
				seq, attn_masks = seq.cuda(), attn_masks.cuda()
			pred_temp = net(seq, attn_masks)
			preds = np.append(preds, pred_temp.detach().cpu().numpy())
			truth = np.append(truth, labels)
			if (it + 1) % 500 == 0: 
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
def evaluate_model(weights_location, model_name, dataset_name, data_set_params, data_loader_params, save_location='', use_cuda=True, save_res=True):
	
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
	if save_res:
		pickle.dump(results_dict, open(save_location, 'wb'))

	return results_dict


dataset_folder = 'data/test/'
dataset_files = ['puns_test.csv', 'short_news_test.csv', 'oneliners_neutral_test.csv', 'humi_funlines_test_replacements_unpaired.csv'] 
maxlen = 30
batch_size = 32
model_name = HumorCLS
dataset_name = BasicWeighted
#metadata_file_loc = 'grid/data_options/combined_datasets/metadata.pickle'
#metadata_file_loc = 'grid/data_options/just_datasets/metadata.pickle'
metadata_file_loc = 'grid/data_options_unfreeze/just_datasets/metadata_new.pickle'
#metadata_file_loc = 'grid/data_options_unfreeze/combined_datasets/metadata.pickle'
metadata_file = pickle.load(open(metadata_file_loc, 'rb'))
models = list(metadata_file.keys())

#save_loc = 'grid/data_options/combined_datasets/evaluation_results.pickle'
save_loc = 'grid/data_options_unfreeze/just_datasets/evaluation_results_new.pickle'
#save_loc = 'grid/data_options_unfreeze/combined_datasets/evaluation_results_new.pickle'

res = {}
count = 0
for model in models:
	res_model = {}
	for test_file in dataset_files:
		test_file = dataset_folder+test_file
		print('Checking:', model, test_file)

		data_set_params = {'filename': test_file, 'maxlen': maxlen, 'weight': 1}
		data_loader_params = {'batch_size': batch_size}
		results_dict = evaluate_model(model, model_name, dataset_name, data_set_params, data_loader_params, save_location=save_loc, use_cuda=True, save_res=False)
		res_model[test_file] = results_dict
		print('AUC:', results_dict['metrics']['roc_auc'])
	res[model] = res_model
	pickle.dump(res, open(save_loc, 'wb'))

pickle.dump(res, open(save_loc, 'wb'))












