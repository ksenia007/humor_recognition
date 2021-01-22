from dataset import *
from train import *
from models import *
import torch.optim as optim
import pickle
import uuid
import warnings
from helper_functions import *
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



dataset_function = BasicWeighted

folder_data = 'data/training_datasets/'
datafile_opt = ['humicroedit', 'puns','oneliners', 'short']
base_file = 'output/embeddings/'

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.eval()
bert_model = bert_model.cuda()


for idata, dataf in enumerate(datafile_opt):
	train_set = dataset_function(filename = folder_data+dataf+'_train.csv', maxlen = 30, weight=1)
	print('Work with', dataf)

	results = np.zeros((len(train_set), 768))
	for i in range(len(train_set)):
		
		tokens = train_set[i][0].unsqueeze(0).cuda()
		attn_mask = train_set[i][1].unsqueeze(0).cuda()
		
		_, cls_head = bert_model(tokens, attention_mask = attn_mask)
		results[i, :] = cls_head.cpu().detach()

	filename = base_file+dataf+'_embeddings.npy'
	np.save(filename, results)
