import pandas as pd
import numpy as np
import torch
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import *
import random

class HumorRegressorBase(nn.Module):

    def __init__(self, freeze_bert = True):
        super(HumorRegressorBase, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Regression layer
        self.fc1 = nn.Linear(768, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, seq, attn_masks):

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        out = cont_reps[:, 0]
        
        #Feeding cls_rep to the regressor layer
        #out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out