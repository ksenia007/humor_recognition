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
        
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)

        #Regression layer
        self.fc1 = nn.Linear(12096, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, seq, attn_masks):

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        #print(cont_reps.shape)
        out = cont_reps.unsqueeze(1)
        out = self.pool(F.relu(self.conv1(out)))
        #print(out.shape)
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(out.size(0), 12096)
        #Feeding cls_rep to the regressor layer
        #out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out