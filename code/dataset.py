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



class BasicWeighted(Dataset):

    def __init__(self, filename, maxlen, weight):
        self.df = pd.read_csv(filename)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #Initialize the BERT tokenizer
        self.maxlen = maxlen
        self.weight = weight

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'text']
        try:
            label = self.df.loc[index, 'label']
        except:
            label = self.df.loc[index, 'score']
        try:
            weight = self.df.loc[index, 'weight']
            if weight != 1:
                weight = self.weight
        except:
            weight = 1
        
        #Preprocessing the text to be suitable for BERT
        #Tokenize the sentence
        tokens_orig = self.tokenizer.tokenize(sentence) 
        
        tokens = ['[CLS]'] + tokens_orig + ['[SEP]']
        
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label, weight


class Basic(Dataset):

    def __init__(self, filename, maxlen):
        self.df = pd.read_csv(filename)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #Initialize the BERT tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'text']
        label = self.df.loc[index, 'label']
        
        #Preprocessing the text to be suitable for BERT
        #Tokenize the sentence
        tokens_orig = self.tokenizer.tokenize(sentence) 
        
        tokens = ['[CLS]'] + tokens_orig + ['[SEP]']
        
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


class HumicroeditBasic(Dataset):

    def __init__(self, filename, maxlen):
        self.df = pd.read_csv(filename)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #Initialize the BERT tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'text']
        label = self.df.loc[index, 'scaled_mean']
        
        #Preprocessing the text to be suitable for BERT
        #Tokenize the sentence
        tokens_orig = self.tokenizer.tokenize(sentence) 
        
        tokens = ['[CLS]'] + tokens_orig + ['[SEP]']
        
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label