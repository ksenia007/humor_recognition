import pandas as pd
import numpy as np
import torch
from transformers import BertModel
from transformers import BertTokenizer
import re

def replace_word(sentence, new_word):
    return re.sub('<(.*?)/>', new_word, sentence)

def drop_replacement_symbols(sentence):
    sentence = sentence.replace('<', '')
    return sentence.replace('/>', '')

def run_pretrained_for_sentence(sent, len_sent = 25):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(sent)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    padded_tokens = tokens + ['[PAD]' for _ in range(len_sent - len(tokens))]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    
    #Step 5: Get BERT vocabulary index for each token
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    #Converting everything to torch tensors before feeding them to bert_model
    token_ids = torch.tensor(token_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    #Feed them to bert
    hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask)
    
    return hidden_reps, cls_head



def run_for_sentence(net, sentence, maxlen):
    
    sentence = drop_replacement_symbols(sentence)

    #Preprocessing the text to be suitable for BERT
    tokens_orig = tokenizer.tokenize(sentence) #Tokenize the sentence

    tokens = ['[CLS]'] + tokens_orig + ['[SEP]'] 
    if len(tokens) < maxlen:
        tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))] #Padding sentences
    else:
        tokens = tokens[:maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

    tokens_ids = tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
    tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

    #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
    attn_mask = (tokens_ids_tensor != 0).long()

    return tokens_ids_tensor, attn_mask
