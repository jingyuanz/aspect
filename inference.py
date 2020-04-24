import argparse
import json
import logging
import os, sys
import random
from sklearn.metrics import f1_score
from time import strftime, localtime

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertModel, BertForSequenceClassification
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils.data_utils import ATEPCProcessor, convert_examples_to_features
# from model.lcf_atepc import LCF_ATEPC
from model.lc_pred import LCF_ATEPC
from torch import load, save

class Inferencer:
    def __init__(self):
        self.model_path = 'output/model'
        self.processor = ATEPCProcessor()
        self.labels = self.processor.get_labels()
        self.n_class = len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained('./output/model/vocab.txt')
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device("cpu")

    def load_model(self):
#        self.model = LCF_ATEPC.from_pretrained('./output/model/')
        self.model = load('./output/final.bin', map_location=self.device)
        print(len(list(self.model.modules())))
        #for module in self.model.modules():
         #   print(module)
        self.model.eval()
        #print(self.model)

    def predict(self, sent):
        emb = self.tokenizer.encode(sent, add_special_tokens=True)
#        print(len(emb))
        tokens = self.tokenizer.tokenize(sent)
        print(len(tokens), len(emb))
        origin_len = len(emb)
        input_mask = torch.LongTensor([[1]*len(emb) + [0]*(40-len(emb))]).to(self.device)
        valid_ids = torch.LongTensor([[1]*40]).to(self.device)
        l_mask = input_mask
        segment_ids = torch.LongTensor([[0]*40]).to(self.device)


        if len(emb)<40:
            emb+=[0]*(40-len(emb))
        
        emb = torch.LongTensor([emb]).to(self.device)
        ate, apc = self.model(emb, segment_ids, input_mask, None, None, valid_ids, None)

        ate_logits = torch.argmax(F.log_softmax(ate, dim=2), dim=2)
        ate= ate_logits.detach().cpu().numpy()[0]
        apc = torch.argmax(apc, -1).item()
        emotion = 'pos' if apc else 'neg'
        print(ate)
        spans = self.decode(ate, origin_len, tokens)
        print(emotion, spans)

    def decode(self, seq, olen, tokens):
        seq = seq[1:olen-1]
        print(seq, tokens)
#        assert len(tokens) == len(seq)
        lens = len(seq)
        i = lens - 1
        spans = []
        tmp = []
        while i >= 0:
            w = seq[i]
            if w == 2 or w == 3:
                tmp.append(tokens[i])
                i -= 1
                
            else:
                if tmp:
                    span = tmp[::-1]
                    span = ''.join(span)
                    spans.append(span)
                    tmp = []
                i -= 1
        if tmp:
            span = tmp[::-1]
            span = ''.join(span)
            spans.append(span)
        return spans




if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.load_model()
    while True:
        sent = input('sent: ')
        inferencer.predict(sent)




