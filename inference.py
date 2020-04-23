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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device("cpu")

    def load_model(self):
        self.model = LCF_ATEPC.from_pretrained('./output/model/')
        self.model.eval()
        #print(self.model)

    def predict(self, sent):
        emb = [self.tokenizer.encode(sent, add_special_tokens=True)]
        emb = torch.LongTensor(emb).to(self.cpu)
        ate, apc = self.model(emb)

        ate_logits = torch.argmax(F.log_softmax(ate, dim=2), dim=2)
        ate= ate_logits.detach().cpu().numpy()
        apc = torch.argmax(apc, -1).item()
        print(ate, apc)

if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.load_model()
    inferencer.predict("我不喜欢这个老师")




