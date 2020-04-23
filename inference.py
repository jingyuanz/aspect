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
from pytorch_transformers.modeling_bert import BertModel
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils.data_utils import ATEPCProcessor, convert_examples_to_features
from model.lcf_atepc import LCF_ATEPC
from torch import load

class Inferencer:
    def __init__(self):
        self.model_path = 'output/model'
        self.processor = ATEPCProcessor()
        self.labels = self.processor.get_labels()
        self.n_class = len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained('./output/model/vocab.txt')


    def load_model(self):
        self.model = load('./output/model/pytorch_model.bin')

    def predict(self, sent):
        emb = self.tokenizer.encode(sent, add_special_tokens=True)
        print(emb)

if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.predict("我不喜欢这个老师")




