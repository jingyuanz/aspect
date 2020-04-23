
from pytorch_transformers.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention

from torch.nn import Linear, CrossEntropyLoss
import torch
import torch.nn as nn
import copy
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, 40),
                                            dtype=np.float32), dtype=torch.float32).to(self.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class LCF_ATEPC(BertForTokenClassification):

    def __init__(self, bert_base_model):
        super(LCF_ATEPC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
        self.bert = bert_base_model
        # do not init lcf layer if BERT-SPC or BERT-BASE specified
        self.local_bert = copy.deepcopy(self.bert)
        self.pooler = BertPooler(config)
        self.dense = torch.nn.Linear(768, 2)
        self.bert_global_focus = self.bert
        self.bert_SA = SelfAttention(config)
        self.dropout = torch.nn.Dropout(0.0)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)


    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None, valid_ids=None,
                attention_mask_label=None):
        global_context_out, _ = self.bert(input_ids_spc, token_type_ids, attention_mask)
        # code block for ATE task
        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)


        local_context_ids = input_ids_spc
        local_context_out, _ = self.local_bert(input_ids_spc)
        batch_size, max_len, feat_dim = local_context_out.shape
        local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    local_valid_output[i][jj] = local_context_out[i][j]
        local_context_out = self.dropout(local_valid_output)

        cdm_vec = self.feature_dynamic_mask(local_context_ids, polarities)
        cdm_context_out = torch.mul(local_context_out, cdm_vec)
        cat_out = torch.cat((global_context_out, cdm_context_out), dim=-1)
        cat_out = self.linear_double(cat_out)
        sa_out = self.bert_SA(cat_out)
        pooled_out = self.pooler(sa_out)
        pooled_out = self.dropout(pooled_out)
        apc_logits = self.dense(pooled_out)


        return ate_logits, apc_logits