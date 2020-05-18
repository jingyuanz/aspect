import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers.tokenization_bert import BertTokenizer
import re
from nlutools import NLU
NLUTOOLS = NLU()
NLUTOOLS.add_stopwords(['喜欢','不喜欢','想学', '不感兴趣', '兴趣', '感兴趣', '不想学', '关系', '没关系', '推荐', '不推荐', '推荐点','推荐些', '适合', '不适合'])
from torch import load
from jieba.analyse import extract_tags
from sentiment import sentiment_324
from collections import defaultdict
class Inferencer:
    def __init__(self):
        self.model_path = 'output/model'
        #self.processor = ATEPCProcessor()
        #self.labels = self.processor.get_labels()
        #self.n_class = len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained('./output/model/vocab.txt')
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device("cpu")

    def load_model(self):
#        self.model = LCF_ATEPC.from_pretrained('./output/model/')
        self.model = load('./output/model.bin', map_location=self.device)
        
        self.model.args.device = self.device
        print(self.model.args)
        print(len(list(self.model.modules())))
        #for module in self.model.modules():
         #   print(module)
        self.model.eval()
        #print(self.model)

    def predict(self, sent):
        sents = self.cut_sent(sent)
        emotion_dict = {}
        res_dict = {'pos':[], 'neg':[], 'neu':[]}
        raw_spans = []
        sent_aspects = defaultdict(list)
        for sent in sents:
            if len(sent)>40:
                sent = sent[:40]
            
            emb = self.tokenizer.encode(sent, add_special_tokens=True)
    #        print(len(emb))
            tokens = self.tokenizer.tokenize(sent)
            origin_len = len(emb)
            input_mask = torch.LongTensor([[1]*len(emb) + [0]*(40-len(emb))]).to(self.device)
            valid_ids = torch.LongTensor([[1]*40]).to(self.device)
            l_mask = input_mask.to(self.device)
            segment_ids = torch.LongTensor([[0]*40]).to(self.device)


            if len(emb)<40:
                emb+=[0]*(40-len(emb))

            emb = torch.LongTensor([emb]).to(self.device)
            ate, apc = self.model(emb, segment_ids, input_mask, None, None, valid_ids, None)
            print(apc)

            ate_logits = torch.argmax(F.log_softmax(ate, dim=2), dim=2)
            ate= ate_logits.detach().cpu().numpy()[0]
            apc = torch.argmax(apc, -1).item()
            print(apc)
            emotion = 'pos' if apc else 'neg'
            print(sent, emotion)
            spans = self.decode(ate, origin_len, tokens)


            if not spans:
                kws = extract_tags(sent, topK=1)
                spans = kws
            spans = [s for s in spans if s not in NLUTOOLS.stopwords]
            spans_raw = ''.join(spans)
            print(spans_raw)
            print(len(spans_raw), len(sent))
            if len(spans_raw)>int(0.7*len(sent)):
                emotion = 'neu'
            res_dict[emotion]+=spans
            sent_aspects[sent] = spans
            raw_spans += spans
            emotion_dict[sent] = emotion
        #kws = raw_spans
        #emotion_dict = sentiment_324(sents, kws)
#        for sent in emotion_dict:
#            if emotion_dict[sent] == 'pos'
#            res_dict[emotion_dict[sent]] += sent_aspects[sent]
        print(res_dict)
        print(emotion_dict)
#        raw_kws = 
        return res_dict, emotion_dict, raw_spans

    def decode(self, seq, olen, tokens):
        seq = seq[1:olen-1]
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
                if w == 3 and (i<0 or seq[i] == 1):
                    seq[i] = 2
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
        spans = [x.replace('#','') for x in spans]
        
        return spans

    def cut_sent(self, sent):
        sent = sent.replace('，',',')
        p = re.compile('(只是|况且|不过|但|虽然|尽管|而|却|且|(而却)|并|(并且)|(但是)|(然而)|(可是)|(\s+))')
        sent = re.sub(p, ',', sent)
        print("HERE: ",sent)
        subs = NLUTOOLS.split(sent, cut_all=True)
        subs = [x for x in subs if x.strip()]
        return subs



if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.load_model()
    while True:
        sent = input('sent: ')
        inferencer.predict(sent)
#    print(inferencer.cut_sent('我不想看网络相关的,我想要关注细节'))




