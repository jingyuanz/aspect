#encoding=utf-8
import sys
from flask import Flask, request, jsonify
from flask_json import FlaskJSON, as_json
import json
import torch
import numpy as np
from inference import Inferencer
from flask_cors import CORS

#nlu = NLU()
app = Flask(__name__)
CORS(app)

inferencer = Inferencer()
inferencer.load_model()
#trainer = MLPTrainer('online_dev', dropout=0.3, cuda=1)
#trainer.train_on_corpus('baidu/train.txt.2', 'baidu/dev.txt.2', evaluate=True)
#trainer.load_model('model/model.h5', 'model/class.dict')
#trainer.model.eval()
@app.route('/predict', methods=['POST'])
@as_json
def process():
    data = request.get_json(force=True, silent=False)
        # request_encoder = json.loads(json.dumps(data))
    # query = request_encoder['query']
    print(data)
    query = data['text']
#    emb = nlu.bert_encode([query], False)['vec']
#    emb = torch.FloatTensor(emb).to(trainer.device)
#    res = trainer.model(emb).tolist()
#    res = np.argmax(res, axis=-1).squeeze()
#    res = int(res)
#    res = trainer.class_dict[str(res)]
    aspect_dict, emotions_dict, kws = inferencer.predict(query)

    msg = f"{aspect_dict} , {emotions_dict} , {kws}"
    print(msg)

    response = {}
    response['msg'] = msg
    response['emotion'] = emotions_dict
    response['aspects'] = aspect_dict
    response['kws'] = kws
#    response['sentiment'] = res
    return jsonify(response)
    


if __name__ == "__main__":
    port = str(sys.argv[1])
    app.run('0.0.0.0',port)
    
    # bert.fit()
    # bert.save()
    # bert.load()
    #print(bert.process('垃圾分类'))
