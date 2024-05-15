import numpy as np
import torch
from sklearn.metrics import accuracy_score
from pathlib import Path
from peft_predictor import PeftPredictor
from NER.bert_fc.ner_predictor import BertFCPredictor




sentence="我无聊的时候很喜欢去广州的博物馆逛，因为那能让我想起在北京的日子。"


line = sentence.strip()


text=[list(line)]
print(text)

current_dir = Path(__file__).parent
pretrained_model_dir = current_dir / 'model' / 'bert-base-chinese'

model_dir = current_dir / 'temp' / 'bertTestNER'
predictor= BertFCPredictor(
    pretrained_model_dir=pretrained_model_dir,model_dir=model_dir
)

predict_labels = predictor.predict(text, batch_size=64)
for i in range(len(text[0])):
    print(text[0][i], " ", predict_labels[i])
    
    
