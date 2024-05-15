import numpy as np
import torch
from sklearn.metrics import accuracy_score
from data_process import *
from NER.bert_fc.ner_trainer import BertFCTrainer
from NER.bert_fc.ner_predictor import BertFCPredictor
from utils import create_batches

# setup random seed

seed =0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# generate_annotation()
# split_sample(0.2)

train_texts,train_labels = create_batches('train',64)
test_texts,test_labels = create_batches('test',64)


for lr_times in [10,100,1000]:
    trainer= BertFCTrainer(
        pretrainerd_model_dir='./model/bert-base-chinese',model_dir='./temp/bertNER_CRF_lr_{}'.format(lr_times),learning_rate=5e-5,lr_times = lr_times
    )

    trainer.train(
        train_texts,train_labels,validate_texts = test_texts, validate_labels = test_labels, batch_size=64,epochs=4
    )

# trainer= BertFCTrainer(
#         pretrainerd_model_dir='./model/bert-base-chinese',model_dir='./temp/bertNER',learning_rate=5e-5
# )

# trainer.train(
#     train_texts,train_labels,validate_texts = test_texts, validate_labels = test_labels, batch_size=64,epochs=4
# )


# predictor= BertFCPredictor(
#         pretrained_model_dir='./model/bert-base-uncased',model_dir='./temp/bertfc'
#     )

# predict_labels,_,index = predictor.predict(test_texts, batch_size=64)

