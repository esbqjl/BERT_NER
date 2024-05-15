from transformers import AutoTokenizer

import torch
from datasets import load_dataset, load_from_disk

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from data_process import *
from NER.bert_fc.ner_trainer import BertFCTrainer
from NER.vocab import Vocab
from utils import create_batches
from transformers import BertTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        #names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

        #在线加载数据集
        #dataset = load_dataset(path='peoples_daily_ner', split=split)

        #离线加载数据集
        dataset = load_from_disk(dataset_path='./test_data')[split]

        #过滤掉太长的句子
        def f(data):
            return len(data['tokens']) <= 512 - 2

        dataset = dataset.filter(f)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]['tokens']
        labels = self.dataset[i]['ner_tags']

        return tokens, labels


train_dataset = Dataset('train')
validation_dataset = Dataset('validation')

# setup random seed
seed =0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
train_texts,train_labels = [],[]
validation_texts,validation_labels = [],[]
for i in train_dataset:
    text, label = i
    train_texts.append(text)
    train_labels.append(label)

for i in validation_dataset:
    text, label = i
    validation_texts.append(text)
    validation_labels.append(label)


trainer= BertFCTrainer(
    pretrainerd_model_dir='./model/bert-base-chinese',model_dir='./temp/bertTestNER',learning_rate=5e-5
)

trainer.train(
    train_texts,train_labels,validate_texts = validation_texts, validate_labels = validation_labels, batch_size=64,epoch=8
)