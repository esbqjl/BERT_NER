import torch
from torch.utils import data
from config import *
import pandas as pd

# def get_vocab():
#     df = pd.read_csv(VOCAB_PATH, names=['word','id'])
    
#     return list(df['word']), dict(df.values)

# def get_label():
#     df = pd.read_csv(LABEL_PATH, names=['label','id'])
#     return list(df['label']), dict(df.values)


# class Dataset(data.Dataset):
#     def __init__(self,type = 'train', base_len=50):
#         super(Dataset,self).__init__()
#         self.base_len = base_len
#         sample_path = TRAIN_SAMPLE_PATH  if type =='train' else TEST_SAMPLE_PATH
#         self.df = pd.read_csv(sample_path,names = ['word','label'])
#         self.id2label, self.word2id = get_vocab()
#         self.id2label, self.label2id = get_label()
#         self.data = self._create_batches()
        
#     def _create_batches(self):
#         batches = []
#         words = list(self.df['word'])
#         labels = list(self.df['label'])
#         i = 0
#         while i < len(words):
#             # 定位批次终点
#             end = min(i + self.base_len, len(words))
#             # 防止切割实体
#             while end < len(words) and labels[end - 1].startswith('I-'):
#                 end += 1
            
#             batch_words = words[i:end]
#             batch_labels = labels[i:end]
#             batches.append((batch_words, batch_labels))
#             i = end
#         return batches
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         words, labels = self.data[idx]
#         return {
#             'words': torch.tensor([self.word2id.get(word, self.word2id['<UNK>']) for word in words], dtype=torch.long),
#             'labels': torch.tensor([self.label2id[label] for label in labels], dtype=torch.long)
#         }

def create_batches(type='train',base_len=50):
    
    sample_path = TRAIN_SAMPLE_PATH  if type =='train' else TEST_SAMPLE_PATH
    df = pd.read_csv(sample_path,names = ['word','label'])
    batch_text,batch_label = [],[]
    words = list(df['word'])
    labels = list(df['label'])
    
    for i in range(len(words)):
        if pd.isna(words[i]):
            words[i]=' '
    i=0      
    while i < len(words):
        # 定位批次终点
        end = min(i + base_len, len(words))
        # 防止切割实体
        
        while end < len(words) and labels[end - 1].startswith('I-'):
            end += 1
        
        batch_words = words[i:end]
        batch_labels = labels[i:end]
        batch_text.append(batch_words)
        batch_label.append(batch_labels)
        i = end
    return batch_text,batch_label






    