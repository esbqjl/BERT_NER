from glob import glob
import os
import random
import pandas as pd

from config import *

def get_annotation(ann_path):
    with open(ann_path, encoding="utf-8") as file:
        anns={}
        for line in file.readlines():
            arr = line.split('\t')[1].split()
            name = arr[0]
            start = int(arr[1])
            end = int(arr[-1])
            if end-start>50:
                continue
            anns[start] = "B-" + name
            for i in range(start+1,end):
                anns[i]= 'I-' + name
        return anns
        
def get_text(text_path):
    with open(text_path,encoding="utf-8") as file:
        return file.read()
    
def generate_annotation():
    for text_path in glob(ORIGIN_DIR + "*.txt"):
        ann_path = text_path[:-3] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(text_path)
        
        
        df = pd.DataFrame({'word':list(text),'label':['O'] *len (text)})
        df.loc[anns.keys(),'label'] = list(anns.values())
        file_name = os.path.split(text_path)[1]
        df.to_csv(ANNOTATION_DIR+file_name,header=None,index=None)

def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    merge_files(train_files, TRAIN_SAMPLE_PATH)
    merge_files(test_files, TEST_SAMPLE_PATH)
    
    
def merge_files(files, path):
    with open(path, 'a',encoding="utf-8") as file:
        for f in files:
            text = open(f,encoding='utf-8').read()
            file.write(text)


def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v:k for k,v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header = None, index=None)
    
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    
    vocab_list = df['label'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v:k for k,v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(LABEL_PATH, header = None, index=None)
   
