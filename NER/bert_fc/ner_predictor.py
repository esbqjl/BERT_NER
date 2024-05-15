import torch
import json
import math
from NER.base.base_predictor import BasePredictor
from NER.bert_fc.ner_model import BertFCModel
from NER.vocab import Vocab
from pathlib import Path
import torch.nn.functional as F
import os
class BertFCPredictor(BasePredictor):
    def __init__(self, pretrained_model_dir, model_dir, vocab_name = 'vocab.json'):
        self.pretrained_mode_dir = pretrained_model_dir
        self.model_dir=model_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vocab = Vocab()
        
        self._load_config()
        
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()
    def _load_config(self):
        print(os.getcwd())
        with open('{}/train_config.json'.format(self.model_dir),'r') as f:
            self._config = json.loads(f.read())
    
    def _load_model(self):
        self.model=BertFCModel(
            self.pretrained_mode_dir,self._config['label_size']
        )
        print('{}/{}'.format(self.model_dir,self._config['ckpt_name']))
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir,self._config['ckpt_name']), map_location = self.device)
        )
        self.model.eval()
        
        self.model.to(self.device)
        
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])
    
    def predict(self,texts,batch_size=64, max_len=512):
        '''
        Args:
            texts: list[list[str]]. to predict the sample, if it's segment, make sure it is segmented by bert's tokenizer
            batch_size: int
            max_len: int the longest sequence length (make sure it remain the same with max_postion_embeding in bert pretrained model)\
            
        Returns:
            list[list[str]] label sequence
        
        '''
        batch_labels = []
        probabilities=[]
        possible_index=[]
        for batch_idx in range(math.ceil(len(texts)/batch_size)):
            text_batch = texts[batch_size * batch_idx : batch_size * (batch_idx+1)]
            
            # longest length of current batch        
            batch_max_len = min(max([len(texts) for texts in text_batch])+2,max_len)
            
            batch_input_ids,batch_att_mask = [],[]
            for text in text_batch:
                assert isinstance(text,list)
                text = ' '.join(text)
                encoded_dict = self.model.bert_tokenizer.encode_plus(text, max_length=batch_max_len,padding='max_length',
                                                                        return_tensors='pt',truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'].squeeze(0))
                batch_att_mask.append(encoded_dict['attention_mask'].squeeze(0))
                
            batch_input_ids = torch.stack(batch_input_ids)
            batch_att_mask = torch.stack(batch_att_mask)
            
            batch_input_ids, batch_att_mask = \
                batch_input_ids.to(self.device), batch_att_mask.to(self.device)
            
            with torch.no_grad():
                logits = self.model(batch_input_ids,batch_att_mask)
                print(logits.shape)
                best_labels = torch.argmax(logits, dim=-1)  
                best_labels = best_labels.to(self.device)
                batch_labels.extend([self.vocab.id2tag[i.item()] for label_id in best_labels for i in label_id])
                
        return batch_labels
    
