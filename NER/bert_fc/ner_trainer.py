import json
import math
import os
from transformers.utils import logging
import torch
from sklearn.utils import shuffle
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from NER.base.base_trainer import BaseTrainer
import numpy as np
from NER.bert_fc.ner_model import BertFCModel
from NER.vocab import Vocab
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
writer = SummaryWriter()
class BertFCTrainer(BaseTrainer):
    def __init__(self,pretrainerd_model_dir, model_dir,learning_rate=5e-5,
                 ckpt_name='pytorch_model.bin',vocab_name='vocab.json', lr_times = 1):
        self.pretrained_model_dir = pretrainerd_model_dir
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name
        self.vocab_name = vocab_name
        self.lr_times = lr_times
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self. learning_rate = learning_rate
        self.batch_size = None
        
        self.vocab = Vocab()
        
    def _build_model(self):
        '''build up bert-fc model'''
        self.model = BertFCModel(self.pretrained_model_dir,self.vocab.label_size)
        # setup AdamW optimizer
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n,p in self.model.named_parameters() if not any (nd in n for nd in no_decay) and 'crf'not in n],
             'weight_decay':0.01},
            {'params': [p for n,p in self.model.named_parameters() if any (nd in n for nd in no_decay) and 'crf' not in n ],
             'weight_decay':0.0},
            # switch crf and GRU
            {'params': [p for n, p in self.model.named_parameters() if 'crf' in n],
            'lr': self.learning_rate*self.lr_times, 'weight_decay': 0.01},
        ]
        for i, group in enumerate(optimizer_grouped_parameters, 1):
            print(f"Group {i}:")
            print(f"Learning Rate: {group.get('lr', self.learning_rate)}")  # Default learning rate if not set
            print(f"Weight Decay: {group['weight_decay']}")
            
        for n, p in self.model.named_parameters():
            if 'crf' in n:
                print(n)   
        
        self.optimizer = AdamW(optimizer_grouped_parameters,lr = self.learning_rate)    
        
        # use bert's vocab to update our vocab object
        
        self.vocab.set_vocab2id(self.model.get_bert_tokenizer().vocab)
        
        self.vocab.set_id2vocab({_id:char for char, _id in self.vocab.vocab2id.items()})
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])
        
        self.model.to(self.device)
    
    def _save_config(self):
        config={
            'vocab_size': self.vocab.vocab_size,
            'label_size': self.vocab.label_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch':self.epochs,
            'ckpt_name':self.ckpt_name,
            'vocab_name':self.vocab_name,
            'pretrained_model':os.path.basename(self.pretrained_model_dir)
        }
        with open('{}/train_config.json'.format(self.model_dir),'w') as f:
            f.write(json.dumps(config,indent=4))
            
    def _transform_batch(self, batch_texts, batch_labels, max_length=512):
        
        batch_input_ids, batch_att_mask, batch_label_ids = [], [], []
        
        for text, labels in zip(batch_texts, batch_labels):
            assert isinstance(text, list)
            if len(text)<4:
                continue
            # encoded_dict = self.model.bert_tokenizer.encode_plus(
            #     text, 
            #     max_length=max_length,
            #     padding = 'max_length',
            #     return_tensors='pt',  
            #     truncation=True,
            #     is_split_into_words=True
            # )
            text = ['[CLS]'] + text + ['[SEP]']
            token_ids = self.model.bert_tokenizer.convert_tokens_to_ids(text)
            padding_length = max_length - len(token_ids)
            token_ids += [self.vocab.vocab2id['[PAD]']] * padding_length
            attention_mask = [1] * (len(token_ids) - padding_length) + [0] * padding_length
            # Directly append tensors to lists
            # batch_input_ids.append(encoded_dict['input_ids'].squeeze(0))  # Remove batch dimension
            
            # batch_att_mask.append(encoded_dict['attention_mask'].squeeze(0))  # Remove batch dimension
            batch_input_ids.append(torch.tensor(token_ids))
            batch_att_mask.append(torch.tensor(attention_mask))
            current_label = [self.vocab.tag2id['O']]+[self.vocab.tag2id[i] for i in labels]+[self.vocab.tag2id['O']]
            for i in range(max_length-len(current_label)):
                current_label.append(self.vocab.vocab2id['[PAD]'])

            batch_label_ids.append(torch.tensor(current_label))
        
        batch_input_ids = torch.stack(batch_input_ids)
        batch_att_mask = torch.stack(batch_att_mask)
        batch_label_ids = torch.stack(batch_label_ids) 
        
        # Sending to device
        batch_input_ids, batch_att_mask, batch_label_ids = \
            batch_input_ids.to(self.device), batch_att_mask.to(self.device), batch_label_ids.to(self.device)

        # Return corrected variables (the original return statement had a mistake)
        return batch_input_ids, batch_att_mask, batch_label_ids
            
    def _reshape_and_remove_pad(self,outs, labels, attention_mask):
        #变形,便于计算loss
        #[b, lens, 8] -> [b*lens, 8]
        b,l,d = outs.size()
        outs = outs.reshape(b*l,d)
        #[b, lens] -> [b*lens]
        labels = labels.reshape(-1)

        #忽略对pad的计算结果
        #[b, lens] -> [b*lens - pad]
        select = attention_mask.reshape(-1) == 1
        
        outs = outs[select]
        labels = labels[select]

        return outs, labels       
            
    def train(self, train_texts, labels, validate_texts, validate_labels, batch_size = 30, epochs =10):
        ''' train
        Args:
            train_text: list[list[str]] training dataset
            labels: list[list[str]] dataset labels
            validate_texts: list[list[str]] validate dataset
            validate_labels: list[list[str]] validate dataset labels
            batch_size: int
            epoch: int
        
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.vocab.build_vocab(labels=labels,build_texts=False,with_build_in_tag_id=False) # only bulid up the labels, bert vocab has those
        self._build_model()
        self.vocab.save_vocab('{}/{}'.format(self.model_dir,self.vocab_name))
        self._save_config()
        
        best_loss = float("inf")
        loss_buff = []
        max_loss_num = 10
        step = 0
        for epoch in range(epochs):
            for batch_idx in range(math.ceil(len(train_texts)/batch_size)):
                text_batch = train_texts[batch_size * batch_idx:batch_size * (batch_idx+1)] 
                labels_batch = labels[batch_size * batch_idx:batch_size * (batch_idx+1)] 
                step = step+1
                self.model.train()
                self.model.zero_grad()
                
                # the training process
                batch_max_len = max([len(text) for text in  text_batch]) + 2
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(text_batch,
                                                                                         labels_batch,
                                                                                         max_length=batch_max_len)
                
                # # if we adding CRF layer, we don't need to use CrossEntropy anymore
                # logits= self.model(batch_input_ids,batch_att_mask,labels = batch_label_ids)
                # logits,batch_label_ids = self._reshape_and_remove_pad(logits,batch_label_ids,batch_att_mask)
                # best_label = torch.argmax(logits, dim=-1)  
                # best_label = best_label.to(self.device)
                # loss = CrossEntropyLoss(ignore_index=-1)(logits,batch_label_ids) 
                
                _, loss= self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
                
                loss.backward()                
                
                self.optimizer.step()
                writer.add_scalar('Loss/Train', loss.item(), step)
                
                valid_acc, valid_loss = self.validate(validate_texts, validate_labels, step = step, sample_size = batch_size)
                loss_buff.append(valid_loss)   
                if len(loss_buff)>max_loss_num:
                    loss_buff=loss_buff[1:]             
                avg_loss = sum(loss_buff)/len(loss_buff) if len(loss_buff) == max_loss_num else None
                if avg_loss:
                    writer.add_scalar('Loss/Val', avg_loss, step)
                logger.info(
                        'epoch %d, step %d, train loss %.4f, valid acc %.4f, valid acc (without O) %.4f, last %d avg valid loss %s' % (
                            epoch, step, loss, valid_acc[0]/valid_acc[1], valid_acc[2]/valid_acc[3], max_loss_num,
                            '%.4f' % avg_loss if avg_loss else avg_loss
                    )
                )
                
                
                if avg_loss and avg_loss < best_loss:
                    best_loss =  avg_loss
                    torch.save(self.model.state_dict(), '{}/{}'.format(self.model_dir,self.ckpt_name))
                    logger.info("model saved")
        logger.info('finished')
        writer.close()
    def validate(self,validate_texts,validate_labels,step, sample_size=100):
        '''use validation dataset set from current model
        Args:
            validate_texts: list[list[str]] or np.array. Validate data
            validate_labels: list[list[str]] or np.array. Validate labels
            sample_size: int. sample size(only use batch size, to prevent validate dataset too large)

        Returns:
            float. validate acc, loss
        '''        
        self.model.eval()
        
        # randomly sample texts and labels of number of sample_size
        
        batch_texts, batch_labels=[
            return_val[:sample_size] for return_val in shuffle(validate_texts,validate_labels)
        ]  
        batch_max_len = max([len(text) for text in batch_texts])+2
        with torch.no_grad():
            batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(batch_texts,
                                                                                     batch_labels,
                                                                                     max_length=batch_max_len)
            
            # # same reason as the train, if we use CRF layer we need to change following
            # logits= self.model(batch_input_ids,batch_att_mask,labels=batch_label_ids)
            # logits,batch_label_ids = self._reshape_and_remove_pad(logits,batch_label_ids,batch_att_mask)
            # loss = CrossEntropyLoss(ignore_index=-1)(logits,batch_label_ids)
            
            logits, loss = self.model(batch_input_ids, batch_att_mask,labels= batch_label_ids)
            
            validate_acc = self.get_correct_and_total_count(outs=logits,labels=batch_label_ids)
            # calculate f1_score
            select = batch_att_mask.reshape(-1) == 1
            filtered_labels = batch_label_ids.reshape(-1)[select].cpu()
            pred = [item for sublist in logits for item in sublist]
            pred = torch.tensor(pred) 
            
            # logits = logits.argmax(dim=1)
            
            f1 = f1_score(pred, filtered_labels, average='macro')
            writer.add_scalar('f1/Val', f1, step)
            return validate_acc,loss
               
    def _get_acc_one_step(self, labels_predict_batch, labels_batch):
        acc = (labels_predict_batch==labels_batch).sum().item()/labels_batch.shape[0]
        return float(acc)
     
    
    '''after using CRF, original correctness calculation is not working, becuase the output is list object now'''            
    # def get_correct_and_total_count(self, labels, outs):
    #     #[b*lens, 8] -> [b*lens]
        
    #     outs = outs.argmax(dim=1)
    #     correct = (outs == labels).sum().item()
    #     total = len(labels)

    #     #计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    #     select = labels != 0
    #     outs = outs[select]
    #     labels = labels[select]
    #     correct_content = (outs == labels).sum().item()
    #     total_content = len(labels)

    #     return correct, total, correct_content, total_content    
    
    
    def get_correct_and_total_count(self, labels, outs):
        correct, total = 0, 0
        correct_content, total_content = 0, 0
        
        # 迭代每个批次的输出和标签
        
        for batch_outs, batch_labels in zip(outs, labels):
            # 计算总标签数量
            total += len(batch_labels)
            
            # 迭代每个序列的输出和标签
            for out, label in zip(batch_outs, batch_labels):
                # 过滤掉填充标签
                if label != 0:
                    total_content += 1
                    
                    # 判断是否预测正确
                    if out == label:
                        correct_content += 1
            
            # 计算总的正确数量
            correct += sum([1 for out, label in zip(batch_outs, batch_labels) if out == label])
        
        return correct, total, correct_content, total_content
    

