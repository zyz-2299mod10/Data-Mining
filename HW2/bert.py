import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import json
import random
from preprocess import preprocessing_function

class BERTDataset(Dataset):
    def __init__(self, data_path, preprocess = True, mode='train', tokenizer='google-bert/bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mode = mode
        self.preprocess = preprocess

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        if mode == 'train':
            self.data = self.data[:int(len(self.data)*0.8)]
        elif mode == 'val':
            self.data = self.data[int(len(self.data)*0.8):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.preprocess:
            item['text'] = preprocessing_function(item['text'])

        data = ['Title: ' + item['title'] + '.', 'Comment: ' + item['text'], 'Verified purchase: ' + str(item['verified_purchase']) + '.']
        random.shuffle(data)
        data = ' '.join(data)
        output = self.tokenizer(data, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        output = {key: value.squeeze(0) for key, value in output.items()}
        if self.mode == 'infer':
            return output
        label = int(item['rating']) - 1 # for rating 1~5 -> 0~4 avoiding error
        return output, label

class bert(nn.Module):
    '''
    Fine-tuning Bert with MLP.
    '''

    def __init__(self, pretrained_type):
        super().__init__()

        num_labels = 5 # ratin 1~5
        self.pretrained_model = AutoModel.from_pretrained(pretrained_type)
        
        hidden_size = self.pretrained_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
            nn.Softmax(dim = 1)
        )

    def forward(self, input_id, mask):
        outputs = self.pretrained_model(input_ids=input_id, attention_mask=mask).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        logits = self.classifier(pretrained_output)

        return logits

class BERT():
    def __init__(self, pretrained_type, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_type)
        self.model = bert(pretrained_type).to(config['device'])

    def forward(self, input_id, mask):        
        outputs = self.model(input_id=input_id, mask=mask)
        return outputs

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_weights(self, pretrain_path):
        self.model.load_state_dict(torch.load(pretrain_path))