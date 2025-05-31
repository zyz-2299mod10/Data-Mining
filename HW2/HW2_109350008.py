from tqdm import tqdm
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
import random
from preprocess import preprocessing_function

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import re

def remove_stopwords(text: str) -> str:
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def remove_HTML(text: str) -> str:
    rmHTML = re.sub("<[^>]+>", "", text).strip()

    return rmHTML

def lemmatization(text: str) -> str:
    filter = nltk.stem.wordnet.WordNetLemmatizer()
    text = text.split(" ")
    lemed = []
    for i in text:
        lem = filter.lemmatize(i, "n")
        if(lem == i): lem = filter.lemmatize(i, "v")
        lemed.append(lem)
    
    preprocessed_text = ' '.join(lemed)

    return preprocessed_text
    

def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    preprocessed_text = remove_HTML(preprocessed_text)
    preprocessed_text = lemmatization(preprocessed_text)

    return preprocessed_text

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

def agent(save_path, pretrain_path, bert_config, data_path, mode = 'train', data_preprocess = True, model_mode = 'distil'):
    if model_mode == 'distil':
        model = BERT('distilbert/distilbert-base-uncased', config = bert_config)
    elif model_mode == 'bert':
        model = BERT('google-bert/bert-base-uncased', config = bert_config)

    if pretrain_path != None:
        model.load_weights(pretrain_path=pretrain_path)

    if mode == 'train':
        train_data = BERTDataset(data_path=data_path, preprocess=data_preprocess, mode='train')
        test_data = BERTDataset(data_path=data_path, preprocess=data_preprocess, mode='val')
        train_dataloader = DataLoader(train_data, batch_size= bert_config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=bert_config['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=model.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

        train(
            save_path=save_path,
            model=model,
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=nn.CrossEntropyLoss().to(model.config['device']),
            config=model.config
            )
    
    if mode == 'infer':
        test_data = BERTDataset(data_path=data_path, preprocess=data_preprocess, mode='infer')
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        model.eval()
        results = []
        with torch.no_grad():
            for input in tqdm(test_dataloader):
                input_id = input['input_ids'].to(model.config['device'])
                mask = input['attention_mask'].to(model.config['device'])

                outputs = model.forward(input_id, mask).squeeze(-1).to(model.config['device'])
                outputs = torch.argmax(outputs, dim = 1)
                results.extend(outputs.tolist())
                # print(results)
        return results


def train(save_path, model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, config):
    last_best_model_acc = 0

    for epoch in range(config['epochs']):
        train_loss = 0
        # training stage
        model.train()
        for batch in tqdm(train_dataloader):
            inputs, targets = batch

            input_ids = inputs['input_ids'].to(config['device'])
            mask = inputs['attention_mask'].to(config['device'])
            targets = targets.to(config['device'])

            optimizer.zero_grad()
            outputs = model.forward(input_ids, mask).squeeze(-1).to(config['device'])
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        
        # testing stage
        val_loss = 0
        val_acc = 0
        targets, outputs = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader):
                inputs, targets = batch
                input_ids = inputs['input_ids'].to(config['device'])
                mask = inputs['attention_mask'].to(config['device'])
                targets = targets.to(config['device'])
                outputs = model.forward(input_ids, mask).squeeze(-1).to(config['device'])

                loss = loss_fn(outputs, targets)

                outputs = torch.argmax(outputs, dim = 1)
                targets = targets.cpu().numpy()
                outputs = outputs.cpu().numpy()

                val_acc += (outputs == targets).sum()
                val_loss += loss.item()

        train_avg_loss = round(train_loss/len(train_dataloader), 4)
        val_avg_loss = round(val_loss/len(test_dataloader), 4)
        acc = round(val_acc/(len(test_dataloader)*config['batch_size']), 4)
        print(f"Epoch: {epoch}, Training loss: {train_avg_loss}, Val loss: {val_avg_loss}, Accuracy: {acc}")

        if (epoch+1) % 2 == 0:
            # save model
            model.save_model(save_path = os.path.join(save_path, 'epoch_' + str(epoch) + '.pth'))
        
        if last_best_model_acc < acc :
            # save best acc model
            last_best_model_acc = acc
            model.save_model(save_path = os.path.join(save_path, 'best.pth')) 

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--mode",
                        type=str,
                        choices=['train', 'infer']
                        help="train or inference")

    args = opt.parse_args()
    return args

if __name__ == '__main__':
    args = get_argument()

    train_path = './data/train.json'
    test_path = './data/test.json'
    save_model_path = './model_nopre/050702'

    os.makedirs(save_model_path, exist_ok=True)

    bert_config = {
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    }
    # training
    if args.mode == 'train':
        print("training....... ")
        agent(save_path=save_model_path, pretrain_path=None, bert_config=bert_config, data_path=train_path, mode='train', data_preprocess=False, model_mode='distil')

    # inference
    if args.mode == 'infer':
        print("inference.......")
        result = agent(save_path=save_model_path, pretrain_path='./model_nopre/050702/best.pth', bert_config=bert_config, data_path=test_path, mode='infer', data_preprocess=False)
        # print("result: ", result)
        with open('submission.csv', 'w') as f:
            f.write('index,rating\n')
            for i, ans in enumerate(result):
                f.write('index_' + str(i) + ',' + str(int(ans) + 1) + '\n')



