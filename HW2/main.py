from tqdm import tqdm
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bert import BERT, BERTDataset

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
                        help="train or inference")

    args = opt.parse_args()
    return args

if __name__ == '__main__':
    args = get_argument()

    train_path = './data/train.json'
    test_path = './data/test.json'
    save_model_path = './model_nopre/050702'

    os.makedirs(save_model_path, exist_ok=True)

    if args.mode not in ['train', 'infer']:
        raise AssertionError('train or infer')

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



