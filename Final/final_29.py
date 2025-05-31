import pandas as pd
import torch
import os
import copy
import json

from typing import List
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
MAX_SIZE = 1e6

def data_preprocess(mode, verbose=False):
    assert mode in ['train', 'test']
    behaviors_path = f"../data/{mode}/{mode}_behaviors.tsv"
    news_path = f"../data/{mode}/{mode}_news.tsv"
    embed_path = f"../data/{mode}/{mode}_entity_embedding.vec"
    
    behaviors = pd.read_csv(behaviors_path, sep='\t')
    news = pd.read_csv(news_path, sep='\t')
    with open(embed_path, 'r') as vecfile:  
        embeds = {line.split('\t')[0]: [float(x) for x in line.split('\t')[1:-1]] for line in vecfile.readlines()}
    
    title_embeds = []
    abstract_embeds = []
    
    for title_json_str in news['title_entities']:
        if type(title_json_str) == type(''):
            title_embeds.append(json.dumps([embeds.get(title_entity['WikidataId'], None) for title_entity in json.loads(title_json_str)]))
        else:
            title_embeds.append(json.dumps([]))
    for abstract_json_str in news['abstract_entities']:
        if type(abstract_json_str) == type(''):
            abstract_embeds.append(json.dumps([embeds.get(abstract_entity['WikidataId'], None) for abstract_entity in json.loads(abstract_json_str)]))
        else:
            abstract_embeds.append(json.dumps([]))
    
    news['title_embeds'] = title_embeds
    news['abstract_embeds'] = abstract_embeds
    news.drop(columns=['title_entities', 'abstract_entities', 'title', 'abstract', 'URL'], inplace=True)
    news.to_csv(f'../data/{mode}/{mode}_news_processed.tsv', index=False, sep='\t')
    
    verbose and print(len(behaviors), len(news), len(embeds))
    verbose and print(behaviors)
    verbose and print(news)
 
def read_tsv(mode):
    assert mode in ['train', 'test']
    behaviors = pd.read_csv(f'../data/{mode}/{mode}_behaviors.tsv', sep='\t')
    new_processed = pd.read_csv(f'../data/{mode}/{mode}_news_processed.tsv', sep='\t')
    
    return behaviors, new_processed

def parse_str_tensor(lists_str: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if lists_str == '[]' or lists_str == '[null]':
        return []
    
    embed_list = []
    try:
        lists_str = lists_str[1:-1].replace('null', '[]').replace(' ', '').replace('],[', '|').replace('[', '').replace(']', '')
        lists_str = lists_str.split('|')
        for list_str in lists_str:
            embed_list.append([float(x) for x in list_str.split(',')])
    except:
        return []
    return torch.tensor(embed_list, dtype=torch.float32, device=device)   

def calculate_cosine_similarity(news_id1, news_id2, news_embed_dict: dict) -> float:
    vec1_list = news_embed_dict.get(news_id1, [])
    vec2_list = news_embed_dict.get(news_id2, [])
    if len(vec1_list) == 0 or len(vec2_list) == 0:
        return -1
    cnt = 0
    batch = torch.zeros((2, len(vec1_list) * len(vec2_list), len(vec1_list[0])))
    for vec1 in vec1_list:
        for vec2 in vec2_list:
            batch[0, cnt] = vec1
            batch[1, cnt] = vec2
            cnt += 1
    cos_sim = cosine_similarity(batch[0], batch[1])
    return torch.mean(cos_sim).item()
    return cosine_similarity(torch.mean(vec1_list), torch.mean(vec2_list), dim=0).item()

def calcaulate_embed_sim(behaviors: pd.DataFrame, news_embed_dict_list, mode):
    assert mode in ['train', 'test']
    new_data = []
    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
        clicked_news = row['clicked_news']
        impressions = row['impressions']
        
                                
        for impression in impressions.split():
            impression = impression.split('-')
            if mode == 'train' and len(impression) != 2: continue
            
            impression = impression[0]
            
            total_title_sim = 0
            total_abstract_sim = 0
            for clicked_new in clicked_news.split():
                total_title_sim += calculate_cosine_similarity(clicked_new, impression, news_embed_dict_list[0]) 
                total_abstract_sim += calculate_cosine_similarity(clicked_new, impression, news_embed_dict_list[1]) 
            total_title_sim /= len(clicked_news)
            total_abstract_sim /= len(clicked_news)
            new_data.append([total_title_sim, total_abstract_sim])
        if len(new_data) > MAX_SIZE:
            break
    return pd.DataFrame(new_data, columns=['title_sim', 'abstract_sim'])

def count_subcategory(behaviors: pd.DataFrame,
                      colunms: str, 
                      newsId_2_sub: dict,
                      sub_2_col: dict,
                      mode: str,
                      split=False) -> pd.DataFrame: 
    '''
    return
        pd.DataFrame
    '''
    assert mode in ['train', 'test']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    new_data = []
    
    if mode == 'train':    
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
            clicked_news = row['clicked_news']
            impressions = row['impressions']
            
            tmp_row = torch.zeros((1 if not split else 2, len(colunms)), dtype=torch.uint8, device=device)
            
            tmp_click = [newsId_2_sub.get(clicked_new, None) for clicked_new in clicked_news.split()]
            tmp_click_idx = [sub_2_col.get(sub, None) for sub in tmp_click]
            
            for sub_idx in tmp_click_idx:
                tmp_row[0, sub_idx] += 1
                                    
            for impression in impressions.split():
                impression = impression.split('-')
                if len(impression) != 2: continue
                
                _impression, clicked = impression
                impression_sub = newsId_2_sub.get(_impression, None)
                impression_sub_idx = sub_2_col.get(impression_sub, None)
                if impression_sub_idx is None: continue
                
                impre_row = copy.deepcopy(tmp_row) 
                impre_row[0 if not split else 1, impression_sub_idx] += 1 
                impre_row[:, -1] = int(clicked) 
                
                new_data.append(impre_row[0].tolist() if not split else impre_row[0].tolist()[:-1] + impre_row[1].tolist())

            if len(new_data) > MAX_SIZE:
                break
        new_data = pd.DataFrame(new_data, columns=colunms if not split else colunms[:-1] + colunms)
            
    elif mode == 'test':
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
            clicked_news = row['clicked_news']
            impressions = row['impressions']
            
            tmp_row = torch.zeros((1 if not split else 2, len(colunms)), dtype=torch.uint8, device=device)
            
            tmp_click = [newsId_2_sub.get(clicked_new, None) for clicked_new in clicked_news.split()]
            tmp_click_idx = [sub_2_col.get(sub, None) for sub in tmp_click]
            for sub_idx in tmp_click_idx:
                tmp_row[0, sub_idx] += 1               
            for impression in impressions.split(): 
                impression_sub = newsId_2_sub.get(impression, None)
                impression_sub_idx = sub_2_col.get(impression_sub, None)
                if impression_sub_idx is None: continue
                
                impre_row = copy.deepcopy(tmp_row) 
                impre_row[0 if not split else 1, impression_sub_idx] += 1
                
                new_data.append(impre_row[0].tolist()  if not split else impre_row[0].tolist() + impre_row[1].tolist())
            
            if len(new_data) > MAX_SIZE:
                break
            
        new_data = pd.DataFrame(new_data, columns=colunms if not split else colunms + colunms)
    
    return new_data 

def preprocess_data(training_split_num = 3):
    '''
    preprocess data by subcategory: count the history and impression subcategory times in each ids
    
    output:
        .csv [row: ids, col: subcategory, label: 0 or 1]
    '''    
    # read tsv
    train_behaviors, train_new_processed = read_tsv('train')
    test_behaviors, test_new_processed = read_tsv('test')
    
    train_new_processed['title_embeds'] = train_new_processed['title_embeds'].apply(lambda lists_str: parse_str_tensor(lists_str))
    train_new_processed['abstract_embeds'] = train_new_processed['abstract_embeds'].apply(lambda lists_str: parse_str_tensor(lists_str))
    
    test_new_processed['title_embeds'] = test_new_processed['title_embeds'].apply(lambda lists_str: parse_str_tensor(lists_str))
    test_new_processed['abstract_embeds'] = test_new_processed['abstract_embeds'].apply(lambda lists_str: parse_str_tensor(lists_str))
    
    # create new data
    train_cat = train_new_processed['category'].unique() # 18-D
    train_sub = train_new_processed['subcategory'].unique() # 285-D
    columns_sub = train_sub.tolist() + ['label']
    columns_cat = train_cat.tolist() + ['label']
        
    # news id to subcategory & sub id in columns
    train_newsId_2_sub = {}
    train_newsId_2_cat = {}
    train_newsId_2_title_embed = {}
    train_newsId_2_abstract_embed = {}
    test_newsId_2_sub = {}
    test_newsId_2_cat = {}
    test_newsId_2_title_embed = {}
    test_newsId_2_abstract_embed = {}
    for i in range(train_new_processed.shape[0]):
        train_newsId_2_sub[train_new_processed.loc[i, 'news_id']] = train_new_processed.loc[i, 'subcategory']
        train_newsId_2_cat[train_new_processed.loc[i, 'news_id']] = train_new_processed.loc[i, 'category']
        train_newsId_2_title_embed[train_new_processed.loc[i, 'news_id']] = train_new_processed.loc[i, 'title_embeds']
        train_newsId_2_abstract_embed[train_new_processed.loc[i, 'news_id']] = train_new_processed.loc[i, 'abstract_embeds']
    for i in range(test_new_processed.shape[0]):
        test_newsId_2_sub[test_new_processed.loc[i, 'news_id']] = test_new_processed.loc[i, 'subcategory']
        test_newsId_2_cat[test_new_processed.loc[i, 'news_id']] = test_new_processed.loc[i, 'category']
        test_newsId_2_title_embed[test_new_processed.loc[i, 'news_id']] = test_new_processed.loc[i, 'title_embeds']
        test_newsId_2_abstract_embed[test_new_processed.loc[i, 'news_id']] = test_new_processed.loc[i, 'abstract_embeds']
        
    sub_2_col = {}
    cat_2_col = {}
    for i in train_sub.tolist():
        sub_2_col[i] = columns_sub.index(i)
    for i in train_cat.tolist():
        cat_2_col[i] = columns_cat.index(i)
    # training preprocess (full training data is too large)
    chunk = train_behaviors.shape[0] // training_split_num if training_split_num != 0 else 0
    for i in range(training_split_num):
        print(f"Processing {i}th training data.....")
        chunk_train_behaviors = train_behaviors.iloc[i * chunk: (i + 1) * chunk, :]
        train_embed_sim = calcaulate_embed_sim(chunk_train_behaviors, [train_newsId_2_title_embed, train_newsId_2_abstract_embed], 'train')
        new_train_data_cat = count_subcategory(behaviors=chunk_train_behaviors, colunms=columns_cat,
                                       newsId_2_sub=train_newsId_2_cat, sub_2_col=cat_2_col,
                                       mode='train', split=True).drop(columns='label')
        new_train_data_sub = count_subcategory(behaviors=chunk_train_behaviors, colunms=columns_sub,
                                       newsId_2_sub=train_newsId_2_sub, sub_2_col=sub_2_col,
                                       mode='train',split=True)
        print(new_train_data_cat.shape)
        print(new_train_data_sub.shape)
        print(train_embed_sim.shape)
        new_train_data = pd.concat([new_train_data_sub, new_train_data_cat, train_embed_sim], join='inner', axis=1)
        print(new_train_data.shape)
        new_train_data.to_csv(f"./new_data/add_embed/new_train_data{i}.csv", index=False)
   
    # testing preprocess
    print(f"Processing testing data.....")
    test_embed_sim = calcaulate_embed_sim(test_behaviors, [test_newsId_2_title_embed, test_newsId_2_abstract_embed], 'test')
    new_test_data_cat = count_subcategory(behaviors=test_behaviors, colunms=columns_cat[:-1],
                                      newsId_2_sub=test_newsId_2_cat, sub_2_col=cat_2_col,
                                      mode='test', split=True)
    new_test_data_sub = count_subcategory(behaviors=test_behaviors, colunms=columns_sub[:-1],
                                      newsId_2_sub=test_newsId_2_sub, sub_2_col=sub_2_col,
                                      mode='test', split=True)
    print(new_test_data_cat.shape)
    print(new_test_data_sub.shape)
    print(test_embed_sim.shape)
    new_test_data = pd.concat([new_test_data_sub, new_test_data_cat, test_embed_sim], join='inner', axis=1)
    print(new_test_data.shape)
    new_test_data.to_csv(f"./new_data/add_embed/new_test_data.csv", index=False) 

def train(data_idx):
    train_data = pd.read_csv(f'./new_data/add_embed/new_train_data{data_idx}.csv')
    
    X = train_data.drop(columns=['label'], axis=1)
    y = train_data['label'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=0)
    print("Finish data prepare")
    
    model = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='Logloss',
                               eval_metric='AUC')
    
    if data_idx != 0:
        print('Loading model......')
        model.load_model('./model/model.pth')
    model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=100)
    model.save_model('./model/model.pth')

def test():
    model = CatBoostClassifier()
    model.load_model('./model/model.pth')

    test_data = pd.read_csv('./new_data/add_embed/new_test_data.csv')
    print("Finish data prepare")
    
    predictions = model.predict_proba(test_data)[:, 1]

    num_per_row = 15
    num_rows = len(predictions) // num_per_row
    reshaped_predictions = predictions[:num_rows * num_per_row].reshape((num_rows, num_per_row))

    columns = ['p' + str(i+1) for i in range(num_per_row)]
    results = pd.DataFrame(reshaped_predictions, columns=columns)
    
    filenames = [path[-1] for path in list(os.walk('./prediction/catboost'))]
    if len(filenames[0]) == 0:
        v = -1 
    else:
        v = max([int(filename.replace('prediction', '').replace('.csv', '')) for filename in filenames[0]])
    print(f'Saving prediction to ./prediction/catboost/prediction{v + 1}.csv ......')
    results.to_csv(f'./prediction/catboost/prediction{v + 1}.csv', index_label='id')

if __name__ == '__main__': 
    data_preprocess('train')
    data_preprocess('test')
    preprocess_data()
    
    filenames = [path[-1] for path in list(os.walk('./new_data/split_train'))]
    filenames[0].remove('new_test_data.csv')
    v = max([int(filename.replace('new_train_data', '').replace('.csv', '')) for filename in filenames[0]])
    for i in range(v+1):
        print(f'Now train {i}th training data......')
        train(i)
    
    print('Inference......')
    test()