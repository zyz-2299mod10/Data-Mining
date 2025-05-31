import pandas as pd
import numpy as np
import csv
import random
data = pd.read_csv('training.csv').drop(columns=['lettr']).to_numpy()

data_pred = pd.read_csv('test_X.csv').to_numpy()

out_dict = {'id':[], 'outliers':[]}

for i, data_pred in enumerate(data_pred):
    smallest = 100000
    for j, data_train in enumerate(data):
        dist = np.linalg.norm(data_pred - data_train)
        if dist ==0 and smallest <=0:
            smallest -= 1
        if dist == 0:
            print(f'Found exact match at {i} and {j//700}')
        if dist < smallest:
            smallest = dist
    if smallest == 0:
        smallest = smallest - random.uniform(0.00001, 0.00002)
    out_dict['id'].append(i)
    out_dict['outliers'].append(smallest)
    
count = 0
for i in out_dict['outliers']:
    if i < 0.0001:
        count += 1
print(f'Found {count} exact matches')
print(len(out_dict['id']))

df = pd.DataFrame.from_dict(out_dict, orient='columns')
df.to_csv('predicted.csv', index=False)
    