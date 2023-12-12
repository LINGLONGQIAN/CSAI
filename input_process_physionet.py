# coding: utf-8
import os
import re
import numpy as np
import pandas as pd
import json
import copy
from sklearn.model_selection import KFold
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle

patient_ids = []

for filename in os.listdir('./physionet_raw'):
    # the patient data in PhysioNet contains 6-digits
    match = re.search('\d{6}', filename)
    if match:
        id_ = match.group()
        patient_ids.append(id_)

no_data = ['141264', '140936', '140501']
patient_ids = [x for x in patient_ids if x not in no_data]

out = pd.read_csv('./physionet_raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def to_time_bin(x):
    h, m = map(int, x.split(':'))
    return h

def parse_data(x):
    x = x.set_index('Parameter')['Value']
    values = []

    for attr in attributes:
        if attr in list(x.index):
            values.append(x[attr].mean())
        else:
            values.append(np.nan)
    return values

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def generate_data(id_list):
    if len(id_list)>0:
        dataset = []
        labels = []
        for id_ in tqdm(id_list):
            try:
                data = pd.read_csv('./physionet_raw/{}.txt'.format(patient_ids[id_]))
                print('Patient {} data siez: {}'.format(patient_ids[id_]), data.shape())
                # accumulate the records within one hour
                data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))
                values = []
                for h in range(48):
                    values.append(parse_data(data[data['Time'] == h]))
                
                labels.append(out.loc[int(patient_ids[id_])])
                dataset.append(np.array(values))
            except Exception as e:
                print(e)
                print('Error with data!', patient_ids[id_])
                continue
        return np.array(dataset), np.array(labels)
    else:
        print('Error with dataset split!')

kfold_data = []
kfold_label = []
kf = KFold(n_splits=5, shuffle=True, random_state=3407)
for ind, (train_eval_id , test_id) in enumerate(kf.split(patient_ids)):
    print('Deal with the {} fold data'.format(str(ind)))
    random.shuffle(train_eval_id)
    eval_id = train_eval_id[0:len(test_id)]
    train_id = train_eval_id[len(test_id):]

    train_data, train_label = generate_data(train_id)
    eval_data, eval_label = generate_data(eval_id)
    test_data, test_label = generate_data(test_id)
    kfold_data.append([train_data, eval_data, test_data])
    kfold_label.append([train_label, eval_label, test_label])

pickle.dump(kfold_data, open('./data/physionet/data_nan.pkl', 'wb'), -1)
pickle.dump(kfold_label, open('./data/physionet/label.pkl', 'wb'), -1)

