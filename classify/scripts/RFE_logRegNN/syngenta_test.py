#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:20 2018

@author: ajaver
"""
import os
import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit

from trainer import softmax_RFE

col2ignore = ['Unnamed: 0', 'id', 'strain', 'directory', 'base_name', 'exp_name',
              'MOA general', 'MOA specific', 'CSN', 'Quantity available', 'MW', 'label_id']

def _scale_feats(df):
    col_val = [x for x in df.columns if x not in col2ignore]
    dd = df[col_val]
    z = (dd-dd.mean())/(dd.std())
    df[col_val] = z

def fold_generator(fold_param, labels, lab_str2id):
    for lab in labels:
        print(lab)
        col_feats = [x for x in feats.columns if x not in col2ignore]
        ss = np.sort(feats[lab].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['label_id'] = feats[lab].map(s_dict)
        lab_str2id[lab] = s_dict
    
        
        
        feats_r = feats.drop_duplicates('id')
        y_r = feats_r[lab].values
        exp_ids = feats_r['id'].values
        for i_fold, (train_index, test_index) in enumerate(cv.split(exp_ids, y_r)):
            good_train = feats['id'].isin(exp_ids[train_index])
            y_train = feats.loc[good_train, 'label_id'].values.copy()
            x_train = feats.loc[good_train, col_feats].values.copy()
            
            good_test = feats['id'].isin(exp_ids[test_index])
            y_test = feats.loc[good_test, 'label_id'].values.copy()
            x_test = feats.loc[good_test, col_feats].values.copy()
            
            
            fold_data = (x_train, y_train), (x_test, y_test), col_feats.copy()
            fold_id = (lab, i_fold)
            
            yield (fold_id, fold_data, fold_param)
        
            print(len(set(y_train)), len(set(y_test)))
    
    

if __name__ == '__main__':
    
    cuda_id = 0
    batch_size = 250
    
    metric2exclude = 'loss'
    n_feats2remove = 'log2'
    
    pool_size = 10
    n_epochs = 150
    test_size = 1/3
    n_folds = 10
    
    experimental_dataset = 'Syngenta'
    
    fold_param = (cuda_id, n_epochs, batch_size, metric2exclude, n_feats2remove)
    labels = ['MOA general', 'MOA specific', 'CSN']
    
    save_dir = '../../data/{}'.format(experimental_dataset)
    bn = 'F0.025_tierspy_features_augmented_{}.csv'.format(experimental_dataset)
    
    fname = os.path.join(save_dir, bn)
    df = pd.read_csv(fname)
    df.index = df['base_name'].apply(lambda x : x.split('_')[2])
    
    
    fname_l = os.path.join(save_dir, 'compounds_actions.xlsx')
    df_l = pd.read_excel(fname_l)
    df_l.index = df_l['CSN'].str.lower()
    
    feats = df.join(df_l)
    feats.loc['no', ['Quantity available', 'MW']] = 0
    feats.loc['no', ['MOA general', 'MOA specific', 'CSN']] = 'N/A'
    
    bad = feats['MOA specific'].isnull()
    feats.loc[bad, 'MOA specific'] = feats.loc[bad, 'MOA general']
    
    _scale_feats(feats)
    
    
    cv = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=777)
    
    lab_str2id = {}
    gen = fold_generator(fold_param, labels, lab_str2id)
    all_data_in = []
    results = []
    p = mp.Pool(pool_size)
    for dat in iter(gen):
        all_data_in.append(dat)
        if len(all_data_in) == pool_size:
            results += p.map(softmax_RFE, all_data_in)
            all_data_in = []
    results += p.map(softmax_RFE, all_data_in)
    
    
    save_name = 'SyngentaLabeled.pkl'
    with open(save_name, "wb" ) as fid:
        pickle.dump(results, fid)
    #%%
    with open(save_name, "rb" ) as fid:
            results = pickle.load(fid)
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        res_db[db_name].append((feats, loss, acc, f1))
    
    for set_type, dat in res_db.items():
        res_db[set_type] = list(zip(*dat))
    #%%
    for n, m_type in enumerate(['loss', 'acc', 'f1']):
        print(m_type)
        for set_type, dat in res_db.items():
            vv = dat[n + 1]
            best_vv = np.max(vv, axis=1)
            dd = 'Best {} : {:.2f} {:.2f}'.format(set_type, np.mean(best_vv), np.std(best_vv))
            print(dd)
    #%%
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(1, 1)
    for db_name, dat in res_db.items():
        val = dat[3]
        feats = dat[0]
        
        
        
        yy = np.mean(val,axis=0)
        err = np.std(val,axis=0)
        
        tot = len(sum(feats[0], []))
        n2 = int(np.floor(np.log2(tot - 1e-5)))
        xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
        
        print(db_name, np.max(yy))
        
        h = ax.errorbar(xx, yy, yerr=err, label = db_name)
    plt.legend()