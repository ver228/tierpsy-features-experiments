#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:07:50 2018

@author: ajaver
"""
import os
import numpy as np
import multiprocessing as mp
import pickle

import sys
sys.path.append('../../helper')
from misc import results_root_dir
from reader import read_feats
from process_simple import get_softmax_clf, fold_generator

#%%    
if __name__ == '__main__':
    pool_size = 10
    cuda_id = None
    
    #%%
    train_args = dict(n_epochs = 250,  batch_size = 250, lr = 0.01, momentum = 0.9)
    test_size = 0.2
    experimental_dataset = 'SWDB'
    
#    test_size = 1/3
#    experimental_dataset = 'CeNDR'
#    train_args = dict(n_epochs = 150,  batch_size = 250,  lr = 0.01, momentum = 0.9)

#    test_size = 1/3
#    experimental_dataset = 'MMP'
#    train_args = dict(n_epochs = 40,  batch_size = 150,  lr = 0.01, momentum = 0.9)
    
#    test_size = 1/3
#    experimental_dataset = 'Syngenta'
    #%% #read the selected set of features 
    dname = os.path.join(results_root_dir, 'RFE')
    bn = 'reduced_feats_{}.txt'.format(experimental_dataset)
    with open(os.path.join(dname, bn), 'r') as fid:
        reduced_feats = fid.read().split('\n')
    
    #%%
    results_dir = os.path.join(results_root_dir, 'simple')
    bn = 'model_results_reduced_{}.pkl'.format(experimental_dataset)
    save_name = os.path.join(results_dir, bn)
    #%%
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    if 'all' in feat_data:
        del feat_data['all']
        del feat_data['OW']
    #%%
    df_filt = feat_data['tierpsy']
    all_feat_cols = [x for x in df_filt.columns if not x in col2ignore_r]
    strain_dicts = {x[1]:x[0] for x in df_filt[['strain', 'strain_id']].drop_duplicates().values.tolist()}
    cols_types = {'all' : all_feat_cols, 'reduced' : reduced_feats}
    
    #%%
    random_state = 777
    n_folds = 6
    
    #dum variables keep for compatibility
    metric2exclude = 'loss'
    n_feats2remove = 'log2'
    
    fold_param = (cuda_id, train_args, metric2exclude, n_feats2remove)
    all_data_in = fold_generator(df_filt, cols_types, n_folds, test_size, fold_param)
    #%%
    p = mp.Pool(pool_size)
    results = p.map(get_softmax_clf, all_data_in)
    #%%
    with open(save_name, "wb" ) as fid:
        pickle.dump((strain_dicts, results), fid)
    
    #%%
    res_db = {}
    for (set_type, i_fold), dat in results:
        if set_type not in res_db:
            res_db[set_type] = []
        res_db[set_type].append(dat)
    
    for set_type, dat in res_db.items():
        res_db[set_type] = list(zip(*dat))
        
    for n, m_type in enumerate(['loss', 'acc', 'f1']):
        if m_type == 'loss':
            continue
        
        print(m_type, '**************')
        for set_type, dat in res_db.items():
            vv = dat[n]
            
            dd = '{} : {:.2f} {:.2f}'.format(set_type, np.mean(dat[n]), np.std(dat[n]))
            print(dd)
   
        