#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:07:50 2018

@author: ajaver
"""
from simple_trainer import TrainerSimpleNet2

import os
import torch
from torch.autograd import Variable
import numpy as np
import multiprocessing as mp
import pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import sys
sys.path.append('../RFE_logRegNN')
from reader import read_feats
#%%

save_clf_dir = './classifiers'
_is_save_models = True

if not os.path.exists(save_clf_dir):
    os.makedirs(save_clf_dir)

def softmax_clf(data_in):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), _ = fold_data
    (cuda_id, trainer_args, _, _) = fold_param
    
    
    n_classes = int(y_train.max() + 1)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() 
    
    input_v = Variable(x_test.cuda(cuda_id), requires_grad=False)
    target_v = Variable(y_test.cuda(cuda_id), requires_grad=False)
    
    input_train = x_train
    target_train = y_train
    
    
    n_features = x_train.shape[1]
    trainer = TrainerSimpleNet2(n_classes, n_features, cuda_id=cuda_id, **trainer_args)
    
    trainer.fit(input_train, target_train)
    fold_res = trainer.evaluate(input_v, target_v)
    
    if _is_save_models:
        save_name = os.path.join(save_clf_dir, '{}_{}.pth.tar'.format(*fold_id))
        torch.save(trainer.model, save_name)
    
    print(fold_id)
    print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*fold_res))
    
    return (fold_id, fold_res)
#%%    
if __name__ == '__main__':
    pool_size = 10
    cuda_id = 1
    
#    train_args = dict(n_epochs = 250,  batch_size = 250, lr = 0.01, momentum = 0.9)
#    test_size = 0.2
#    experimental_dataset = 'SWDB'
    
    
    test_size = 1/3
    experimental_dataset = 'CeNDR'
    train_args = dict(n_epochs = 150,  batch_size = 250,  lr = 0.01, momentum = 0.9)
    
#    n_epochs = 150
#    test_size = 1/3
#    experimental_dataset = 'Syngenta'
    
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    if 'all' in feat_data:
        del feat_data['all']
        del feat_data['OW']
    #%%
    with open('reduced_feats_SWDB.txt', 'r') as fid:
        reduced_feats = fid.read().split('\n')
    #%%
    df_filt = feat_data['tierpsy']
    all_feat_cols = [x for x in df_filt.columns if not x in col2ignore_r]
    strain_dicts = {x[1]:x[0] for x in df_filt[['strain', 'strain_id']].drop_duplicates().values.tolist()}
    #%%
    random_state = 777
    n_folds = 6
    
    #dum variables keep for compatibility
    metric2exclude = 'loss'
    n_feats2remove = 'log2'
    
    all_data_in = []
    
    cols_types = {'all' : all_feat_cols, 'reduced' : reduced_feats}
    #, 'no_motion': list(set(feats_cols_filt)-set(motion_cols_filt))}
    
    #s_g = StratifiedKFold(n_splits = n_folds,  random_state=random_state)
    s_g = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=random_state)
    
    if not 'id' in df_filt:
        folds_indexes = list(s_g.split(df_filt[cols_types['all']].values, df_filt['strain_id'].values))
    else:
        #deal with augmented data, we cannot select randomly we have to assign data from the same video to different videos
        df_r = df_filt.drop_duplicates('id')
        exp_ids = df_r['id'].values
        
        gen = s_g.split(df_r[cols_types['all']].values, df_r['strain_id'].values)
        folds_indexes = []
        for train_index_r, test_index_r in gen:
            good_train = df_filt['id'].isin(exp_ids[train_index_r])
            good_test = df_filt['id'].isin(exp_ids[test_index_r])
            folds_indexes.append((good_train, good_test))
            
    
    val_data = {}
    for set_type, cols in cols_types.items():
        Xd = df_filt[cols].values
        yd = df_filt['strain_id'].values
        
        for i_fold, (train_index, test_index) in enumerate(folds_indexes):
            
            fold_param = (cuda_id, train_args, metric2exclude, n_feats2remove)
            
            x_train, y_train  = Xd[train_index].copy(), yd[train_index].copy()
            x_test, y_test  = Xd[test_index].copy(), yd[test_index].copy()
            
            fold_data = (x_train, y_train), (x_test, y_test), all_feat_cols
            fold_id = (set_type, i_fold)
            
            all_data_in.append((fold_id, fold_data, fold_param))    

    #%%
    
    
    p = mp.Pool(1)
    results = p.map(softmax_clf, all_data_in)
    #%%
    save_name = 'model_results_{}.pkl'.format(experimental_dataset)
    with open(save_name, "wb" ) as fid:
        pickle.dump((strain_dicts, results), fid)
    #%%
    save_name = 'model_results_{}.pkl'.format(experimental_dataset)
    with open(save_name, "rb" ) as fid:
        (strain_dicts, results) = pickle.load(fid)
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
   
        