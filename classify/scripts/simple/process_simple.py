#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver
"""
import os
import torch
from torch.autograd import Variable
from sklearn.model_selection import StratifiedShuffleSplit

from trainer import TrainerSimpleNet

import sys
sys.path.append('../../helper')

def get_softmax_clf(data_in, save_clf_dir = None):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), _ = fold_data
    (cuda_id, trainer_args, _, _) = fold_param
    
    
    n_classes = int(y_train.max() + 1)
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() 
    
    if cuda_id is not None:    
        x_test = x_test.cuda(cuda_id)
        y_test = y_test.cuda(cuda_id)
        
    input_v = Variable(x_test, requires_grad=False)
    target_v = Variable(y_test, requires_grad=False)
    
    input_train = x_train
    target_train = y_train
    
    
    n_features = x_train.shape[1]
    trainer = TrainerSimpleNet(n_classes, n_features, cuda_id=cuda_id, **trainer_args)
    
    trainer.fit(input_train, target_train)
    fold_res = trainer.evaluate(input_v, target_v)
    
    if save_clf_dir is not None:
        if not os.path.exists(save_clf_dir):
            os.makedirs(save_clf_dir)
        save_name = os.path.join(save_clf_dir, '{}_{}.pth.tar'.format(*fold_id))
        torch.save(trainer.model, save_name)
    
    print(fold_id)
    print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*fold_res))
    
    return (fold_id, fold_res)

def fold_generator(df_filt,
                    cols_types,
                    n_folds, 
                    test_size, 
                    fold_param,
                    random_state=777):
    
    assert 'all' in cols_types
    
    s_g = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=random_state)
    
    
    all_feat_cols = cols_types['all']
    
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
            
    for set_type, cols in cols_types.items():
        Xd = df_filt[cols].values
        yd = df_filt['strain_id'].values
        
        for i_fold, (train_index, test_index) in enumerate(folds_indexes):
            
            
            x_train, y_train  = Xd[train_index].copy(), yd[train_index].copy()
            x_test, y_test  = Xd[test_index].copy(), yd[test_index].copy()
            
            fold_data = (x_train, y_train), (x_test, y_test), all_feat_cols
            fold_id = (set_type, i_fold)
            
            yield (fold_id, fold_data, fold_param)