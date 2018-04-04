#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver
"""
import torch
import numpy as np
from torch.autograd import Variable

import sys
sys.path.append('../../helper')
from trainer import TrainerSimpleNet, SimpleNet

def remove_feats(importance_metrics, 
                 metric2exclude, 
                 input_v, 
                 input_train, 
                 col_feats_o,
                 n_feats2remove):
    metrics = importance_metrics[metric2exclude]
    
    ind_orders = np.argsort(metrics)
    
    #remove the least important feature
    if metric2exclude != 'loss':
        ind_orders = ind_orders[::-1]
    
    ind2exclude = ind_orders[:n_feats2remove].tolist()
    
    n_features = input_v.size(1)
    ind_r = [x for x in range(n_features) if x not in ind2exclude]
    
    input_r = input_v[:, ind_r]
    input_train_r = input_train[:, ind_r]
    
    
    col_feats_r, feat2exclude = [], []
    for ii, f in enumerate(col_feats_o):
        if ii in ind2exclude:
            feat2exclude.append(f)
        else:
            col_feats_r.append(f)
    
    return input_r, input_train_r, col_feats_r, feat2exclude

def softmax_RFE(data_in):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), col_feats_r = fold_data
    (cuda_id, trainer_args, metric2exclude, n_feats2remove) = fold_param
    
    
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
    
    fold_res = []
    while len(col_feats_r)>1: #continue as long as there are any feature to remove
        
        n_features = input_v.size(1)
        trainer = TrainerSimpleNet(n_classes, n_features, cuda_id=cuda_id, **trainer_args)
        trainer.fit(input_train, target_train)
        
        loss, acc, f1, y_test_l, y_pred_l = trainer.evaluate(input_v, target_v)
        res = (loss, acc, f1)
        
        print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
        print(db_name, metric2exclude, i_fold + 1, n_features)
        
        if n_features > 1:
            if n_feats2remove == 'log2':
                n2remove =  n_features - int(2**np.floor(np.log2(n_features - 1e-5))) #lowest power of 2
            else:
                n2remove = n_feats2remove
            
            
            importance_metrics = trainer.get_feat_importance(input_v, target_v)
            
            input_v, input_train, col_feats_r, feat2remove = \
            remove_feats(importance_metrics, 
                         metric2exclude, 
                         input_v, 
                         input_train, 
                         col_feats_r, 
                         n_feats2remove = n2remove)
            fold_res.append((feat2remove, res))
        else:
            fold_res.append((col_feats_r[0], res))
    
    return fold_id, fold_res
#%%
def get_core_features(feat_data, col2ignore_r, is_expanded = True):
    def _remove_end(col_v, p2rev):
        col_v_f = []
        for x in col_v:
            xf = x
            for p in p2rev:
                if x.endswith(p):
                    xf = xf[:-len(p)]
                    break
            col_v_f.append(xf)
        
        return list(set(col_v_f))
    
    # obtain the core features from the feature list
    core_feats = {}
    
    if 'OW' in feat_data:
        col_v = [x for x in feat_data['OW'].columns if x not in col2ignore_r]
        col_v = _remove_end(col_v, ['_abs', '_neg', '_pos'])
        col_v = _remove_end(col_v, ['_paused', '_forward', '_backward'])
        col_v = _remove_end(col_v, ['_distance', '_distance_ratio', '_frequency', '_time', '_time_ratio'])
        core_feats['OW'] = sorted(col_v)
        
    if 'tierpsy' in feat_data:
        col_v = [x for x in feat_data['tierpsy'].columns if x not in col2ignore_r]
        col_v = _remove_end(col_v, ['_10th', '_50th', '_90th', '_95th', '_IQR'])
        
        col_v = _remove_end(col_v, ['_frequency', '_fraction', '_duration', ])
        
            
        #the other is important, this must be at the end
        if not is_expanded:
            col_v = list(set([x[2:] if x.startswith('d_') else x for x in col_v]))
            col_v = _remove_end(col_v, ['_abs'])
            col_v = _remove_end(col_v, ['_norm'])
            col_v = _remove_end(col_v, ['_w_forward', '_w_backward', '_w_paused'])
            
        
        core_feats['tierpsy'] = sorted(col_v)
    
    if ('OW' in feat_data) and ('tierpsy' in feat_data):
        #all
        core_feats['all']  = core_feats['tierpsy'] + ['ow_' + x for x in core_feats['OW']]
    
    return core_feats
#%%
def get_feat_group_indexes(core_feats_v, col_feats):
    '''
    Get the keys, i am assuming there is a core_feats for each of the col_feats
    '''
    
    c_feats_dict = {x:ii for ii, x in enumerate(core_feats_v)}
    
    #sort features by length. In this way I give priority to the longer feat e.g area_length vs area 
    core_feats_v = sorted(core_feats_v, key = len)[::-1]
    
    def _search_feat(feat):
        for core_f in core_feats_v:
            if feat.startswith(core_f):
                return c_feats_dict[core_f]
        #the correct index was not found return -1
        return -1 
        
    is_expanded = any(x.startswith('d_') for x in core_feats_v)
    if not is_expanded: 
        #if it is not expanded put derivatives as the same
        col_feats = [x[2:] if x.startswith('d_') else x for x in col_feats]
    f_groups_inds = np.array([_search_feat(x) for x in col_feats])
    
    
    return f_groups_inds, c_feats_dict
#%%
def softmax_RFE_grouped(data_in):
    
    fold_id, fold_data, fold_param = data_in
    
    (db_name, i_fold) = fold_id
    (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict) = fold_data
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
    
    core_feats_l = list(core_feats_dict.keys())
    
    fold_res = []
    
    
    while core_feats_l: #continue as long as there are any feature to remove
        n_features = input_v.size(1)
        trainer = TrainerSimpleNet(n_classes, n_features, cuda_id=cuda_id, **trainer_args)
        trainer.fit(input_train, target_train)
        res = trainer.evaluate(input_v, target_v)
        
        print('Test: loss={:.4}, acc={:.2f}%, f1={:.4}'.format(*res))
        
        n_core_f = len(core_feats_l)
        print(db_name, i_fold + 1, n_core_f, n_features)
        

        if n_core_f < 2:
            fold_res.append((core_feats_l[0], res))
            core_feats_l = []
        else:
            # get group importance by removing the features from the model and calculating the result
            
            model = trainer.model
            n_features = model.fc.in_features
            
            res_selection = []
            for f_core in core_feats_l:
                ind = core_feats_dict[f_core]
                ind_valid,  = np.where(feats_groups_inds != ind)
                ind_valid = ind_valid.tolist()
                
                n_features_r = len(ind_valid)
                model_reduced = SimpleNet(n_features_r, n_classes)
                model_reduced.eval()
                
                model_reduced.fc.weight.data = model.fc.weight[:, ind_valid].data
                input_r = input_v[:, ind_valid]
                
                loss, acc, f1 = trainer._evaluate(model_reduced, input_r, target_v)
                
                #i am only using the acc to do the feature selection
                
                res_selection.append((f_core, loss))
            
            #select the group that has the least detrimental effect after being removed 
            group2remove = min(res_selection, key= lambda x : x[1])[0]
            ind = core_feats_dict[group2remove]
            ind_valid,  = np.where(feats_groups_inds != ind)
            ind_valid = ind_valid.tolist()
            
            assert len(ind_valid) > 0
            
            input_v = input_v[:, ind_valid]
            input_train = input_train[:, ind_valid]
            feats_groups_inds = feats_groups_inds[ind_valid]
            core_feats_l.remove(group2remove)
            
            assert input_v.size(1) < n_features
            
            #add progress
            fold_res.append((group2remove, res))
        
    return fold_id, fold_res