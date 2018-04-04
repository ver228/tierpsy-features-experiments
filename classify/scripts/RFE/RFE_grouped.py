#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver

"""

import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

import multiprocessing as mp

import os
import sys
sys.path.append('../../helper')
from misc import results_root_dir
from reader import read_feats
from process_RFE import softmax_RFE_grouped, get_core_features, get_feat_group_indexes

#%%    
def fold_generator_grouped(feat_data, n_folds, test_size, fold_param):
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        feats_groups_inds, core_feats_dict = get_feat_group_indexes(core_feats[db_name], col_feats)
        
        cv = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=777)
        if not 'id' in feats:
            y = feats['strain_id'].values
            X = feats[col_feats].values
            
            for i_fold, (train_index, test_index) in enumerate(cv.split(X, y)):
                x_train, y_train  = X[train_index], y[train_index]
                x_test, y_test  = X[test_index], y[test_index]
                
                fold_data = (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict)
                fold_id = (db_name, i_fold)
                
                yield (fold_id, fold_data, fold_param)
                
                print(len(set(y_train)), len(set(y_test)))
            
        else:
            feats_r = feats.drop_duplicates('id')
            
            y_r = feats_r['strain_id'].values
            exp_ids = feats_r['id'].values
            for i_fold, (train_index, test_index) in enumerate(cv.split(exp_ids, y_r)):
                good_train = feats['id'].isin(exp_ids[train_index])
                y_train = feats.loc[good_train, 'strain_id'].values.copy()
                x_train = feats.loc[good_train, col_feats].values.copy()
                
                good_test = feats['id'].isin(exp_ids[test_index])
                y_test = feats.loc[good_test, 'strain_id'].values.copy()
                x_test = feats.loc[good_test, col_feats].values.copy()
                
                
                fold_data = (x_train, y_train), (x_test, y_test), (feats_groups_inds, core_feats_dict)
                fold_id = (db_name, i_fold)
                
                yield (fold_id, fold_data, fold_param)
            
                print(len(set(y_train)), len(set(y_test)))

#%%
core_feats_reduced = [
 'curvature_head',
 'curvature_neck',
 'curvature_midbody',
 'curvature_hips',
 'curvature_tail',
 
 'curvature_mean_hips',
 'curvature_mean_neck',
 
 'curvature_std_head',
 'curvature_std_hips',
 'curvature_std_midbody',
 'curvature_std_neck',
 'curvature_std_tail',
 
 'width_head_base',
 'width_midbody',
 'width_tail_base',
 
 'area',
 'length',
 'major_axis',
 'minor_axis',
 'quirkiness',
 
 'speed_head_tip',
 
 'angular_velocity',
 'angular_velocity_head_base',
 'angular_velocity_head_tip',
 'angular_velocity_tail_tip',
 
 'relative_to_body_radial_velocity_head_tip',
 'relative_to_body_radial_velocity_tail_tip',
 
 'relative_to_head_base_angular_velocity_head_tip',
 'relative_to_head_base_radial_velocity_head_tip',
 
 'relative_to_neck_angular_velocity_head_tip',
 'relative_to_neck_radial_velocity_head_tip',
 
 'relative_to_hips_radial_velocity_tail_tip',
 
 'relative_to_tail_base_radial_velocity_tail_tip'
 ]


#%%
if __name__ == "__main__":
    cuda_id = None
    n_folds = 10
    pool_size = 2
    
    train_args = dict(n_epochs = 250,  batch_size = 250,  lr = 0.01, momentum = 0.9)
    test_size = 0.2
    experimental_dataset = 'SWDB'
    
#    train_args = dict(n_epochs = 40,  batch_size = 250, lr = 0.01, momentum = 0.9)
#    test_size = 1/3
#    experimental_dataset = 'CeNDR'
        
#    pool_size = 15
#    train_args = dict(n_epochs = 40,  batch_size = 250, lr = 0.01, momentum = 0.9)
#    test_size = 1/3
#    experimental_dataset = 'Syngenta'

#    train_args = dict(n_epochs = 40,  batch_size = 250, lr = 0.01, momentum = 0.9)
#    test_size = 1/3
#    experimental_dataset = 'MMP'  
    
    is_expanded = False
    
    results_dir = os.path.join(results_root_dir, 'RFE')
    #save_dir = os.path.join(data_root_dir, experimental_dataset)
    
    #%%
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    core_feats = get_core_features(feat_data, col2ignore_r, is_expanded)    
    
    if is_expanded:
        save_postfix = '{}_RFE_G_SoftMax_R_expanded.pkl'
        
        df = feat_data['tierpsy']
        v_cols = [x for x in df.columns if any(x.startswith(f) or x.startswith('d_' + f) for f in core_feats_reduced)]
        index_cols = [x for x in col2ignore_r if x in df]
        
        df_reduced = df[index_cols + v_cols]
        feat_data['tierpsy_reduced'] = df_reduced.copy()
        feat_data['tierpsy_reduced_sub'] =  df_reduced.copy()
        feat_data['tierpsy_reduced_only_norm'] =  df_reduced.copy()
        
        c_cols_sub = core_feats['tierpsy']
        c_cols_sub = [x for x in c_cols_sub if any(x.startswith(f) or x.startswith('d_' + f) for f in core_feats_reduced)]
        core_feats['tierpsy_reduced_sub'] = c_cols_sub
        
        c_cols = [x for x in c_cols_sub if not '_w_' in x]
        core_feats['tierpsy_reduced'] = c_cols
        core_feats['tierpsy_reduced_only_norm'] = [x for x in c_cols if 'norm' in x]
        
        
        if experimental_dataset == 'SWDB':
            feat_data['tierpsy_reduced_only_abs'] =  df_reduced.copy()
            feat_data['tierpsy_reduced_only_noabs'] =  df_reduced.copy()
            
            core_feats['tierpsy_reduced_only_abs'] = [x for x in c_cols if ('abs' in x)]
            core_feats['tierpsy_reduced_only_noabs'] = [x.replace('_abs', '') for x in c_cols if ('abs' in x)]
            
        del core_feats['tierpsy']
        del feat_data['tierpsy']
        
        if 'all' in feat_data:
            del core_feats['OW']
            del feat_data['OW']
            del core_feats['all']
            del feat_data['all']
    else:
        save_postfix = '{}_RFE_G_SoftMax_R.pkl'
    
        df = feat_data['tierpsy']
        cols_no_blob_no_eigen = [x for x in df.columns if not (('eigen' in x) or ('blob' in x))]
        feat_data['tierpsy_no_blob_no_eigen'] = df[cols_no_blob_no_eigen]
        core_feats['tierpsy_no_blob_no_eigen'] = [x for x in core_feats['tierpsy'] if not (('eigen' in x) or ('blob' in x))]
        
        
        
        if 'all' in feat_data:
            # i will only remove the features of OW in the hope of finding the ones that are still usefull
            feat_data['all_ow'] = feat_data['all']
            
            dd = ['ow_' + x for x in core_feats['OW']]
            dd = [x for x in dd if x in core_feats['all']]
            core_feats['all_ow'] = dd
        
        
            #i am not really interesting in the combination of all the features, and it takes a lot of time to calculate
            del core_feats['all']
            del feat_data['all']
    
    results_dir = os.path.join(results_root_dir, 'RFE')
    save_name = os.path.join(results_dir, save_postfix.format(experimental_dataset))
    #%%
    fold_param = (cuda_id, train_args, '', '')
    gen = fold_generator_grouped(feat_data, n_folds, test_size, fold_param)
    #%%
    all_data_in = []
    results = []
    p = mp.Pool(pool_size)
    for gen in iter(gen):
        all_data_in.append(gen)
        if len(all_data_in) == pool_size:
            results += p.map(softmax_RFE_grouped, all_data_in)
            all_data_in = []
    results += p.map(softmax_RFE_grouped, all_data_in)
    
    #%%
    with open(save_name, "wb" ) as fid:
        pickle.dump(results, fid)
    #%%
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    #%%
    results = [x for x in results if x[0] is not None]
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        res_db[db_name].append((feats, loss, acc, f1))
    #%%
    import matplotlib.pyplot as plt
    for k, dat in res_db.items():
        plt.figure()
        for (feats, loss, acc, f1) in dat:
            plt.plot(acc)
        plt.title(k)
        
        plt.ylim((0, 70))
    
    #%%
    fig, ax = plt.subplots(1, 1)
    for k, dat in res_db.items():
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        
        tot = len(feats)
        xx = np.arange(tot, 0, -1)
        
        h = ax.errorbar(xx, yy, yerr=err, label = k)
    #plt.xlim(0, 10)
    plt.legend()
    #%%
    from collections import Counter
    for k, dat in res_db.items():
        #if k != 'tierpsy': continue
        
        plt.figure()
        
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        tot = len(feats)
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        plt.errorbar(xx, yy, yerr=err)
        
        ind = np.argmax(yy)
        
        th = yy[ind] - err[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(k, x_t, yy[min_ind])
        
        plt.title(k)
    
        
        feats = [x[0] for x in dat]
        
        useless_feats = sum([list(x[:min_ind]) for x in feats], [])
        
        usefull_feats = sum([list(x[min_ind:]) for x in feats], [])
        
        useless_feats = sorted(Counter(useless_feats).items(), key = lambda x : x[1])[::-1]
        usefull_feats = sorted(Counter(usefull_feats).items(), key = lambda x : x[1])[::-1]
        
    #%%
    plt.show()
    
    
    