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
from process_RFE import softmax_RFE

core_feats_reduced = ['angular_velocity',
 'angular_velocity_head_base',
 'angular_velocity_head_tip',
 'angular_velocity_tail_tip',
 'area',
 'curvature_head',
 'curvature_hips',
 'curvature_mean_hips',
 'curvature_mean_neck',
 'curvature_midbody',
 'curvature_std_head',
 'curvature_std_hips',
 'curvature_std_midbody',
 'curvature_std_neck',
 'curvature_std_tail',
 'curvature_tail',
 'length',
 'major_axis',
 'minor_axis',
 'quirkiness',
 'relative_to_body_radial_velocity_head_tip',
 'relative_to_body_radial_velocity_tail_tip',
 'relative_to_head_base_angular_velocity_head_tip',
 'relative_to_head_base_radial_velocity_head_tip',
 'relative_to_hips_radial_velocity_tail_tip',
 'relative_to_neck_angular_velocity_head_tip',
 'relative_to_neck_radial_velocity_head_tip',
 'relative_to_tail_base_radial_velocity_tail_tip',
 'speed_head_tip',
 'width_head_base',
 'width_midbody',
 'width_tail_base']

#%%
def fold_generator(fold_param):
    
    for db_name, feats in feat_data.items():
        print(db_name)
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        
        
        cv = StratifiedShuffleSplit(n_splits = n_folds, test_size = test_size, random_state=777)
        if not 'id' in feats:
            y = feats['strain_id'].values
            X = feats[col_feats].values
            for i_fold, (train_index, test_index) in enumerate(cv.split(X, y)):
                x_train, y_train  = X[train_index], y[train_index]
                x_test, y_test  = X[test_index], y[test_index]
                
                fold_data = (x_train, y_train), (x_test, y_test), col_feats.copy()
                fold_id = (db_name, i_fold)
                
                yield (fold_id, fold_data, fold_param)
                
                print(len(set(y_train)), len(set(y_test)))
            
            
        else:
            #deal with augmented data, we cannot select randomly we have to assign data from the same video to different videos
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
                
                
                fold_data = (x_train, y_train), (x_test, y_test), col_feats.copy()
                fold_id = (db_name, i_fold)
                
                yield (fold_id, fold_data, fold_param)
            
                print(len(set(y_train)), len(set(y_test)))

if __name__ == "__main__":
    #is_super_reduced = False
    is_extra_comp = True
    
    is_extra_comp_less = True
    
    pool_size = 2#10
    
    cuda_id = None#0
    
    
    metric2exclude = 'loss'
    n_feats2remove = 'log2'
    #n_folds = 10
    #n_feats2remove = 1
    
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

#%%
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    
    
    #%%
    
    if is_extra_comp:
        n_folds = 10
        
        df = feat_data['tierpsy']
        index_cols = [x for x in col2ignore_r if x in df]
    
        v_cols = [x for x in df.columns if any(x.startswith(f) or x.startswith('d_' + f) for f in core_feats_reduced)]    
        feat_data['tierpsy_reduced'] = df[index_cols + v_cols].copy() #reduce to only the selected features 
        
        v_cols = [x for x in df.columns if not '_norm' in x]
        feat_data['tierpsy_no_norm'] =  df[v_cols].copy()
        
        v_cols = [x for x in df.columns if not '_w_' in x]
        feat_data['tierpsy_no_sub'] =  df[v_cols].copy()
        
        v_cols = [x for x in df.columns if not x.startswith('d_')]
        feat_data['tierpsy_no_dev'] =  df[v_cols].copy()
        
        v_cols = [x for x in df.columns if not (('eigen' in x) or ('blob' in x))]
        feat_data['tierpsy_no_blob_no_eigen'] = df[v_cols].copy()
        
        df_r = feat_data['tierpsy_no_blob_no_eigen']
        v_cols = [x for x in df_r if not '_norm' in x]
        feat_data['tierpsy_no_blob_no_eigen_no_norm'] = df_r[v_cols].copy()
        
        if experimental_dataset == 'SWDB':
            v_cols = [x for x in df.columns if not '_abs' in x]
            feat_data['tierpsy_no_abs'] =  df[v_cols].copy()
            
            v_cols_remove = [x.replace('_abs', '') for x in df.columns if '_abs' in x]
            v_cols = list(set(df.columns) - set(v_cols_remove))
            feat_data['tierpsy_only_abs'] =  df[v_cols].copy()
            
            df_r = feat_data['tierpsy_no_blob_no_eigen']
            v_cols_remove = [x.replace('_abs', '') for x in df_r.columns if '_abs' in x]
            v_cols = list(set(df_r.columns) - set(v_cols_remove))
            feat_data['tierpsy_no_blob_no_eigen_only_abs'] =  df_r[v_cols].copy()
            
            df_r = feat_data['tierpsy_no_blob_no_eigen_only_abs']
            v_cols = [x for x in df_r if not '_norm' in x]
            feat_data['tierpsy_no_blob_no_eigen_only_abs_no_norm'] = df_r[v_cols].copy()
            
    
        #if is_extra_comp_less:
        #   feat_data = {k:feat_data[k] for k in ['tierpsy', 'OW', 'all', 'tierpsy_no_blob_no_eigen_only_abs']}
    
    elif is_super_reduced:
        
        if 'all' in feat_data:
            del feat_data['all']
            del feat_data['OW']
        
        pool_size = 15
        n_folds = 30#500
        
        df = feat_data['tierpsy']
        #remove ventral signed columns that where not abs (This ones seemed useless...)
        v_cols = [x for x in df.columns if not (('eigen' in x) or ('blob' in x))]
        v_cols = [x for x in v_cols if not '_w_' in x]
        v_cols = [x for x in v_cols if not ('curvature_mean' in x) and not ('curvature_std' in x)]
        v_cols = [x for x in v_cols if not x.endswith('1')]
        v_cols_remove = [x.replace('_abs', '') for x in v_cols if '_abs' in x]
        v_cols_c = list(set(v_cols) - set(v_cols_remove))
        feat_data['tierpsy_super_reduced'] =  df[v_cols].copy()
        
        del feat_data['tierpsy']
        
    
    else:
        n_folds = 500
        df = feat_data['tierpsy']
        #remove ventral signed columns that where not abs (This ones seemed useless...)
        v_cols = [x for x in df.columns if not (('eigen' in x) or ('blob' in x))]
        v_cols_remove = [x.replace('_abs', '') for x in v_cols if '_abs' in x]
        v_cols = list(set(v_cols) - set(v_cols_remove))
    
        feat_data['tierpsy'] =  df[v_cols].copy()
        if experimental_dataset == 'SWDB':
            index_cols = [x for x in col2ignore_r if x in df]
            v_cols = [x for x in v_cols if any(x.startswith(f) or x.startswith('d_' + f) for f in core_feats_reduced)]    
            feat_data['tierpsy_reduced'] = df[index_cols + v_cols].copy() #reduce to only the selected features 
    

    #%%
    
    fold_param = (cuda_id, train_args, metric2exclude, n_feats2remove)
    gen = fold_generator(fold_param)
    #%%
    all_data_in = []
    results = []
    p = mp.Pool(pool_size)
    for gen in iter(gen):
        all_data_in.append(gen)
        if len(all_data_in) == pool_size:
            results += p.map(softmax_RFE, all_data_in)
            all_data_in = []
    results += p.map(softmax_RFE, all_data_in)
    
    #%%
    results_dir = os.path.join(results_root_dir, 'RFE')
    bn = '{}_RFE_SoftMax_F{}_reduced.pkl'.format(experimental_dataset, n_feats2remove)
    save_name = os.path.join(results_dir,  bn)
    
    
    if not is_extra_comp:
        save_name = 'R_' + save_name
        
    if is_super_reduced:
        all_feats = [x for x in feat_data['tierpsy_super_reduced'].columns if not x in col2ignore_r ]
        save_name = 'RSuper_' + save_name
        results = (all_feats, results)
    
    with open(save_name, "wb" ) as fid:
        pickle.dump(results, fid)
    #%%
    
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    
    
    #res = softmax_RFE(all_data_in[4])
    #%%
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
        
        plt.ylim((0, 60))
    
    
    #%%
    fig, ax = plt.subplots(1, 1)
    for k, dat in res_db.items():
        
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        tot = len(sum(feats, []))
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        
        
        if n_feats2remove == 'log2':
            n2 = int(np.floor(np.log2(tot - 1e-5)))
            xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
        else:
            xx = np.arange(tot, 0, -n_feats2remove) + 1
        
        print(k, np.max(yy))
        
        h = ax.errorbar(xx, yy, yerr=err, label = k)
    #plt.xlim(0, 64)
    #plt.ylim(10, 45)
    plt.legend()
    
    #%%
    
    for k, dat in res_db.items():
        #if k != 'OW': continue
        
        plt.figure()
        
        dd = []
        for (feats, loss, acc, f1) in dat:
            dd.append(acc)
        tot = len(sum(feats, []))
        
        yy = np.mean(dd,axis=0)
        err = np.std(dd,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        
        plt.errorbar(xx, yy, yerr=err)
        
        
        ind = np.argmax(yy)
        #x_max = xx[ind]
        #plt.plot((x_max, x_max), plt.ylim())
        
        
        th = yy[ind] - err[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(k, x_t, yy[min_ind])
        
        plt.title(k)
    
        
        feats = [x[0] for x in dat]
        feats = [sum(x, []) for x in feats]
        
        
        
        col_feats = [x for x in feat_data[k].columns if x not in col2ignore_r]
        for ff in feats:
            dd = list(set(col_feats) - set(ff))
            assert len(dd) == 1
            ff.append(dd[0])
        
        
        #min_ind = 20
        rr = None
        for ff in feats:
            s = ff[:min_ind]
            if rr is None:
                rr = set(s)
            else:
                rr.intersection_update(s)
        
        rr2 = None
        for ff in feats:
            s = ff[min_ind:]
            if rr2 is None:
                rr2 = set(s)
            else:
                rr2.intersection_update(s)
               
        print(k, tot, min_ind,len(rr), tot-min_ind, len(rr2))
        
        for ff in feats:
            print(ff[-10:])
        print('%%%%%%%%%%%%%%%%%%%%%%%%')
        
    #plt.xlim((0,20))
    #%%
    [x[0][0] for x in res_db['tierpsy_reduced']]
    
    