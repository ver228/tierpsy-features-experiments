#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:58:24 2018

@author: ajaver
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from reader import read_feats

#%%


if __name__ == '__main__':
    experimental_dataset = 'SWDB'
    #experimental_dataset = 'Syngenta'
    #experimental_dataset = 'CeNDR'
    #save_name = './results_data/{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
    save_name = './results_data/R_{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
    
    #%%
    
    #%%
    
    #save_name = 'SyngentaLabeled.pkl'
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
    #%%
    
    key = 'tierpsy'#'all' if 'all' in res_db else 'tierpsy'
    
    feats, _,_, val = res_db[key]
    
    
    tot = len(sum(feats[0], []))
    n2 = int(np.floor(np.log2(tot - 1e-5)))
    all_xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
    all_yy = np.mean(val,axis=0)
    all_err = np.std(val,axis=0)
    
    for db_name, dat in res_db.items():
        #if k != 'tierpsy': continue
        val = dat[3]
        feats = dat[0]
        
        plt.figure()
        
        
        yy = np.mean(val,axis=0)
        err = np.std(val,axis=0)
        
        tot = len(sum(feats[0], []))
        n2 = int(np.floor(np.log2(tot - 1e-5)))
        xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
        
        plt.errorbar(all_xx, all_yy, yerr=all_err, color='gray')
        plt.errorbar(xx, yy, yerr=err)
        
        ind = np.argmax(yy)
        
        th = yy[ind] - err[ind]
        #th = yy[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(db_name, x_t, yy[min_ind], yy.max())
        
        plt.title(db_name)
        
        
    
    
     #%%       
     #I forgot to add the last feature remaining so I have to do a dirty hack 
    if os.path.basename(save_name).startswith('R_'):
        if False:
            #i forgot to add the last feature extracted...
            feat_data, col2ignore_r = read_feats(experimental_dataset)
            all_feats = [x for x in feat_data['tierpsy'].columns if x not in col2ignore_r]
            del feat_data
            #remove ventral signed columns that where not abs (This ones seemed useless...)
            v_cols = [x for x in all_feats if not (('eigen' in x) or ('blob' in x))]
            v_cols_remove = [x.replace('_abs', '') for x in v_cols if '_abs' in x]
            all_feats = list(set(v_cols) - set(v_cols_remove))
            all_feats = set(all_feats)
        
        #%%
        feats_div = {}
        for db_name, dat in res_db.items():
            feat_orders = {}
            feats_in_all_folds, _,_, val = dat
            
            #I forgot to add the last feature remaining so I have to do a dirty hack 
            # I am assuming all the features were selected at least once
            all_feats = set(sum([x[0] for x in feats_in_all_folds],[]))
            
            feat_orders = {}
            for feats_in_fold in map(list, feats_in_all_folds):
                last_feat = list(all_feats - set(sum(feats_in_fold, [])))
                assert len(last_feat) == 1
                feats_in_fold = feats_in_fold + [last_feat]
                
                tot_feats = sum(map(len,feats_in_fold))
                
                n_left = tot_feats
                for ii, val in enumerate(feats_in_fold):
                    rank_i = n_left
                    n_left -= len(val)
                    for feat in val:
                        if not feat in feat_orders:
                            feat_orders[feat] = []
                        feat_orders[feat].append(rank_i)
            #%%
            
            df = pd.DataFrame(feat_orders).T
            df_m = df.median(axis=1).sort_values()
        #%%reduced data     
        reduced_feats = df_m.index[df_m<=512]
        with open('reduced_feats_{}.txt'.format(experimental_dataset), 'w') as fid:
            ss = '\n'.join(reduced_feats)
            fid.write(ss)
