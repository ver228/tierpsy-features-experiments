#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:58:24 2018

@author: ajaver
"""

import pickle
import numpy as np
import matplotlib.pylab as plt
import pandas as pd


if __name__ == '__main__':
    #save_name = 'CeNDR_RFE_G_SoftMax_R.pkl'
    save_name = 'SWDB_RFE_G_SoftMax_R.pkl'
    #save_name = 'RFE_G_SoftMax_R_expanded.pkl'
    #%%
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    
    #%%
    res_db = {}
    for (db_name, i_fold), dat in results:
        if db_name not in res_db:
            res_db[db_name] = []
            
        feats, vals = zip(*dat)
        loss, acc, f1 = map(np.array, zip(*vals))
        
        res_db[db_name].append((feats, loss, acc, f1))
    #%%
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
    plt.figure()
    for db_name, dat in res_db.items():
        #if k != 'tierpsy': continue
        
        val = dat[3]
        feats = dat[0]
        
        tot = len(feats[0])
        
        yy = np.mean(val,axis=0)
        err = np.std(val,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        plt.errorbar(xx, yy, yerr=err)
        
        
    #%%
    feats_div = {}
    for db_name, dat in res_db.items():
        #if k != 'tierpsy': continue
        
        val = dat[3]
        feats = dat[0]
        
        plt.figure()
        
        tot = len(feats[0])
        
        yy = np.mean(val,axis=0)
        err = np.std(val,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        plt.errorbar(xx, yy, yerr=err)
        
        max_ind = np.argmax(yy)
        
        th = yy[max_ind] - err[max_ind]
        #th = yy[ind]
        max_err_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[max_err_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(db_name, x_t, yy[max_err_ind], yy.max())
        
        plt.title(db_name)
        
        
        feat_orders = {}
        
        for feats_in_fold in feats:
            for ii, feat in enumerate(feats_in_fold):
                if not feat in feat_orders:
                    feat_orders[feat] = []
                feat_orders[feat].append(ii)
            
            
        feats, order_vals = zip(*feat_orders.items())
        df = pd.DataFrame(np.array(order_vals), index=feats)
        df_m = df.median(axis=1).sort_values()
        
        feats_div[db_name] = (df_m, max_err_ind, max_ind)
        #feats_div[db_name] = (useless_feats, usefull_feats)
    #%%
    
    #useless_feats, usefull_feats = feats_div['all_ow']
    #df_m, max_err_ind, max_ind = feats_div['all_ow']
    df_m, max_err_ind, max_ind = feats_div['tierpsy_no_blob_no_eigen']
    #df_m, max_err_ind, max_ind = feats_div['tierpsy']
    
    useless_feats = df_m.index[:max_err_ind]
    usefull_feats = df_m.index[max_err_ind:]
    
    #useless_feats = df_m.index[:max_ind]
    #usefull_feats = df_m.index[max_ind:]
    
    #%%
    