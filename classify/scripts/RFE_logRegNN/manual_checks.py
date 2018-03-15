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

if __name__ == '__main__':
    n_feats2remove = 'log2'
    experimental_dataset = 'SWDB'
    
    save_name = '{}_RFE_SoftMax_F{}_reduced.pkl'.format(experimental_dataset, n_feats2remove)
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
        
        ind = np.argmax(yy)
        
        th = yy[ind] - err[ind]
        #th = yy[ind]
        min_ind = np.where(yy >= th)[0][-1]
        
        
        x_t = xx[min_ind]
        plt.plot((x_t, x_t), plt.ylim())
        
        print(db_name, x_t, yy[min_ind], yy.max())
        
        plt.title(db_name)
        
        
        feat_orders = {}
        
        for feats_in_fold in feats:
            for ii, feat in enumerate(feats_in_fold):
                
                if isinstance(feat, list):
                    for ff in feat:
                        if not ff in feat_orders:
                            feat_orders[ff] = []
                        feat_orders[ff].append(ii)
                else:
                    if not feat in feat_orders:
                        feat_orders[feat] = []
                    feat_orders[feat].append(ii)
            
            
        feats, order_vals = zip(*feat_orders.items())
        df = pd.DataFrame(np.array(order_vals), index=feats)
        df_m = df.median(axis=1).sort_values()
        
    
        
        useless_feats = df_m.index[:min_ind]
        usefull_feats = df_m.index[min_ind:]
        feats_div[db_name] = (useless_feats, usefull_feats)
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
        
        
        h = ax.errorbar(xx, yy, yerr=err, label = k)
    #plt.xlim(0, 32)
    plt.legend()
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    #%%
    #feats = [x[0] for x in res_db['all']]
    
    
    feats = [x[0] for x in res_db['tierpsy']]
    