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

import sys
sys.path.append('../../helper')
from reader import read_feats
from misc import results_root_dir

if __name__ == '__main__':
    experimental_dataset = 'SWDB'
    #experimental_dataset = 'Syngenta'
    #experimental_dataset = 'CeNDR'
    #experimental_dataset = 'MMP'
    
    results_dir = os.path.join(results_root_dir, 'RFE')
    fnames = os.listdir(results_dir)
    
    bn = '{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
    #bn = 'R_{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
    save_name = os.path.join(results_dir, bn)
    
    
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
        if True:
            #i forgot to add the last feature extracted...
            feat_data, col2ignore_r = read_feats(experimental_dataset)
            all_feats = [x for x in feat_data['tierpsy'].columns if x not in col2ignore_r]
            
           
            del feat_data
            #remove ventral signed columns that where not abs (This ones seemed useless...)
            v_cols = [x for x in all_feats if not (('eigen' in x) or ('blob' in x))]
            v_cols_remove = [x.replace('_abs', '') for x in v_cols if '_abs' in x]
            all_feats = list(set(v_cols) - set(v_cols_remove))
            all_feats = set(all_feats)
            
            
            if 'tierpsy_reduced' in res_db:
                from RFE_simple_reduced import core_feats_reduced
                reduced_feats = set([x for x in all_feats if any(x.startswith(f) or x.startswith('d_' + f) for f in core_feats_reduced)])
                
        #%%
        feats_div = {}
        for db_name, dat in res_db.items():
            if db_name == 'tierpsy_reduced':
                all_feats_r = set(reduced_feats)
            else:
                all_feats_r = set(all_feats)
            
            feat_orders = {}
            feats_in_all_folds, _,_, val = dat
            
            #I forgot to add the last feature remaining so I have to do a dirty hack 
            # I am assuming all the features were selected at least once
            #all_feats = set(sum([x[0] for x in feats_in_all_folds],[]))
            
            feat_orders = {}
            for feats_in_fold in map(list, feats_in_all_folds):
                
                ff = sum(feats_in_fold, [])
                last_feat = list(all_feats_r - set(ff))
                
                
                assert len(last_feat) <= 1
                if len(last_feat) > 0:
                    feats_in_fold = feats_in_fold + [last_feat]
                
                if len(sum(feats_in_fold, [])) != len(all_feats_r):
                    raise ValueError
                
                
                tot_feats = sum(map(len,feats_in_fold))
                
                n_left = tot_feats
                for ii, val in enumerate(feats_in_fold):
                    rank_i = n_left
                    n_left -= len(val)
                    for feat in val:
                        if not feat in feat_orders:
                            feat_orders[feat] = []
                        feat_orders[feat].append(rank_i)
            
            df = pd.DataFrame(feat_orders).T
            df_m = df.median(axis=1).sort_values()
            feats_div[db_name] = df_m
            
            df_m.plot()
        
        key = 'tierpsy_reduced' if 'tierpsy_reduced' in feats_div else 'tierpsy'
        
        df_m = feats_div[key]
        reduced_feats = df_m.index[df_m<=512]
        
        fname = os.path.join(results_dir, 'reduced_feats_{}.txt'.format(experimental_dataset))
        with open(fname, 'w') as fid:
            ss = '\n'.join(reduced_feats)
            fid.write(ss)
            #%%
        feats_div = {}
        for db_name, dat in res_db.items():
            if db_name == 'tierpsy_reduced':
                all_feats_r = set(reduced_feats)
            else:
                all_feats_r = set(all_feats)
                
            feat_orders = {}
            feats_in_all_folds, _,_, val = dat
            
            #I forgot to add the last feature remaining so I have to do a dirty hack 
            # I am assuming all the features were selected at least once
            #all_feats = set(sum([x[0] for x in feats_in_all_folds],[]))
            
            ordered_feats = []
            for feats_in_fold in map(list, feats_in_all_folds):
                last_feat = list(all_feats_r - set(sum(feats_in_fold, [])))
                assert len(last_feat) <= 1
                feats_in_fold = feats_in_fold + [last_feat]
                ff = list(sum(feats_in_fold, []))
                
                
                ordered_feats.append(ff)
        for ii in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, len(all_feats_r)]:
            top_feats = [tuple(sorted(x[-ii:])) for x in ordered_feats]
            u_feats = set(top_feats)
            break
            print(ii, len(u_feats))
        #%%
        feat_reduced = [x[-16:] for x in ordered_feats]
        
        
        #%%
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
        
        def get_core_feats(feats):
            
            col_v = _remove_end(feats, ['_10th', '_50th', '_90th', '_95th', '_IQR'])
            
            col_v = _remove_end(col_v, ['_frequency', '_fraction', '_duration', ])
            
                
            #the other is important, this must be at the end
            
            #col_v = list(set([x[2:] if x.startswith('d_') else x for x in col_v]))
            col_v = _remove_end(col_v, ['_abs'])
            col_v = _remove_end(col_v, ['_norm'])
            col_v = _remove_end(col_v, ['_w_forward', '_w_backward', '_w_paused'])
        
            
            return col_v
        #%%
        tot_feats = 1843
        
        for tt in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, tot_feats]:
            top_feats = [tuple(sorted(x[-tt:])) for x in ordered_feats]
            c_feats = [get_core_feats(x) for x in top_feats]
            dd = sorted(map(len, map(set, c_feats)))
            
            print(tt, min(dd), dd[len(dd)//2], max(dd))
            
            #uCore = set(sum(c_feats, []))
            #print(tt, len(uCore))
        
        #%%
        for db_name, dat in res_db.items():
            feat_orders = {}
            feats_in_all_folds, _,_, val = dat
        
        