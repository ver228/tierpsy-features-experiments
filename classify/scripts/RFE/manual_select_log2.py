#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:58:24 2018

@author: ajaver
"""
import pickle
import numpy as np
import matplotlib.pylab as plt
import os


import sys
sys.path.append('../../helper')
from reader import read_feats
from misc import results_root_dir

#%%


if __name__ == '__main__':
    results_dir = os.path.join(results_root_dir, 'RFE')
    
    experimental_dataset = 'SWDB'
    #experimental_dataset = 'Syngenta'
    #experimental_dataset = 'CeNDR'
    #experimental_dataset = 'MMP'
    
    save_name = os.path.join(results_dir, 'R_{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset))
    save_name_rs = os.path.join(results_dir, 'RSuper_R_{}_RFE_SoftMax_Flog2_reduced_v1.pkl'.format(experimental_dataset))
    
    #%%
    with open(save_name, "rb" ) as fid:
        results = pickle.load(fid)
    
    with open(save_name_rs, "rb" ) as fid:
        all_feats_rs, results_rs = pickle.load(fid)
        results += results_rs
    
    
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
    plt.xlim([0, 128])
    
        
     #%%       
     #I forgot to add the last feature remaining so I have to do a dirty hack 
    if os.path.basename(save_name).startswith('R_'):
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
    
        all_feats_d = {'tierpsy':all_feats,
         'tierpsy_reduced':reduced_feats,
         'tierpsy_super_reduced':set(all_feats_rs)}
        
        
        #%%
        feats_div = {}
        for db_name, dat in res_db.items():
            all_feats_r =  all_feats_d[db_name]
            
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
            
            print(db_name)
            for ii in [1, 2, 4, 8, 16]:
                
                top_feats = [tuple(sorted(x[-ii:])) for x in ordered_feats]
                u_feats = set(top_feats)
                print(ii, len(u_feats))
        #%%
        from collections import Counter
        
        def get_top_feats(ordered_feats, top_n, prev_top):
            
            remainder = [x[-top_n:][::-1] for x in ordered_feats].copy()
            
            remainder = [x for x in remainder if all(f in x for f in prev_top)]
            top_feats = [] 
            
            for nn in range(top_n): 
                dd = list(zip(*remainder))
                cc = Counter(sum(dd, ()))
                
                for ff in top_feats:
                    if ff in cc:
                        del cc[ff]
                
                feat, _ = max(cc.items(), key=lambda x : x[1])
                
                remainder = [x for x in remainder if feat in x]
                
                if feat in top_feats:
                    raise
                top_feats.append(feat)
            
            return sorted(top_feats)
        
        
        top8_feats = get_top_feats(ordered_feats, top_n = 8, prev_top=[])
        top16_feats = get_top_feats(ordered_feats, top_n = 16, prev_top=top8_feats)
            
        