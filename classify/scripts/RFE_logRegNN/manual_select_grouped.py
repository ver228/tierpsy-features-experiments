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
import glob
import os

def load_data(fname):
    with open(fname, "rb" ) as fid:
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
    return res_db


def get_break_points(res_db):
    feats_div = {}
    data4plots = {}
    for db_name, dat in res_db.items():
        val = dat[3]
        feats = dat[0]
        
        tot = len(feats[0])
        
        yy = np.mean(val,axis=0)
        err = np.std(val,axis=0)
        xx = np.arange(tot, 0, -1) + 1
        
        max_ind = np.argmax(yy)
        
        th = yy[max_ind] - err[max_ind]
        better_ind, =np.where(yy >= th)
        max_err_ind = better_ind[-1]
        min_err_ind = better_ind[0]
        
        
        data4plots[db_name] = ((xx, yy, err), (xx[max_err_ind], xx[min_err_ind]))
        
        #get an average of the feature order in the different sets
        feat_orders = {}
        
        for feats_in_fold in feats:
            for ii, feat in enumerate(feats_in_fold):
                if not feat in feat_orders:
                    feat_orders[feat] = []
                feat_orders[feat].append(ii)
            
            
        feats, order_vals = zip(*feat_orders.items())
        df = pd.DataFrame(np.array(order_vals), index=feats)
        df_m = df.median(axis=1).sort_values()
        
        useless_feats = df_m.index[:min_err_ind]
        mediocre_feats = df_m.index[min_err_ind:max_ind]
        usefull_feats = df_m.index[max_ind:]
        
        feats_div[db_name] = list(map(list, (useless_feats, mediocre_feats, usefull_feats)))
    return feats_div, data4plots

if __name__ == '__main__':
    fnames = glob.glob('./results_data/*_RFE_G_SoftMax_R.pkl')
    
    ordered_feats = {}
    for fname in fnames:
        db_name = os.path.basename(fname).partition('_')[0]
        res_db = load_data(fname)
        feats_div, data4plots = get_break_points(res_db)
        ordered_feats[db_name] = feats_div
        
    #%%
    feat_set = 'tierpsy_no_blob_no_eigen'
    #feat_set = 'tierpsy'
    for db_name, feats_div in ordered_feats.items():
        useless_feats, mediocre_feats, usefull_feats = feats_div[feat_set]
    #%%
    db_names, useless_feats, mediocre_feats, usefull_feats = \
    zip(*[(k, *v[feat_set]) for k,v in ordered_feats.items()])
    #%%
    maybe_good = set(sum(usefull_feats, [])) - set(sum(useless_feats, []))
    maybe_good = sorted(iter(maybe_good))
    
        
    
    