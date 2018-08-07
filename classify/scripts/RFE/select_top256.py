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
    bn = 'F_{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
    #bn = '{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
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
    # i forget to add the last features to the lists so i have to do a dirty hack
    
    #first collect all the features. I am assuming that all the features were eliminated at least once..
    all_feats = set()
    for feat_folds in res_db['tierpsy_no_blob_no_eigen_only_abs_no_norm'][0]:
        feats_flat = [x for ll in feat_folds for x in ll]
        all_feats = all_feats | set(feats_flat)
       
    top256 = []
    #now add the feature and select the top 256
    for feat_folds in res_db['tierpsy_no_blob_no_eigen_only_abs_no_norm'][0]:
        feats_flat = [x for ll in feat_folds for x in ll]
        remaining_feat = all_feats - set(feats_flat)
        feats_flat += list(remaining_feat)
        top256.append(feats_flat[-256:][::-1])
    top256 = pd.DataFrame(top256).T
    top256.to_csv('top256_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv', index=False)
    
    