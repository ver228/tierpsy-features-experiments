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
from misc import results_root_dir

if __name__ == '__main__':
    
    
    results_dir = os.path.join(results_root_dir, 'RFE')
    
    save_dir  = os.path.join(results_dir, 'figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fnames = os.listdir(results_dir)
    
    experimental_dataset = 'SWDB'
    bn = 'F_{}_RFE_SoftMax_Flog2_reduced.pkl'.format(experimental_dataset)
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
    labels_n1 = {
            'tierpsy': 'Tierpsy', 
            'OW': 'Original',
            'all': 'Tierpsy + Original'
            }
    
    labels_n2 = {
            'tierpsy': 'All', 
            'tierpsy_no_sub' : 'No Motion Subdivisions', 
            'tierpsy_no_dev': 'No Derivatives',
            'tierpsy_no_blob_no_eigen_only_abs' : 'No Eigenprojections,\nBlob Features, \nAbs Ventral/Dorsal'
            }
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    for ii, labels_n in enumerate([labels_n1, labels_n2]):
    
    
        for db_name, lab in labels_n.items():
            dat = res_db[db_name]
            val = dat[3]
            feats = dat[0]
            
            yy = np.mean(val,axis=0)
            err = np.std(val,axis=0)
            
            tot = len(sum(feats[0], []))
            n2 = int(np.floor(np.log2(tot - 1e-5)))
            xx = np.array([tot] + [2**x for x in range(n2, 0, -1)])
            
            h = ax[ii].errorbar(xx, yy, yerr=err, label = lab, lw=2.5)
        
        ax[ii].set_xlabel('Number of Features')
        ax[ii].set_ylabel('Accuray')
        ax[ii].legend()
        
    ff = os.path.join(save_dir, 'Tierpsy_RFE.pdf')
    plt.savefig(ff)
    