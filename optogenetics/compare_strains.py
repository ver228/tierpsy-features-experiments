#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:57:20 2018

@author: ajaver
"""

#https://brainder.org/tag/permutation-test/

import pickle
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from calculate_histograms import get_JSD
import seaborn as sns
#%%
if __name__ == '__main__':
    exp_df = pd.read_csv('./data/index.csv', index_col=0)
    uStrain = exp_df['strain'].unique()
    
    hist_results = {}
    for strain in tqdm.tqdm(uStrain):
        fname = './results/{}.p'.format(strain)
        
        with open(fname, 'rb') as fid:
            dat = pickle.load(fid)
            hist_results[strain] = dat
    #%%
    hist_df = []
    for k, val in hist_results.items():
        (all_JSD, pvals_ts, pvals_feat, all_pJSD, pJSD_ranges) = val 
        inds, dat = zip(*[(x[0], x[2][0]) for x in all_JSD])
        df = pd.DataFrame(np.array(dat), index=inds, columns=[k])
        hist_df.append(df)
    hist_df = pd.concat(hist_df, axis=1).T
    
    sns.clustermap(hist_df.T, method='ward', figsize=(7, 50))
    
    
    hist_df_z = (hist_df - hist_df.mean())/hist_df.std()
    sns.clustermap(hist_df_z.T, method='ward', figsize=(5, 50))
    #%%
    hist_df = []
    for k, val in hist_results.items():
        (all_JSD, pvals_ts, pvals_feat, all_pJSD, pJSD_ranges) = val 
        #inds, dat = zip(*[(x[0], x[2][0]) for x in all_JSD])
        inds, dat = zip(*pvals_feat.items())
        df = pd.DataFrame(np.log10(dat), index=inds, columns=[k])
        hist_df.append(df)
    hist_df = pd.concat(hist_df, axis=1).T
    
    sns.clustermap(hist_df.T, method='ward', figsize=(7, 50))
    #%%
    if False:
        hist_df = []
        for k, val in hist_results.items():
            (all_JSD, pvals_ts, pvals_feat, all_pJSD, pJSD_ranges) = val 
            #inds, dat = map(np.array, zip(*[(x[0], x[1][0]) for x in all_JSD]))
            inds, dat = map(np.array, zip(*pvals_ts.items()))
            dat = dat[:, ~np.all(np.isinf(dat),axis=0)]
            dat[np.isinf(dat)] = np.nan
            dat = np.nanpercentile(dat, [5], axis=1)[0]
            
            
            df = pd.DataFrame(np.log10(dat), index=inds, columns=[k])
            hist_df.append(df)
        hist_df = pd.concat(hist_df, axis=1).dropna().T
        
        sns.clustermap(hist_df.T, method='ward', figsize=(5, 20))
    #%%
    
        hist_df_z = (hist_df - hist_df.mean())/hist_df.std()
        sns.clustermap(hist_df_z.T, method='ward', figsize=(5, 20))
        #%%
        
        ks_index = 1
        
        JS_map = {}
        for ind_set, str_set in [(1, 'ATR'), (2, 'EtOH')]:
            all_maps = []
            for ss in uStrain:
                all_JSD = hist_results[ss]
                dat = [x[ks_index][ind_set] for x in all_JSD]
                all_maps.append(dat)
            
            all_maps = np.array(all_maps)
            
            all_maps[np.isinf(all_maps)] = np.nan
            
            
            n_strains, n_feats = all_maps.shape[:2]
            
            
            
            all_JS_C = []
            for nf in tqdm.tqdm(range(n_feats), desc=str_set):
                
                JS_C = np.zeros((n_strains, n_strains))
                
                for s1 in range(n_strains):
                    for s2 in range(n_strains):
                        P = all_maps[s1, nf] + 1e-12
                        Q = all_maps[s2, nf] + 1e-12
                        
                        JS = get_JSD(P,Q)
                        
                        if not isinstance(JS, float):
                            JS[np.isinf(JS)] = np.nan
                            JS = np.nanpercentile(JS, [95])[0]
                        
                        
                        JS_C[s1, s2] = JS
                all_JS_C.append(JS_C) 
            
            JS_map[str_set] = all_JS_C
        #%%
        
        
        for k,val in JS_map.items():
            val_n = np.linalg.norm(val, axis=(1,2))
            val_n = val/np.linalg.norm(val, axis=(1,2))[:, None, None]
            
            #jsd = np.nanmean(val_n, axis=0)
            jsd = np.nanpercentile(val_n, [50], axis=0)[0]
            #plt.figure()
            #plt.imshow(jsd)
            #plt.title(k)
            #print(np.max(jsd))
            
            
            df = pd.DataFrame(jsd, index=uStrain, columns=uStrain)
            sns.clustermap(df, method='ward')
            plt.suptitle(k)
            