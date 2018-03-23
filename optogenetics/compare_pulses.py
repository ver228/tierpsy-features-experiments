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
from calculate_hist_per_video import get_JSD
import seaborn as sns
#%%
def read_data(fname):
    with open(fname, 'rb') as fid:
        _hist, _stats = pickle.load(fid)
    dat = []
    for ss, val in _stats.items():
        for feat, (jsd, pval) in val.items():
            dat.append((ss, feat, jsd, pval))
    
    df = pd.DataFrame(dat, columns=['strain', 'feat', 'JSD', 'pval'])
    
    jsd = df.pivot('strain', 'feat', 'JSD')
    pvals = df.pivot('strain', 'feat', 'pval')
    return _hist, pvals, jsd
    
if __name__ == '__main__':
    exp_df = pd.read_csv('./data/index.csv', index_col=0)
    
    avg_hist_short, pvals_short, jsd_short = read_data('pulses_short_hist.p')
    avg_hist_long, pvals_long, jsd_long = read_data('pulses_long_hist.p')
    
    #%%
    sns.clustermap(np.log10(pvals_short.T), method='ward', figsize=(10, 50))
    plt.savefig('pvals_short.pdf', bbox_inches='tight')
    sns.clustermap(np.log10(pvals_long.T), method='ward', figsize=(10, 50))
    plt.savefig('pvals_long.pdf', bbox_inches='tight')
    #sns.clustermap(jsd_short.T, method='complete', figsize=(10, 50))
    #sns.clustermap(jsd_long.T, method='complete', figsize=(10, 50))
    
    #%%
    dat2plot = {'short':avg_hist_short,
            'long':avg_hist_long
            }
    
    for set_type, avg_hist in dat2plot.items():
        #avg_hist = avg_hist_short
        
        uStrains = sorted(avg_hist.keys())
        uFeats = sorted(avg_hist[uStrains[0]].keys())
        
        n_strains = len(uStrains)
        n_features = len(uFeats)
        
        JS_atr = np.zeros((n_features, n_strains, n_strains))
        JS_ctr = np.zeros((n_features, n_strains, n_strains))
        for ii, feat in enumerate(uFeats):    
            for i1, s1 in enumerate(uStrains):
                for i2, s2 in enumerate(uStrains):
                    P_atr, P_ctr = avg_hist[s1][feat]
                    Q_atr, Q_ctr = avg_hist[s2][feat]
                    JS_atr[ii, i1, i2] = get_JSD(P_atr, Q_atr) 
                    JS_ctr[ii, i1, i2] = get_JSD(P_ctr, Q_ctr) 
        
        
        reduc_func = np.mean
        #def reduc_func(x, **argkws): return np.percentile(x, [95], **argkws)[0]
        
        dd = JS_atr
        dd = dd/np.linalg.norm(dd, axis=(1,2))[:, None, None]
        dd = reduc_func(dd, axis=0)
        JS_atr_df = pd.DataFrame(dd, index=uStrains, columns=uStrains)
        
        dd = JS_ctr
        dd = dd/np.linalg.norm(dd, axis=(1,2))[:, None, None]
        dd = reduc_func(dd, axis=0)
        JS_ctr_df = pd.DataFrame(dd, index=uStrains, columns=uStrains)
        
        
        sns.clustermap(JS_atr_df)
        plt.title('ATR')
        plt.savefig('CrossJS_{}_ATR.pdf'.format(set_type))
        
        
        sns.clustermap(JS_ctr_df)
        plt.title('EtOH')
        plt.savefig('CrossJS_{}_EtOH.pdf'.format(set_type))
        #break
    #%%
    dd = JS_atr
    dd = dd/np.linalg.norm(dd, axis=(1,2))[:, None, None]
    for nn in [8, 9]:#range(len(uStrains)):
        plt.plot(np.sort(dd[:, 5, nn]))
    
    
    #%%
    