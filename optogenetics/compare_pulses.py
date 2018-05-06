#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:57:20 2018

@author: ajaver
"""

#https://brainder.org/tag/permutation-test/

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from calculate_hist_per_video import get_JSD
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

feat_reduced = [
     'length',
     'width_midbody',
     
     'curvature_tail',
     'curvature_hips',
     'curvature_midbody',
     'curvature_neck',
     'curvature_head',
     
     'd_curvature_tail',
     'd_curvature_hips',
     'd_curvature_head',
     'd_curvature_neck',
     'd_curvature_midbody',
     
     'minor_axis',
     'major_axis',
     'quirkiness',
     
     'relative_to_hips_radial_velocity_tail_tip',
     'relative_to_hips_angular_velocity_tail_tip',
     
     'relative_to_body_radial_velocity_tail_tip',
     'relative_to_body_angular_velocity_tail_tip',
     
     'relative_to_body_radial_velocity_head_tip',
     'relative_to_body_angular_velocity_head_tip',
     
     'relative_to_neck_radial_velocity_head_tip',
     'relative_to_neck_angular_velocity_head_tip',
     
     'speed'
     ]

#%%

#root_dir = '/Users/avelinojaver/OneDrive - Imperial College London/'
root_dir = '/Users/ajaver/OneDrive - Imperial College London/'

data_dir = root_dir + 'tierpsy_features_experiments/optogenetics/data/'
results_dir = root_dir + 'tierpsy_features_experiments/optogenetics/results/'


strains2ignore = ['ZX819', 'ZX991']
def read_data(fname):
    with open(fname, 'rb') as fid:
        _hist, _stats = pickle.load(fid)
    
    
    for ss in strains2ignore:
        del _hist[ss]
        del _stats[ss]
    
    dat = []
    for ss, val in _stats.items():
        
        for feat, (jsd, pval) in val.items():
            dat.append((ss, feat, jsd, pval))
    
    df = pd.DataFrame(dat, columns=['strain', 'feat', 'JSD', 'pval'])
    
    jsd = df.pivot('strain', 'feat', 'JSD')
    pvals = df.pivot('strain', 'feat', 'pval')
    return _hist, pvals, jsd
#%%
if __name__ == '__main__':
    exp_df = pd.read_csv(os.path.join(data_dir, 'index.csv'), index_col=0)
    
    
    ff = os.path.join(data_dir, 'pulses_short_hist.p')
    avg_hist_short, pvals_short, jsd_short = read_data(ff)
    ff = os.path.join(data_dir, 'pulses_long_hist.p')
    avg_hist_long, pvals_long, jsd_long = read_data(ff)
    
    #%%
    sns.clustermap(np.log10(pvals_short.T), method='ward', figsize=(10, 50))
    dd = os.path.join(results_dir, 'pvals_short.pdf')
    plt.savefig(dd, bbox_inches='tight')
    
    sns.clustermap(np.log10(pvals_long.T), method='ward', figsize=(10, 50))
    dd = os.path.join(results_dir, 'pvals_long.pdf')
    plt.savefig(dd, bbox_inches='tight')
    #%%
    g = sns.clustermap(np.log10(pvals_short[feat_reduced].T), method='ward', figsize=(5, 10))
    dd = os.path.join(results_dir, 'R_pvals_short.pdf')
    plt.savefig(dd, bbox_inches='tight')
    #%%
    g = sns.clustermap(np.log10(pvals_long[feat_reduced][feat_reduced].T), method='ward', figsize=(5, 10))
    dd = os.path.join(results_dir, 'R_pvals_long.pdf')
    plt.savefig(dd, bbox_inches='tight')
    #%%
    
    xcols = ['HBR187',
 'AQ3071',
 'MW544',
 'N2',
 'HBR222',
 'HBR520',
 'AQ2052',
 'AQ2232',
 'HBR180',
 'AQ2235',
 'AQ2050',
 'AQ2028',
]
    ycols = ['relative_to_body_radial_velocity_tail_tip',
 'relative_to_body_angular_velocity_tail_tip',
 'relative_to_body_angular_velocity_head_tip',
 'relative_to_body_radial_velocity_head_tip',
 'relative_to_neck_radial_velocity_head_tip',
 'd_curvature_neck',
 'd_curvature_midbody',
 'd_curvature_tail',
 'd_curvature_hips',
 'relative_to_neck_angular_velocity_head_tip',
 'relative_to_hips_radial_velocity_tail_tip',
 'relative_to_hips_angular_velocity_tail_tip',
 'd_curvature_head',
 'speed',
 'length',
 'width_midbody',
 'curvature_head',
 'curvature_hips',
 'curvature_midbody',
 'curvature_neck',
 'major_axis',
 'minor_axis',
 'curvature_tail',
 'quirkiness']
    #%%
    from statsmodels.sandbox.stats.multicomp import multipletests
    
    def _prepare_pvals(X):
        for ss in X:
            _, X[ss], _, _ = multipletests(X[ss], method='fdr_bh')
        X = np.clip(X, 0, 0.05)
        X = np.log10(X)
        return X
    
    plt.figure(figsize=(4, 6))
    
    H = _prepare_pvals(pvals_short.loc[xcols, ycols].T)
    #clip so anything more than 0.05 is white
    sns.heatmap(H, vmin=-1.3, vmax=-3)
    dd = os.path.join(results_dir, 'Hmap_R_pvals_short.pdf')
    
    plt.savefig(dd, bbox_inches='tight')
    
    plt.figure(figsize=(4, 6))
    H = _prepare_pvals(pvals_long.loc[xcols, ycols].T)
    sns.heatmap(H, vmin=-1.3, vmax=-3)
    dd = os.path.join(results_dir, 'Hmap_R_pvals_long.pdf')
    plt.savefig(dd, bbox_inches='tight')
    #%%
    dat2plot = {'short':avg_hist_short,
            'long':avg_hist_long
            }
    
    for set_type, avg_hist in dat2plot.items():
        #avg_hist = avg_hist_short
        
        uFeats_dict = dict(
        full = sorted(next(iter(avg_hist.values())).keys()), 
        reduced = sorted(feat_reduced)
        )
        for f_size, uFeats in uFeats_dict.items():
            uStrains = sorted(avg_hist.keys())
            
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
            
            
            sns.clustermap(JS_atr_df, figsize=(4.5, 4.5))
            plt.title('ATR ' + f_size)
            
            dd = os.path.join(results_dir, 'CrossJS_{}_ATR_{}.pdf'.format(set_type, f_size))
            plt.savefig(dd, bbox_inches='tight')
            
            sns.clustermap(JS_ctr_df, figsize=(4.5, 4.5))
            plt.title('EtOH ' + f_size)
    
            dd = os.path.join(results_dir, 'CrossJS_{}_EtOH_{}.pdf'.format(set_type, f_size))
            plt.savefig(dd, bbox_inches='tight')
        #break
    #%%
    
    #%%
    dat2plot = {'short':avg_hist_short,
            'long':avg_hist_long
            }
    
    for set_type, avg_hist in dat2plot.items():
        #avg_hist = avg_hist_short
        
        uFeats_dict = dict(
        full = sorted(next(iter(avg_hist.values())).keys()), 
        #reduced = sorted(feat_reduced)
        )
        for f_size, uFeats in uFeats_dict.items():
            uStrains = sorted(avg_hist.keys())
            
            n_strains = len(uStrains)
            n_features = len(uFeats)
            
            JS_atr = np.zeros((n_strains, n_strains))
            JS_ctr = np.zeros((n_strains, n_strains))
            
            for i1, s1 in enumerate(uStrains):
                for i2, s2 in enumerate(uStrains):
                    hist_ctr = []
                    hist_atr = []
                    for ii, feat in enumerate(uFeats):   
                        P_atr, P_ctr = avg_hist[s1][feat]
                        Q_atr, Q_ctr = avg_hist[s2][feat]
                        hist_atr.append((P_atr, Q_atr))
                        hist_ctr.append((P_ctr, Q_ctr))
                        
                    P, Q = list(map(np.array, zip(*hist_atr)))
                    JS_atr[i1, i2] = get_JSD(P, Q) 
                    
                    P, Q = list(map(np.array, zip(*hist_ctr)))
                    JS_ctr[i1, i2] = get_JSD(P, Q) 
                    

            JS_atr_df = pd.DataFrame(JS_atr, index=uStrains, columns=uStrains)
            JS_ctr_df = pd.DataFrame(JS_ctr, index=uStrains, columns=uStrains)


            max_KL = 0.25
            sns.clustermap(JS_atr_df, figsize=(4.5, 4.5), vmin=0, vmax=max_KL)
            plt.title('ATR ' + f_size)
            
            dd = os.path.join(results_dir, 'R_CrossJS_{}_ATR_{}.pdf'.format(set_type, f_size))
            plt.savefig(dd, bbox_inches='tight')
            
            sns.clustermap(JS_ctr_df, figsize=(4.5, 4.5), vmin=0, vmax=max_KL)
            plt.title('EtOH ' + f_size)
    
            dd = os.path.join(results_dir, 'R_CrossJS_{}_EtOH_{}.pdf'.format(set_type, f_size))
            plt.savefig(dd, bbox_inches='tight')
    
    #%%
    uStrains = sorted(avg_hist_short.keys())
    for feat in feat_reduced:
        
        save_name = os.path.join(results_dir, 'short_hist2D_{}.pdf'.format(feat))
        
        with PdfPages(save_name) as fid_pdf:
            for strain in uStrains:
                h_atr, h_ctr = avg_hist_short[strain][feat]
            
                fig_l = (10, 5)
                fig, axs = plt.subplots(1,2, figsize = fig_l, sharex=True)
                
                axs[0].imshow(h_ctr, aspect='auto', interpolation='none', cmap="inferno")
                axs[0].invert_yaxis()
                axs[0].set_title('ctr')
                
                axs[1].imshow(h_atr, aspect='auto', interpolation='none', cmap="inferno")
                axs[1].invert_yaxis()
                axs[1].set_title('atr')
                
                plt.suptitle(strain)
                
                fid_pdf.savefig(fig)
                plt.close(fig) 
    #%%
               