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

feat_reduced = ['d_curvature_head',
 'd_curvature_hips',
 'd_curvature_midbody',
 'd_width_head_base',
 'd_width_midbody',
 'relative_to_hips_radial_velocity_tail_tip',
 'speed']

tick_labels = dict(
    d_curvature_head = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05]),
    d_curvature_hips = np.array([0.    , 0.0078, 0.0156, 0.0234, 0.0312, 0.039 ]),
    d_curvature_midbody = np.array([0. , 0.0063, 0.0126, 0.0189, 0.0252, 0.0315]),
    d_width_head_base = np.array([-60., -36., -12.,  12.,  36.,  60.]),
    d_width_midbody = np.array([-25., -15.,  -5.,   5.,  15.,  25.]),#-24.004119873046875 24.520137786865234 48.52425765991211
    relative_to_hips_radial_velocity_tail_tip = np.array([-160.,  -92.,  -24.,   44.,  112.,  180.]),
    speed = np.array([-400, -255, -110,   35,  180,  325,  470])
    )

feat_labels = {
        'd_curvature_head' : 'Derivate of Absolute  \n Curvature Heads ($rad \cdot \mu m^{-1}\cdot s^{-1}$)',
        'd_curvature_hips' : 'Derivate of Absolute \n Curvature Hips ($rad \cdot \mu m^{-1}\cdot s^{-1}$)',
        'd_curvature_midbody' : 'Derivate of Absolute \n Curvature Midbody ($rad \cdot \mu m^{-1}\cdot s^{-1}$)',
        
        'd_width_head_base' : 'Derivate of \n Heads Base Width ($\mu m \cdot s^{-1}$)',
        'd_width_midbody' : 'Derivate of \n Midbody Width ($\mu m \cdot s^{-1}$)',
        
        'relative_to_hips_radial_velocity_tail_tip' : 'Radial Tail Tip Velocity \n Relative to Hips ($\mu m \cdot s^{-1}$)',
        'speed' : 'Speed ($\mu m \cdot s^{-1}$)',
    }


#%%
data_dir = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_features_experiments/optogenetics/data/'
results_dir = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_features_experiments/optogenetics/results/'


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
        
    bin_ranges_file = os.path.join(data_dir, 'bin_limits.csv')
    bin_lims = pd.read_csv(bin_ranges_file, index_col=0)
    
    ff = os.path.join(data_dir, 'pulses_short_hist.p')
    avg_hist_short, pvals_short, jsd_short = read_data(ff)
    ff = os.path.join(data_dir, 'pulses_long_hist.p')
    avg_hist_long, pvals_long, jsd_long = read_data(ff)
    
    uStrains = sorted(avg_hist_short.keys())
    for feat in feat_reduced:
        
        save_name = os.path.join(results_dir, 'R_short_hist2D_{}.pdf'.format(feat))
            
        n_strains = len(uStrains)
        
        
        fig_l = (3, 3*n_strains)#(2*n_strains, 5)
        fig, axs = plt.subplots(n_strains, 2, figsize = fig_l, sharex=True, sharey=True)
            
        for i_type, s_type in enumerate(('Atr', 'Ctr')):
            i2 = 1 if i_type==0 else 0
            
           
            
            for i1, strain in enumerate(uStrains):
                h_data = avg_hist_short[strain][feat][i_type]
                h_data = h_data/np.sum(h_data)
                h_data = h_data[1:, :]
                
                clip_v = 0.02
                h_data = np.clip(h_data, 0, clip_v)
                h_data = np.round(h_data/clip_v*100)
                
                
                axs[i1][i2].imshow(h_data, aspect='auto', interpolation='none', cmap="inferno")
                axs[i1][i2].invert_yaxis()
                axs[i1][i2].set_title('{} {}'.format(s_type, strain))
            
            axs[i1][i2].set_xticks([0, 4.5, 9])
            axs[i1][i2].set_xticklabels([0, 5, 10])
            
            axs[i1][i2].set_yticks(np.linspace(0, h_data.shape[0]-1, 6))
            axs[i1][i2].set_yticklabels(tick_labels[feat])
            
            
            
        axs[i1][0].set_ylabel(feat_labels[feat])
        
        
        fig.savefig(save_name, bbox_inches='tight')
    