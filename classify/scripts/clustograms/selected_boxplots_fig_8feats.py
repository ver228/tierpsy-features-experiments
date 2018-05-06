#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:10:08 2018

@author: avelinojaver
"""

import seaborn as sns
import matplotlib.pylab as plt
import os

import sys
sys.path.append('../../helper')
from reader import read_feats
from misc import results_root_dir
#%%
feat_props = {
     'length_90th' : ('Length $90^{th}$ ($\mu m$)', (700, 1400)),
     'width_midbody_norm_10th' : ('Width/ Leghth $10^{th}$', (0.05, 0.16)),
     'curvature_hips_abs_90th' : ('Absolute Curvature Hips $90^{th}$\n ($rad \cdot \mu m^{-1}$)', (0.0, 0.012)),
     'curvature_head_abs_90th' : ('Absolute Curvature Head $90^{th}$\n ($rad \cdot \mu m^{-1}$)', (0.0, 0.012)),
     'motion_mode_paused_fraction' : ('Fraction of Motion Paused', (-0.025, 1.025)),
     'motion_mode_paused_frequency' : ('Frequency of Motion Paused\n ($event/s$)', (-0.0025, 0.07)),
     'd_curvature_hips_abs_90th' : ('Derivate of Absolute Hips\n Curvature $90^{th}$ ($rad \cdot \mu m^{-1} \cdot s^{-1}$)', (0.0, 0.035)),
     'd_curvature_head_abs_90th' : ('Derivate of Absolute Head\n Curvature $90^{th}$ ($rad \cdot \mu m^{-1} \cdot s^{-1}$)', (0.0, 0.035)),
     }
#%%
if __name__ == '__main__':
    experimental_dataset = 'SWDB'
    feat_data, col2ignore_r = read_feats(experimental_dataset, z_transform = False)
    #feat_data, col2ignore_r = read_feats(experimental_dataset, z_transform = True)
    
    df = feat_data['tierpsy']
    del feat_data
    
    #this videos are clearly outliers. They are from 2009. Andre's normally says this date was
    #before they developed the final experimental protocol so I assume therewas a problem with them.
    bad_index = df.index[(df['strain']=='N2') & (df['length_50th']>1500)]
    df.drop(bad_index, inplace=True)
    
    
    #%%
    df.loc[df['strain_description'] == 'Schafer Lab N2 (Bristol, UK)', 'strain_description'] = 'N2'
    #%%
    strain_sets = dict(
    
    mutants_short = [
          'N2',
          'dpy-20(e1282)IV',
          'egl-5(n486)III',
          'unc-9(e101)X'
          ]
    )
    #%%
    results_dir = os.path.join(results_root_dir, 'clustograms')
    for s_type, strains2check in strain_sets.items():
        df_s = df.loc[df['strain_description'].isin(strains2check)]
        for feat, (s_xlabel, s_xlim) in feat_props.items():
            fig, ax = plt.subplots(1, 1, figsize = (1.5, 3))
            
            g = sns.boxplot(x = 'strain_description', y = feat, data = df_s, order=strains2check)
            ax.set_xlabel('')  
            ax.set_ylabel(s_xlabel)  
            ax.set_ylim(s_xlim)
            
            loc, labels = plt.xticks()
            g.set_xticklabels(labels, rotation=90)
            
            fname = 'boxplot_{}_{}.pdf'.format(s_type, feat)
            fname = os.path.join(results_dir, fname)
            fig.savefig(fname, bbox_inches="tight")
              