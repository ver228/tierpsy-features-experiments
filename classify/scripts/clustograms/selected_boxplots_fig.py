#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:10:08 2018

@author: avelinojaver
"""

import seaborn as sns
import matplotlib.pylab as plt
import os

from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../../helper')
from reader import read_feats
from misc import results_root_dir
#%%
feat_props = {
     'length_90th' : ('Length $90^{th}$ ($\mu m$)', (500, 1400)),
     'curvature_hips_abs_90th' : ('Absolute Curvature Hips $90^{th}$ ($rad \cdot \mu m^{-1}$)', (0.0018, 0.0132)),
     'motion_mode_paused_fraction' : ('Fraction of Motion Paused', (-0.025, 1.025)),
     'd_curvature_hips_abs_90th' : ('Derivate of Absolute Hips Curvature $90^{th}$\n ($rad \cdot \mu m^{-1} \cdot s^{-1}$)', (0.0, 0.025)),
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
    strain_sets = dict(
    
    mutants = [
          'Schafer Lab N2 (Bristol, UK)',
          'dpy-20(e1282)IV',
          'egl-5(n486)III',
          'unc-9(e101)X',
          'unc-77(e625)IV',
          'ser-4(ok512)III',
          'sma-2(e502)III',
          'trp-4(sy695)I'
          ],
    
    WT = [
          'CGC N2 (Bristol, UK)', 
          'CGC N2, Axenic Liquid Culture (Bristol, UK)',
          'C. elegans Wild Isolate, N3 (Rothamsted, UK)',
          'C. elegans Wild Isolate (Hawaii, USA)', 
          'C. elegans Wild Isolate (Le Blanc, France)',
          'C. elegans Wild Isolate (Ceres, South Africa)',
          'C. elegans Wild Isolate (Beauchne, France)',
          'C. elegans Wild Isolate (Merlet, France)'
          ]
    )
    #%%
    results_dir = os.path.join(results_root_dir, 'clustograms')
    for s_type, strains2check in strain_sets.items():
        df_s = df.loc[df['strain_description'].isin(strains2check)]
        for feat, (s_xlabel, s_xlim) in feat_props.items():
            
            fig = plt.figure(figsize=(5, 3))
            
            fig, ax = plt.subplots(1, 1, figsize = (4, 3.5))

            g = sns.boxplot(y = 'strain_description', x = feat, data = df_s, order=strains2check)
            ax.set_ylabel('')  
            ax.set_xlabel(s_xlabel)  
            ax.set_xlim(s_xlim)
            
            
            fname = 'boxplot_{}_{}.pdf'.format(s_type, feat)
            fname = os.path.join(results_dir, fname)
            fig.savefig(fname, bbox_inches="tight")
              