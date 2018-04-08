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


top16_manual = [
    'length_90th',
     'width_midbody_norm_10th',
     'curvature_hips_abs_90th',
     'curvature_head_abs_90th',
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     'd_curvature_hips_abs_90th',
     'd_curvature_head_abs_90th',
     
     
     'width_head_base_norm_10th',
     'motion_mode_backward_frequency',
     'quirkiness_50th',
     'minor_axis_50th',
     
     'curvature_midbody_norm_abs_50th',
     'relative_to_hips_radial_velocity_tail_tip_50th',
     'relative_to_head_base_radial_velocity_head_tip_50th',
     'relative_to_head_base_angular_velocity_head_tip_abs_90th'                   
    
     ]

if __name__ == '__main__':
    experimental_dataset = 'SWDB'
    feat_data, col2ignore_r = read_feats(experimental_dataset, z_transform = False)
    #feat_data, col2ignore_r = read_feats(experimental_dataset, z_transform = True)
    
    df = feat_data['tierpsy']
    del feat_data
    #%%
    strain_sets = dict(
    
    mutants = [
            'Schafer Lab N2 (Bristol, UK)',
          'dpy-20(e1282)IV',
          'egl-5(n486)III',
          'unc-77(e625)IV',
          'unc-9(e101)X',
          'trp-4(sy695)I'],
    
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
    
    results_dir = os.path.join(results_root_dir, 'clustograms')
    for s_type, strains2check in strain_sets.items():
        
        fname = os.path.join(results_dir, 'boxplot_{}.pdf'.format(s_type))
                
        with PdfPages(fname) as fid:
            df_s = df.loc[df['strain_description'].isin(strains2check)]
            for feat in top16_manual:
                fig = plt.figure(figsize=(10, 5))
                g = sns.boxplot(y = 'strain_description', x = feat, data = df_s, order=strains2check)
                
                fid.savefig(fig, bbox_inches="tight")
                
                plt.close(fig)