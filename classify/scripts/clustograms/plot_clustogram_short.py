#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:07:50 2018

@author: ajaver
"""
import os
import matplotlib.pylab as plt
import seaborn as sns
import sys
sys.path.append('../../helper')
from misc import results_root_dir
from reader import read_feats

#%%

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
    results_dir = os.path.join(results_root_dir, 'clustograms')
    
    experimental_dataset = 'SWDB'
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    if 'all' in feat_data:
        del feat_data['all']
        del feat_data['OW']
    
    for ii in [8, 16]:
        #%%
        feat_cols = top16_manual[:ii]
        
        feats = feat_data['tierpsy']
        
        group_s = 'strain_description'
        rr = [x for x in feat_cols if x in feats] 
        feats = feats[rr + [group_s]]
        df = feats.groupby(group_s).agg('mean')
        
        g = sns.clustermap(df, method = 'ward', figsize=(ii//2, 25), robust=True)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=4)
        
        dd = os.path.join(results_dir, 'Clustogram_{}_top{}.pdf'.format(experimental_dataset, ii))
        g.savefig(dd)