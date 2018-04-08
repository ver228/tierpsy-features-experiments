#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:10:08 2018

@author: avelinojaver
"""
import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt   

import sys
sys.path.append('../../helper')
from reader import read_feats
from misc import results_root_dir

if __name__ == '__main__':
    experimental_dataset = 'SWDB'
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    
    df = feat_data['tierpsy']
    del feat_data
    #%%
    v_cols = [x for x in df.columns if not (('eigen' in x) or ('blob' in x))]
    #v_cols = [x for x in v_cols if not '_norm' in x]
    v_cols = [x for x in v_cols if not '_w_' in x]
    v_cols = [x for x in v_cols if not 'path_curvature' in x]
    v_cols_remove = [x.replace('_abs', '') for x in v_cols if '_abs' in x]
    cols = list(set(v_cols) - set(v_cols_remove))
    df = df[cols]
    
    group_s = 'strain_description'
    feat_cols = [x for x in df if not x in col2ignore_r] 
    df_mean = df[feat_cols + [group_s]].groupby(group_s).agg('mean')
    #%%
    
    dat = df_mean.values.T
    
    Z = hac.linkage(dat, method='ward')
    
    plt.figure()
    hac.dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p = 16,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        )
    
    clusters = hac.fcluster(Z, 16, criterion='maxclust')
    
    #%%
    feat_g = {}
    for ff, ii in zip(feat_cols, clusters):
        if not ii in feat_g:
            feat_g[ii] = []
        feat_g[ii].append(ff)
        
    #%%
    top16_manual = [
     'length_50th',
     'width_midbody_IQR',
     
     'curvature_hips_abs_90th',
     'curvature_head_abs_90th',
     
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     
     'd_curvature_hips_abs_90th',
     'd_curvature_head_abs_90th',
     
     'd_curvature_midbody_abs_50th',
     'motion_mode_forward_frequency',
     'quirkiness_50th',
     'minor_axis_50th',
     
     'd_speed_10th', #'d_length_10th', 'relative_to_tail_base_radial_velocity_tail_tip_10th'
     'speed_10th',
     'speed_90th',
     'motion_mode_backward_duration_50th' 
     
     #'width_midbody_50th',
     #'curvature_midbody_abs_50th',
     #'width_midbody_10th',
     #'width_head_base_10th',
     #'relative_to_hips_radial_velocity_tail_tip_50th',
     #'relative_to_head_base_radial_velocity_head_tip_50th',
     #'relative_to_head_base_angular_velocity_head_tip_abs_90th'                   
     ]
    dat = []
    for ff in top16_manual:
        dd = [x for x in feat_g if ff in feat_g[x]]
        
        dat.append((ff, dd[0]))
        
    for dd in sorted(dat, key = lambda x : x[1]):
        print(dd)
    
    