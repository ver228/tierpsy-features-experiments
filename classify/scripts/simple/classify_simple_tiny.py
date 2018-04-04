#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:07:50 2018

@author: ajaver
"""

import os
import numpy as np
import multiprocessing as mp
import pickle


import sys
sys.path.append('../../helper')
from misc import results_root_dir
from reader import read_feats
from process_simple import get_softmax_clf, fold_generator


cols_types = dict(
    top8_auto_v1 = ['curvature_tail_abs_90th',
     'd_curvature_hips_abs_90th',
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     'quirkiness_50th',
     'width_head_base_norm_10th',
     'width_midbody_10th',
     'width_midbody_norm_10th'],
            
    top8_auto_v2 = ['curvature_head_norm_abs_50th',
     'curvature_tail_abs_90th',
     'd_curvature_hips_abs_90th',
     'd_quirkiness_10th',
     'd_relative_to_head_base_angular_velocity_head_tip_abs_10th',
     'relative_to_head_base_radial_velocity_head_tip_norm_90th',
     'width_head_base_norm_10th',
     'width_midbody_norm_90th'],
    
    top8_manual_naive = ['length_50th',
     'width_midbody_50th',
    'curvature_head_abs_50th',
    'curvature_midbody_abs_50th',
    'speed_90th',
    'speed_10th',
    'motion_mode_backward_fraction',
    'motion_mode_forward_fraction'
    ],
    
    top8_manual = ['length_90th',
     'width_midbody_norm_10th',
     'curvature_hips_abs_90th',
     'curvature_head_abs_90th',
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     'd_curvature_hips_abs_90th',
     'd_curvature_head_abs_90th',
     ],
    
    top16_auto_v1 = ['curvature_head_norm_abs_50th',
     'curvature_head_norm_abs_90th',
     'curvature_tail_abs_90th',
     'd_angular_velocity_neck_abs_10th',
     'd_curvature_hips_abs_90th',
     'd_length_IQR',
     'd_major_axis_norm_10th',
     'd_quirkiness_10th',
     'd_relative_to_head_base_angular_velocity_head_tip_abs_10th',
     'minor_axis_50th',
     'relative_to_head_base_radial_velocity_head_tip_norm_90th',
     'relative_to_hips_radial_velocity_tail_tip_90th',
     'relative_to_neck_radial_velocity_head_tip_norm_50th',
     'width_head_base_norm_10th',
     'width_midbody_norm_90th',
     'width_tail_base_norm_50th'],
    
    top16_auto_v2 =['curvature_head_norm_abs_50th',
     'curvature_head_norm_abs_90th',
     'curvature_midbody_norm_abs_50th',
     'curvature_tail_abs_90th',
     'd_area_IQR',
     'd_curvature_hips_abs_90th',
     'd_length_IQR',
     'minor_axis_50th',
     'motion_mode_forward_frequency',
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     'quirkiness_50th',
     'relative_to_neck_radial_velocity_head_tip_norm_50th',
     'width_head_base_norm_10th',
     'width_midbody_10th',
     'width_midbody_norm_10th'],
    
    top16_auto_v3 = ['curvature_head_norm_IQR',
     'curvature_tail_norm_abs_90th',
     'd_curvature_neck_abs_90th',
     'd_relative_to_head_base_angular_velocity_head_tip_abs_10th',
     'd_speed_head_tip_90th',
     'd_width_head_base_IQR',
     'length_90th',
     'motion_mode_backward_frequency',
     'motion_mode_paused_fraction',
     'motion_mode_paused_frequency',
     'path_coverage_head',
     'quirkiness_50th',
     'relative_to_head_base_radial_velocity_head_tip_50th',
     'width_head_base_norm_10th',
     'width_midbody_norm_10th',
     'width_midbody_norm_90th'],
    
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
    
     ],
    )
#%%    
if __name__ == '__main__':
    pool_size = 15
    #cuda_id = 0
    cuda_id = None
    
    train_args = dict(n_epochs = 250,  batch_size = 250, lr = 0.01, momentum = 0.9)
    test_size = 0.2
    experimental_dataset = 'SWDB'
    
#    test_size = 1/3
#    experimental_dataset = 'CeNDR'
#    train_args = dict(n_epochs = 150,  batch_size = 250,  lr = 0.01, momentum = 0.9)

#    test_size = 1/3
#    experimental_dataset = 'MMP'
#    train_args = dict(n_epochs = 40,  batch_size = 150,  lr = 0.01, momentum = 0.9)

    
#    test_size = 1/3
#    experimental_dataset = 'Syngenta'
#    train_args = dict(n_epochs = 40,  batch_size = 150,  lr = 0.01, momentum = 0.9)
    
    #%%
    results_dir = os.path.join(results_root_dir, 'simple')
    bn = 'model_results_tiny_{}.pkl'.format(experimental_dataset)
    save_name = os.path.join(results_dir, bn)
    
    #%%
    feat_data, col2ignore_r = read_feats(experimental_dataset)
    if 'all' in feat_data:
        del feat_data['all']
        del feat_data['OW']
        
    #%%
    df_filt = feat_data['tierpsy']
    all_feat_cols = [x for x in df_filt.columns if not x in col2ignore_r]
    strain_dicts = {x[1]:x[0] for x in df_filt[['strain', 'strain_id']].drop_duplicates().values.tolist()}
    cols_types['all'] = all_feat_cols
    
    #%%
    random_state = 777
    n_folds = 6
    
    #dum variables keep for compatibility
    metric2exclude = 'loss'
    n_feats2remove = 'log2'
    
    fold_param = (cuda_id, train_args, metric2exclude, n_feats2remove)
    all_data_in = fold_generator(df_filt, cols_types, n_folds, test_size, fold_param)
    #%%
    p = mp.Pool(pool_size)
    results = p.map(get_softmax_clf, all_data_in)
    #%%
    
    with open(save_name, "wb" ) as fid:
        pickle.dump((strain_dicts, results), fid)
    
    #%%
    res_db = {}
    for (set_type, i_fold), dat in results:
        if set_type not in res_db:
            res_db[set_type] = []
        res_db[set_type].append(dat)
    
    for set_type, dat in res_db.items():
        res_db[set_type] = list(zip(*dat))
        
    for n, m_type in enumerate(['loss', 'acc', 'f1']):
        if m_type == 'loss':
            continue
        
        print(m_type, '**************')
        for set_type, dat in res_db.items():
            vv = dat[n]
            
            dd = '{} : {:.2f} {:.2f}'.format(set_type, np.mean(dat[n]), np.std(dat[n]))
            print(dd)
   
        