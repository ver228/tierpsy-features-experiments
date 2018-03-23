#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:57:32 2018

@author: ajaver
"""

import pandas as pd
import tables
import numpy as np
import os

from tierpsy_features import ventral_signed_columns
from collect_data import read_light_data, get_pulses_indexes
        
v_feats = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]

delta4pulse = [0, 65, 130, 195, 260, 325] 
beforeT = 120



def _centred_feat_df(mask_file, feat_file):
    with pd.HDFStore(feat_file) as fid:
        timeseries_data = fid['/timeseries_data']
        trajectories_data = fid['/trajectories_data']
    
    
    light_on = read_light_data(mask_file, trajectories_data)
    
    
    assert trajectories_data['timestamp_raw'].max() + 1 == len(light_on)
        
    timeseries_data[v_feats] = timeseries_data[v_feats].abs()
    
    with tables.File(feat_file) as fid:
        fps = fid.get_node('/trajectories_data')._v_attrs['fps']
    
    min_pulse_size = fps*min_pulse_size_s
    turn_on, turn_off = get_pulses_indexes(light_on, min_pulse_size)
    
    
    assert len(turn_on) == len(delta4pulse)
    
    centered_data = []
    for ii, t_on in enumerate(turn_on):
        
        ini, fin = -25*fps, 35*fps
        if ii == 0:
            ini = -beforeT*fps
        elif ii == len(turn_on)-1:
            fin = (90+45)*fps
        
        
        tb = (timeseries_data['timestamp']-t_on)
        good = (tb>ini) & (tb<fin)
        tb = tb[good]/fps + delta4pulse[ii] + beforeT
        
        
        
        feats = timeseries_data[good].copy()
        feats['timestamp_centered'] = tb
        centered_data.append(feats)
    centered_df = pd.concat(centered_data)
    
    
    return centered_df
#%%
if __name__ == '__main__':
    min_pulse_size_s = 3
    save_dir = './data'
    
    #used to calculate the size of the bins
    q_bins = (0.005, 0.995)
    all_bin_ranges = []
    
    exp_df = pd.read_csv('./data/exp_info.csv', index_col=0)
    
    pulse_cols = [x for x in exp_df if 'short_pulse' in x]
    inter_cols = [x for x in exp_df if 'inter_pulses' in x]
    
    #only keep experiments where all the short pulses last more than 3s and less than 11s
    good = ~np.any((exp_df[pulse_cols]<3) | (exp_df[pulse_cols]>11), axis=1)
    good = good & np.all((exp_df[inter_cols]>25), axis=1)
    exp_df_l = exp_df[good].copy()
    
    
    exp_df_l['exp_type'] = exp_df_l['exp_type'].str.lower()
    assert len(exp_df_l['exp_type'].unique()) == 2
    
    #fix a wrongly named strains
    exp_df_l = exp_df_l.replace({'strain':{'HRB222':'HBR222'}})
    
    #save data
    exp_df_l.to_csv(os.path.join(save_dir, 'index.csv'))
    
    all_binned_data = []
    for strain, s_dat in exp_df_l.groupby('strain'):
        
        data_indexes = {'etoh':[], 'atr':[]}
        centered_data = []
        
        for nn, (irow, row) in enumerate(s_dat.iterrows()):
            print(strain, nn+1, len(s_dat))
            mask_file = row['mask_file']
            feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5').replace('MaskedVideos', 'Results')
            
            try:
                centered_df = _centred_feat_df(mask_file, feat_file)
            except AssertionError:
                print('BAD: ', feat_file)
                continue
            
            centered_df['exp_row'] = irow
            centered_data.append(centered_df)
            
            data_indexes[row['exp_type']].append(irow)
        centered_data = pd.concat(centered_data)
        
        centered_data.to_hdf(os.path.join(save_dir, '{}.hdf5'.format(strain)), 'centered_data')
        
        
        q = centered_data.quantile(q_bins)
        all_bin_ranges.append(q)
        
    q_bot = [q.loc[q_bins[0]] for q in all_bin_ranges]
    q_bot = pd.concat(q_bot, axis=1).min(axis=1)
    
    q_top = [q.loc[q_bins[1]] for q in all_bin_ranges]
    q_top = pd.concat(q_top, axis=1).max(axis=1)
    
    q_lims = pd.concat((q_bot, q_top), axis=1)
    q_lims.columns = ['bot', 'top']
    
    q_lims.to_csv(os.path.join(save_dir, 'bin_limits.csv'))