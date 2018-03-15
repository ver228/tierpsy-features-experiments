#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:57:32 2018

@author: ajaver
"""

import pandas as pd
import tables
import matplotlib.pylab as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from tierpsy_features import timeseries_feats_columns, ventral_signed_columns
from collect_data import read_light_data, get_pulses_indexes


#%%
if __name__ == '__main__':
    min_pulse_size_s = 3
    
    exp_df = pd.read_csv('exp_info.csv', index_col=0)
    
    pulse_cols = [x for x in exp_df if 'short_pulse' in x]
    inter_cols = [x for x in exp_df if 'inter_pulses' in x]
    
    #only keep experiments where all the short pulses last more than 3s and less than 11s
    good = ~np.any((exp_df[pulse_cols]<3) | (exp_df[pulse_cols]>11), axis=1)
    good = good & np.all((exp_df[inter_cols]>25), axis=1)
    exp_df_l = exp_df[good]
    #%%
    
    all_binned_data = []
    for irow, row in exp_df_l.iterrows():
        print(irow, len(exp_df_l))
        #if not ((row['strain'] == 'AQ2052') and (row['exp_type'] == 'ATR')):
        #    continue
    
        mask_file = row['mask_file']
        light_on = read_light_data(mask_file)
        feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5').replace('MaskedVideos', 'Results')
        
        with pd.HDFStore(feat_file) as fid:
            timeseries_data = fid['/timeseries_data']
        
        timeseries_data[ventral_signed_columns] = timeseries_data[ventral_signed_columns].abs()
        
        with tables.File(feat_file) as fid:
            fps = fid.get_node('/trajectories_data')._v_attrs['fps']
        
        min_pulse_size = fps*min_pulse_size_s
        turn_on, turn_off = get_pulses_indexes(light_on, min_pulse_size)
        
        
        delT = 3
        del_frames = fps*delT
        tt = timeseries_data['timestamp']
        
        all_bins = []
        for ii, t_on in enumerate(turn_on):
            
            ini, fin = -25*fps, 35*fps
            if ii == 0:
                ini = -120*fps
            elif ii == len(turn_on)-1:
                fin = 120*fps
            
            
            tb = (tt-t_on)
            good = (tb>ini) & (tb<fin)
            tb = tb[good]
            tb = np.ceil(tb/del_frames).astype(np.int32)
            
            feats = timeseries_data[good].copy()
            feats['ind_t'] = tb
            y_binned = feats.groupby('ind_t').agg(np.median)
            y_binned.index = y_binned.index*delT
            all_bins.append(y_binned)
        
        
        
        row_id = (row['day'], row['strain'], row['exp_type'])
        
        all_binned_data.append((row_id, all_bins))
    #%%
    dat2plot = {} 
    for (day, strain, exp_type), all_bins in all_binned_data:
        if not strain in dat2plot:
            dat2plot[strain] = {}
            
        if not exp_type in dat2plot[strain]:
            dat2plot[strain][exp_type] = []
        
        dat2plot[strain][exp_type].append(all_bins)
    #%%
    delta4pulse = [0, 65, 130, 195, 260, 325]
    for i_strain, (strain, exp_dict) in enumerate(dat2plot.items()):
        print(i_strain, len(dat2plot), strain)
        fid_pdf = PdfPages('{}_{}s.pdf'.format(strain, delT))
        
        for feat in sorted(timeseries_feats_columns):
            #fig, axs = plt.subplots(2,6, figsize=(18, 6), sharey=True)
            fig, axs = plt.subplots(2,1, figsize=(18, 6), sharey=True)
            for exp_type, feat_bins in exp_dict.items():
                i_exp = 0 if exp_type == 'ATR' else 1
                
                for i_bin, feat_bin in enumerate(zip(*feat_bins)):
                    
                    
                    
                    ff = [x[feat] for x in feat_bin]
                    xx, yy = map(np.array, zip(*[(x.index, x.values) for x in ff]))
                    
                    ff_m = pd.concat(ff, axis=1).median(axis=1)
                    x_m, y_m = ff_m.index, ff_m.values 
                    
                    #for x,y in zip(xx,yy):
                    #    axs[i_exp][i_bin].plot(x + delta4pulse[i_bin], y)
                    #axs[i_exp][i_bin].plot(x_m + delta4pulse[i_bin], y_m, 'k', lw=3)
                    
                    
                    for x,y in zip(xx,yy):
                        axs[i_exp].plot(x + delta4pulse[i_bin], y, 'gray')
                    axs[i_exp].plot(x_m + delta4pulse[i_bin], y_m, 'r', lw=3)
                    
                ini_y, fin_y = axs[i_exp].get_ylim()
                for i_bin in range(6):
                    if i_bin == 5:
                        delP = 90
                    else:
                        delP = 5
                    
                    p = patches.Rectangle( (delta4pulse[i_bin], ini_y), 
                                  delP, 
                                  fin_y-ini_y,
                                  alpha=0.2, 
                                  color = 'steelblue')
                    axs[i_exp].add_patch(p)
                    
                    
                plt.suptitle(feat)
                
            fid_pdf.savefig(fig)
            plt.close(fig)
            
        fid_pdf.close()    
        