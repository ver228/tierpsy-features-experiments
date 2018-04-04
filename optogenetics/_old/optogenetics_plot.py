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

#REGION_LABELS = {'after': 1,
# 'before': 0,
# 'inter_pulses_1': 8,
# 'inter_pulses_2': 9,
# 'inter_pulses_3': 10,
# 'inter_pulses_4': 11,
# 'inter_pulses_5': 12,
# 'long_pulse': 2,
# 'short_pulse_1': 3,
# 'short_pulse_2': 4,
# 'short_pulse_3': 5,
# 'short_pulse_4': 6,
# 'short_pulse_5': 7}
#
#AQ2028 - to goes back a lot...
#
#
#'curvature_midbody'
#
#curvature_head_10th
if __name__ == '__main__':
    exp_df_l = pd.read_csv('exp_info.csv', index_col=0)
    #exp_df_l = exp_df_l[~exp_df_l['day'].isin(['day1'])] 
    exp_df_l = exp_df_l[exp_df_l['day'] != 'day1']#day1 is likely to have problems
    
    #strains = ('AQ2028', 'AQ2052', 'AQ2232', 'AQ2235', 'HBR222', 'HBR520')
    strains = exp_df_l['strain'].unique()#('AQ2052',)
    
    
    
    
    for ss in strains:
        print(ss)
        tot = 0
        exp_df = exp_df_l[exp_df_l['strain'] == ss]
        
        # Just for debugging, if i do not find pulses it will show some graphs
        problem_files = exp_df.loc[~exp_df['has_valid_light'], 'mask_file'].values
        
        gg = exp_df.groupby('day')
        
        
        all_dat = [[] for _ in range(4)]
        #select a specific day
        for day, dat in gg: 
            plt.figure(figsize=(12,7))
            #exp_df.sort_values(by=['day', 'exp_type'])
            for _, row in dat.iterrows():
                mask_file = row['mask_file']
                with tables.File(mask_file) as fid:
                    mean_intensity = fid.get_node('/mean_intensity')[:]
                
                
                feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5').replace('MaskedVideos', 'Results')
                
                with pd.HDFStore(feat_file) as fid:
                    timeseries_data = fid['/timeseries_data']
                
                curv_feats = [x for x in timeseries_data.columns if 'curvature' in x]
                timeseries_data[curv_feats] = timeseries_data[curv_feats].abs()
                
                del_T = 25*5
                tt = timeseries_data['timestamp']/del_T
                timeseries_data['tt_ind'] = np.ceil(tt).astype(np.int)
                timeseries_data['timestamp_min'] = tt
                
                
                feat_binned = timeseries_data.groupby('tt_ind').agg(np.median)
                #%%
                ylim_d = {
                        'speed': [(-320, 320), 'Speed [$\mu$m/s]'],
                        'curvature_tail' : [(0, 0.01), 'Tail Curvature']
                        }
                
                
                for i_feat, feat_name in enumerate(ylim_d.keys()):
                    y_ll, lab_s = ylim_d[feat_name]
                    i_exp = int(row['exp_type'] == 'ATR')
                    i_plot = i_feat*2 + i_exp
                    plt.subplot(2,2, i_plot + 1)
                    
                    y = timeseries_data[feat_name]
                    y_f = feat_binned[feat_name].values
                    x_f = feat_binned.index*del_T/25
                    
                    y_ind = mean_intensity-mean_intensity.min()
                    y_ind = y_ind/y_ind.max()
                    
                    y_ind = y_ind*(y_ll[1] - y_ll[0]) + y_ll[0]
                    
                    #fig = plt.figure(figsize=(12,5))
                    
                    
                    xx = np.arange(y_ind.size)/25
                    plt.plot(xx, y_ind, lw=1.2, color='dodgerblue')
                    plt.plot(x_f, y_f, '.-', lw=2, color='rebeccapurple')
                    
                    plt.ylim(*y_ll)
                    plt.xlim(0, 900)
                    #3s_t = '{} {} {} {}'.format(row['exp_type'], feat_name, row['strain'],  row['day'])
                    if i_feat == 0:
                        plt.title(row['exp_type'])
                    
                    plt.ylabel(lab_s)
                    plt.xlabel('Time [s]')
                    #fig.savefig(s_t + '.pdf')
                    #%%
                    
                    all_dat[i_plot].append(((xx, y_ind), (x_f, y_f)))
                
            plt.suptitle(ss)
            
            plt.savefig('{}_{}.pdf'.format(row['strain'], row['day']), bbox_inches='tight')
        #%%
        
        feat_p = [ 
          ('speed', 'EtOH'),
          ('speed', 'ATR'),
          ('curvature_tail', 'EtOH'),
          ('curvature_tail', 'ATR')
         ]
        
        plt.figure(figsize=(12,7))
        for i_plot, feat_dat in enumerate(all_dat):
            feat_name, exp_type = feat_p[i_plot]
            
            
            #dat_feat, dat_int = zip(*feat_dat)
            y_ll, lab_s = ylim_d[feat_name]
            
            for ii, dd in enumerate(zip(*feat_dat)):
                xx, yy = zip(*dd)
            
                tt = min([x.size for x in xx])
                xx = [x[:tt] for x in xx]
                yy = [x[:tt] for x in yy]
            
                
                x_m = np.nanmean(xx, axis=0)
                y_m = np.nanmean(yy, axis=0)
                
                err_m = np.nanstd(yy, axis=0)/np.sqrt(len(yy))
                
                plt.subplot(2,2, i_plot + 1)
                
                if ii == 1:
                    plt.plot(x_m, y_m, '-', lw=1.7, color='rebeccapurple')
                    
                    plt.fill_between(x_m, y_m-err_m, y_m+err_m, color='rebeccapurple', alpha=0.5)
                
                else:
                    plt.plot(x_m, y_m, lw=1.2, color='dodgerblue')
            plt.ylim(*y_ll)
            plt.xlim(0, 900)
            
            if i_plot < 2:
                plt.title(exp_type)
            
            plt.ylabel(lab_s)
            plt.xlabel('Time [s]')
            
            plt.savefig('{}_avg.pdf'.format(row['strain']), bbox_inches='tight')