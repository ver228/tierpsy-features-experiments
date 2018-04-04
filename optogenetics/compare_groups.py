#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:23:47 2018

@author: ajaver
"""
import pandas as pd
import os
import tqdm
import numpy as np
import warnings

import matplotlib.pylab as plt

from calculate_histograms import digitize_df, get_all_JSD, get_bins_ranges
      
        
if __name__ == '__main__':
    save_dir = './data'
    
    exp_df = pd.read_csv(os.path.join(save_dir, 'index.csv'), index_col=0)
    uStrains = exp_df['strain'].unique()
    #uStrains = ['ZX991', 'ZX819']
    
    all_outliers_metrics = []
    for strain_name in tqdm.tqdm(uStrains):
        bin_ranges_file = os.path.join(save_dir, 'bin_limits.csv')
        xbins, num_xbins, ybins, num_ybins = get_bins_ranges(bin_ranges_file)
        
        fname = os.path.join(save_dir, '{}.hdf5'.format(strain_name))
        
        if not os.path.exists(fname):
            break
        
        with pd.HDFStore(fname, 'r') as fid:
            centered_data = fid['centered_data']
            centered_data = centered_data.dropna(how='all')
        #%%
        #get row indexes, and which ones belog to the ATR condition
        all_irows = centered_data['exp_row'].unique()
        dd = exp_df.loc[all_irows, 'exp_type']
        atr_irows = dd.index[dd=='atr'].values
        ctr_irows = dd.index[dd!='atr'].values
        atr_N = len(atr_irows)
        
        binned_data = digitize_df(centered_data, xbins, ybins)
        
        #%%
        atr_ind = binned_data['exp_row'].isin(atr_irows)
        binned_data_atr = binned_data[atr_ind]
        binned_data_ctr = binned_data[~atr_ind]
        
        #%%
        def get_outliers(df):
            all_dat = []
            for exp_row in df['exp_row'].unique():
                all_JSD = get_all_JSD(df, [exp_row], num_xbins, num_ybins, min_valid_counts=0)
                all_dat.append((exp_row, all_JSD))
            exp_row_id, all_JSD = zip(*all_dat)
            
            all_dat_c = []
            for feat_dat in zip(*all_JSD):
                ff, dat_ts, dat_feat = zip(*feat_dat)
                assert all(ff[0] == x for x in ff)
                dat = np.array([x[0] for x in dat_feat])
                all_dat_c.append(dat)
            
            all_dat_c = np.array(all_dat_c)
            
            all_dat_c[np.isinf(all_dat_c)] = np.nan
            
            return all_dat_c, exp_row_id, all_JSD
        
        #df = binned_data_atr
        DD_atr = get_outliers(binned_data_atr)
        DD_ctr = get_outliers(binned_data_ctr)
        
       
        
        
        for ss, dat in zip(('atr', 'ctr'), (DD_atr, DD_ctr)):
            all_dat_c, exp_row_id, all_JSD = dat
            
            #mu = np.nanmean(all_dat_c, axis=1)
            #var = np.nanstd(all_dat_c, axis=1)
            
            mu = np.nanmedian(all_dat_c, axis=1)
            var = np.nanmedian(np.abs(all_dat_c- mu[:, None]), axis=1)
            all_dat_z = (all_dat_c - mu[:, None])/var[:, None]
           
            plt.figure()
            plt.plot(all_dat_z, '.')
            plt.title('{} {}'.format(ss, strain_name))
            
            #cc = np.nanpercentile(all_dat_c, [99], axis=1)[0]
            #all_outliers_metrics.append((exp_row_id,cc))
            
            #for ii, dd in enumerate(cc):
                #if dd > 2:
            #    mm = all_JSD[ii][0]
            for ii, dat in enumerate(all_JSD):
                feat, mm, _ = dat[0]
                plt.figure()
                plt.subplot(2,1,1)
                plt.imshow(mm[-2])
                plt.subplot(2,1,2)
                plt.imshow(mm[-1])
                
                #plt.subplot(3,1,3)
                #plt.plot(all_dat_c[ii])
                
                rr = exp_row_id[ii]
                tt = os.path.basename(exp_df.loc[rr, 'mask_file'])
                plt.suptitle('{} {}'.format(rr, tt))
            
            cc = np.nanmedian(all_dat_z, axis=0)
            all_outliers_metrics.append((exp_row_id,cc))
    #%%
    exp_id, out_metric = map(np.concatenate, list(zip(*all_outliers_metrics)))
        
            
        #%%
        
        
#        for exp_row, exp_data in binned_data.groupby('exp_row'):
#            is_atr = exp_row in atr_irows
#            all_JSD = get_all_JSD(binned_data, [exp_row], num_xbins, num_ybins, min_valid_counts=50)
#            
#            ii = 2 if is_atr else 3
#            hh = [(x[0], x[ii]) for x in all_JSD]
#            
#            
#            plt.figure()
#            
#            dd = exp_df.loc[exp_row, ['strain', 'day', 'exp_type']].values.tolist()
#            plt.title('{}'.format(dd))
#            plt.imshow(hh[0][1])
#            
#        
#        #all_JSD = _get_all_JSD(binned_data, atr_irows, num_xbins, num_ybins)
#        
#        
