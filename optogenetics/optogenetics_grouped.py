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
import random
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from numba import jit
from scipy.stats import entropy
import tqdm
import pickle
from tierpsy_features import timeseries_feats_columns, ventral_signed_columns
from collect_data import read_light_data, get_pulses_indexes
        


all_feats = timeseries_feats_columns + ['d_' + x for x in timeseries_feats_columns]
all_feats = [x for x in all_feats if not 'path_curvature' in x]
v_feats = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]

delta4pulse = [0, 65, 130, 195, 260, 325] 
beforeT = 120

def _centred_feat_df(mask_file, feat_file):
    
    light_on = read_light_data(mask_file)
    
    
    with pd.HDFStore(feat_file) as fid:
        timeseries_data = fid['/timeseries_data']
    
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

@jit(nopython=True)
def _calc_histograms(x_digit, y_digit, is_atr_d, n_y, n_x):
    H_atr = np.zeros((n_x, n_y))
    H_ctr = np.zeros((n_x, n_y))
    for ii in range(y_digit.size):
        x = x_digit[ii]
        y = y_digit[ii]
        is_atr = is_atr_d[ii]
        
        if y >= 0:
            if is_atr:
                H_atr[y, x] += 1
            else:
                H_ctr[y, x] += 1
                
    return H_ctr, H_atr

def _get_JSD(_P, _Q):
    _M = 0.5 * (_P + _Q)
    JS =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))
    return JS

def _get_all_JSD(binned_df, atr_inds, n_xbins, n_ybins, ):
    
    min_valid_counts = 500
    
    is_atr_d = binned_df['exp_row'].isin(atr_inds).values
    x_digit = binned_df['timestamp'].values
    
    all_hist = []
    for feat in all_feats:
        y_digit = binned_df[feat].values
        H_ctr, H_atr = _calc_histograms(x_digit, y_digit, is_atr_d, n_xbins, n_ybins)
        
        N_atr = H_atr.sum(axis=0)
        H_atr /= N_atr
        H_atr[:, N_atr<min_valid_counts] = np.nan
        
        N_ctr = H_ctr.sum(axis=0)
        H_ctr /= N_ctr
        H_ctr[:, N_ctr<min_valid_counts] = np.nan
        
        JS = _get_JSD(H_atr, H_ctr)
        all_hist.append((feat, JS, H_ctr, H_atr))
    
    return all_hist


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
    
    
    exp_df_l['exp_type'] = exp_df_l['exp_type'].str.lower()
    assert len(exp_df_l['exp_type'].unique()) == 2
    
    #fix a wrongly named strains
    exp_df_l = exp_df_l.replace({'strain':{'HRB222':'HBR222'}})
    
    all_binned_data = []
    for strain, s_dat in exp_df_l.groupby('strain'):
        
        data_indexes = {'etoh':[], 'atr':[]}
        centered_data = []
        #%%
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
        #%%
        delT = 5
        num_ybins = 26
        
        dat = centered_data['timestamp_centered']
        
        top = dat.max()
        bot = dat.min()
        xbins = np.arange(bot, top, delT)
        num_xbins = len(xbins) + 1
        
        binned_data = {}
        binned_data['exp_row'] = centered_data['exp_row'].values
        binned_data['timestamp'] = np.digitize(dat, xbins)
        
        ybins = {}
        for feat in all_feats:
            dat = centered_data[feat]
            q = (0.01, .99)
            bot, top = dat.quantile(q)
            
            ybins[feat] = np.linspace(bot, top, num_ybins-1)
            dat = np.clip(dat, bot + 1e-6, top - 1e-6)
            counts = np.digitize(dat, ybins[feat])
            
            
            #flag bad rows with -1
            counts[np.isnan(dat)] = -1
            binned_data[feat] = counts
        
        binned_data = pd.DataFrame(binned_data)
        
        all_JSD = _get_all_JSD(binned_data, data_indexes['atr'], num_xbins, num_ybins)
        
        #%%
        n_permutations = 10000
        
        n_atr = len(data_indexes['atr'])
        all_irows = data_indexes['atr'] + data_indexes['etoh']
        
        
        #%%
        all_p_JSD = {}
        pbar = tqdm.tqdm(range(n_permutations), desc='Permutation Test')
        for i_p in pbar:
            random.shuffle(all_irows)
            p_JSD = _get_all_JSD(binned_data, all_irows[:n_atr], num_xbins, num_ybins)
            p_JSD = [x[:2] for x in p_JSD]
            for feat, JSD in p_JSD:
                if not feat in all_p_JSD:
                    all_p_JSD[feat] = []
                all_p_JSD[feat].append(JSD)
        
        #%%
        pvals = {}
        for dat in all_JSD:
            feat = dat[0]
            JSD = dat[1]
            p_JSD = all_p_JSD[feat]
            p_JSD = np.array(p_JSD)
            
            nn = (p_JSD > JSD[None, :]).sum(axis=0)
            pvals[feat] = (nn + 1)/(n_permutations + 1)
            
            
            pvals[feat][np.isinf(JSD)] = np.nan
            
        #%%
        
        
        
        dat_p_JSD = {k:np.percentile(val, (5, 95), axis=0) for k,val in all_p_JSD.items()}
        #%% Sort using the largest change
        def best_JSD(dat):
            feat, JSD, H_atr, H_ctr = dat
            val = np.sum(pvals[feat]<0.005)
            if val == 0:
                val = 1-np.nanmin(pvals[feat])
            
            return val
        
        all_JSD = sorted(all_JSD, key = best_JSD)[::-1]
        #%%
        with PdfPages('{}_hist.pdf'.format(strain)) as fid_pdf:
            for feat, JSD, H_ctr, H_atr in all_JSD:
                
                H_atr[np.isnan(H_atr)] = 0
                H_ctr[np.isnan(H_ctr)] = 0
                
                
                fig,axs = plt.subplots(4,1, figsize = (14, 9), sharex=True)
                axs[0].imshow(H_atr, aspect='auto', interpolation='none', cmap="inferno")
                axs[0].invert_yaxis()
                axs[1].imshow(H_ctr, aspect='auto', interpolation='none', cmap="inferno")
                axs[1].invert_yaxis()
                
                
                
                bot, top = dat_p_JSD[feat]
                x = np.arange(bot.size)
                
                axs[2].fill_between(x, bot, top, color = 'gray', alpha = 0.1)
                
                axs[2].plot(JSD, 'r')
                
                yl = axs[2].get_ylim()
                axs[2].set_ylim((0, yl[1]))
                
                #axs[2].fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
                
                axs[3].plot(np.log10(pvals[feat]))
                axs[3].set_ylim((np.log10(1/n_permutations) - 0.05, 0.05))
                
                for nn in [2,3]:
                    ini_y, fin_y = axs[nn].get_ylim()
                    for i_bin in range(6):
                        if i_bin == 5:
                            delP = 90
                        else:
                            delP = 5
                        
                        ini_x = (delta4pulse[i_bin] + beforeT)/delT
                        
                        p = patches.Rectangle( (ini_x, ini_y), 
                                      delP/delT, 
                                      fin_y-ini_y,
                                      alpha=0.5, 
                                      color = 'steelblue')
                        axs[nn].add_patch(p)
                
                
                plt.suptitle(feat)
                plt.xlim((-1, H_atr.shape[1]))
                
                tlabs = []
                for tt in axs[1].get_yticks():
                    if tt >=0 and tt <= len(ybins[feat]):
                        tlabs.append('{:0.2f}'.format(ybins[feat][int(tt)]))
                    else:
                        tlabs.append('')
                
                axs[0].set_yticklabels(tlabs)
                axs[1].set_yticklabels(tlabs)
                
                tlabs = []
                for tt in axs[2].get_xticks():
                    if tt >=0 and tt <= len(xbins):
                        tlabs.append(str(int(xbins[int(tt)])))
                    else:
                        tlabs.append('')
                axs[2].set_xticklabels(tlabs)
                plt.xlabel('Time [s]')
                
                fid_pdf.savefig(fig)
                plt.close(fig) 
        
        with open('{}.p'.format(strain), 'wb') as fid:
            dat = (all_JSD, all_p_JSD, pvals)
            pickle.dump(dat,fid) 
        
        
        