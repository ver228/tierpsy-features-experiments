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
from tierpsy_features import timeseries_feats_columns
from numba import jit
from scipy.stats import entropy
from itertools import combinations
import warnings

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

from scipy.misc import comb

import pickle

import multiprocessing as mp
from center_data import delta4pulse, beforeT

all_feats = timeseries_feats_columns
all_feats = [x for x in all_feats if not 'path_curvature' in x]


def get_bins_ranges(bin_ranges_file, num_ybins, xbins_args):
    #%%
    bin_lims = pd.read_csv(bin_ranges_file, index_col=0)
    xbins = np.arange(*xbins_args)
    num_xbins = len(xbins) + 1
    
    ybins = {}
    for feat in all_feats:
        bot, top = bin_lims.loc[feat].values
        ybins[feat] = np.linspace(bot, top, num_ybins-1)
    #%%
    return xbins, num_xbins, ybins, num_ybins

def digitize_df(centered_data, xbins, ybins):
    binned_data = {}
    binned_data['exp_row'] = centered_data['exp_row'].values
    binned_data['timestamp'] = np.digitize(centered_data['timestamp_centered'], xbins)
    
    for feat in all_feats:
        bot, top = ybins[feat][0], ybins[feat][-1]
        dat = np.clip(centered_data[feat], bot + 1e-6, top - 1e-6)
        counts = np.digitize(dat, ybins[feat])
        
        #flag bad rows with -1
        counts[np.isnan(dat)] = -1
        binned_data[feat] = counts
    
    binned_data = pd.DataFrame(binned_data)
    return binned_data

@jit(nopython=True)
def _calc_histograms(x_digit, y_digit, n_x, n_y):
    H = np.zeros((n_y, n_x))
    for ii in range(y_digit.size):
        y = y_digit[ii]
        if y >= 0:
            x = x_digit[ii]
            H[y, x] += 1
                
    return H

def _process_strain(strain_name):
    #%%
    bin_ranges_file = os.path.join(save_dir, 'bin_limits.csv')
    
    num_ybins = 26
    xbins_args = (0, 580, delT)
    
    xbins, num_xbins, ybins, num_ybins = get_bins_ranges(bin_ranges_file,
                                                         num_ybins,
                                                         xbins_args
                                                         )
    #%%
    fname = os.path.join(save_dir, '{}.hdf5'.format(strain_name))
    
    with pd.HDFStore(fname, 'r') as fid:
        centered_data = fid['centered_data']
        centered_data = centered_data.dropna(how='all')
    #%%
    binned_data = digitize_df(centered_data, xbins, ybins)
    pbar = tqdm.tqdm(binned_data.groupby('exp_row'), desc=strain_name)
    
    all_exp = []
    
    for exp_row, exp_binned_dat in pbar:
        exp_hist = {}
        x_digit = exp_binned_dat['timestamp'].values
        
        for feat in all_feats:
            y_digit = exp_binned_dat[feat].values
            exp_hist[feat] = _calc_histograms(x_digit, y_digit, num_xbins, num_ybins)
        all_exp.append((exp_row, exp_hist))
    #%%
    return all_exp
#%%
#tot_comb = comb(len(irows), atr_N) #calculate the total number of combinations
#pbar = tqdm.tqdm(combinations(irows, atr_N), desc=desc, total=tot_comb)

def get_JSD(_P, _Q):
    eps = np.spacing(1) #matlab epsilon
    _P = _P/_P.sum() + eps
    _Q = _Q/_Q.sum() + eps
    _M = 0.5 * (_P + _Q)
    JS =  0.5 * np.sum(_P*np.log2(_P/_M) + _Q*np.log2(_Q/_M))
    return JS


#%%
def get_hist_stats(experiments_data, hist_dict, map_r = None, map_axis_add = (0,)):
    
    strains_data = {}
    strain_avg_hist = {}
    for strain, exp_dat in tqdm.tqdm(experiments_data.groupby('strain')):
        irows = exp_dat.index
        
        is_atr = exp_dat['exp_type'] == 'atr'
        atr_N = is_atr.sum()
        
        
        strains_data[strain] = {}
        strain_avg_hist[strain] = {}
        for feat in hist_dict:
            #feat = 'speed'
            H = hist_dict[feat][irows]
            
            if map_r is not None:
                H = H[:, :, map_r[0]:map_r[1]]
            
            H_atr = np.sum(H[is_atr], axis = map_axis_add)
            H_ctr = np.sum(H[~is_atr], axis = map_axis_add)
            
            strain_avg_hist[strain][feat] = H_atr, H_ctr
            
            jsd_real = get_JSD(H_atr, H_ctr)
            
            #permutation test
            perp_jsd = []
            nn = H.shape[0]
            for atr_rows in map(list, combinations(range(nn), atr_N)):
                good = np.zeros(nn, np.bool)
                good[atr_rows] = 1
                P = np.sum(H[good], axis = map_axis_add)
                Q = np.sum(H[~good], axis = map_axis_add)
                res = get_JSD(P, Q)
                perp_jsd.append(res)
            
            #get the pvalues 
            tot_comb = comb(len(irows), atr_N) #calculate the total number of combinations
            nn = (np.array(perp_jsd) > jsd_real).sum()
            p_vals = (nn + 1)/(tot_comb + 1)
                
            strains_data[strain][feat] = (jsd_real, p_vals)
            
    return strains_data, strain_avg_hist
#%%
def plot_pulses_long(s_avg_hist, s_stat_dict):
   xbins, num_xbins, ybins, num_ybins = get_bins_ranges('./data/bin_limits.csv',
                                                         num_ybins = 26,
                                                         xbins_args = (0, 580, delT)
                                                         )
    
   for strain, exp_dat in tqdm.tqdm(exp_df.groupby('strain'), desc = 'Plotting data'):
        stat_dict = s_stat_dict[strain]
        pvals_t = [(k, x[1]) for k,x in stat_dict.items()]
        pvals_t = sorted(pvals_t, key = lambda x : x[-1])
        save_name = '{}_long.pdf'.format(strain)
    
        with PdfPages(save_name) as fid_pdf:
            for feat, pval in pvals_t:
                
                xx = ybins[feat]
                
                h_atr, h_ctr = s_avg_hist[strain][feat]
                h_atr  = h_atr/h_atr.sum()
                h_ctr  = h_ctr/h_ctr.sum()
                
                fig = plt.figure(figsize = (4,3))
                plt.plot(xx, h_ctr[:-1])
                plt.plot(xx, h_atr[:-1])
                
                tt = '{} | pval:{:.04}'.format(feat, pval)
                plt.suptitle(tt, fontsize = 8)
                
                fid_pdf.savefig(fig)
                plt.close(fig)  
#%%
def plot_pulses_short(s_hist_dict, s_stat_dict, pulse_r):
    
    for strain, exp_dat in tqdm.tqdm(exp_df.groupby('strain'), desc = 'Plotting data'):
        stat_dict = s_stat_dict[strain]
        pvals_t = [(k, x[1]) for k,x in stat_dict.items()]
        pvals_t = sorted(pvals_t, key = lambda x : x[-1])
        
        
        good = exp_dat['exp_type'] == 'atr'
        atr_inds = exp_dat.index[good]
        ctr_inds = exp_dat.index[~good]
        
        save_name = '{}_short.pdf'.format(strain)
    
        with PdfPages(save_name) as fid_pdf:
            for feat, pval in pvals_t:
                h_ctr = np.nansum(s_hist_dict[feat][ctr_inds], axis=0)
                h_atr = np.nansum(s_hist_dict[feat][atr_inds], axis=0)
            
                fig_l = (10, 5)
                fig, axs = plt.subplots(1,2, figsize = fig_l, sharex=True)
                
                axs[0].imshow(h_ctr, aspect='auto', interpolation='none', cmap="inferno")
                axs[0].invert_yaxis()
                axs[0].set_title('ctr'.format(strain))
                
                axs[1].imshow(h_atr, aspect='auto', interpolation='none', cmap="inferno")
                axs[1].invert_yaxis()
                axs[1].set_title('atr'.format(strain))
                
                ini_x, fin_x = pulse_r
                for ii in range(len(axs)):
                    ini_y, fin_y = axs[ii].get_ylim()
                    
                    for ii in range(len(axs)):
                        axs[ii].plot((ini_x,ini_x), (ini_y, fin_y), 'g:')
                        axs[ii].plot((fin_x,fin_x), (ini_y, fin_y), 'g:')
                
                
                tt = '{} | pval:{}'.format(feat, pval)
                plt.suptitle(tt)
                
                fid_pdf.savefig(fig)
                plt.close(fig) 
#%%
if __name__ == '__main__':
    
    delT = 1
    
    #save_dir = './data'
    
    root_dir = '/Users/ajaver/OneDrive - Imperial College London/'
    save_dir = root_dir + 'tierpsy_features_experiments/optogenetics/data/'

    
    exp_df_o = pd.read_csv(os.path.join(save_dir, 'index.csv'), index_col=0)
    uStrains = exp_df_o['strain'].unique()
    
    p = mp.Pool()
    hist_dat =  p.map(_process_strain, uStrains)
    hist_dat = sum(hist_dat, [])
    #%%
    exp_ids, _ = zip(*hist_dat)
    assert len(set(exp_ids)) == len(exp_ids)
    
    exp_ids = np.array(exp_ids)
    tot_exp = np.max(exp_ids) + 1
    
    #make sure you do not include experiments that were not centered correctly
    exp_df = exp_df_o.loc[exp_ids]
    #%%
    nx, ny = hist_dat[0][1]['speed'].shape
    
    hist_packed = {}
    for exp_row, h_dict in hist_dat:
        for feat, dat in h_dict.items():
            if not feat in hist_packed:
                hist_packed[feat] = np.full((tot_exp, nx, ny), np.nan)
            hist_packed[feat][exp_row] = dat
    #%%
    short_pulse_ranges = []
    
    del_ini = 10//delT
    del_fin = 20//delT
    for pulse_time in delta4pulse[:-1]:
        ini_x = (pulse_time + beforeT)//delT
        short_pulse_ranges.append((ini_x-del_ini, ini_x+del_fin))
    
    del_ini = 10//delT
    del_fin = 105//delT
    ini_x = (delta4pulse[-1] + beforeT)//delT
    long_pulse_range = (ini_x-del_ini, ini_x+del_fin)
    
    #%%
    hist_long_pulse = {}
    for feat, H in hist_packed.items():
        hist_long_pulse[feat] = H[:, :, long_pulse_range[0]:long_pulse_range[-1]]
    
    hist_short_pulse = {}
    w_size = short_pulse_ranges[0][1] - short_pulse_ranges[0][0]
    for feat, H in hist_packed.items():
        tot, nn, _ = H.shape
        hist_short_pulse[feat] = np.zeros((tot, nn, w_size))
        for ini, fin in short_pulse_ranges:
            hist_short_pulse[feat] += H[:, :, ini:fin]
    

    
    #%%
    short_map_r = np.array((0, 30))//delT
    long_map_r = np.array((0, 125))//delT
    
    stats_short_pulse, avg_hist_short_pulse = get_hist_stats(exp_df, 
                                       hist_short_pulse, 
                                       map_r = short_map_r)
    
    stats_long_pulse, avg_hist_long_pulse = get_hist_stats(exp_df, 
                                      hist_long_pulse, 
                                      map_r = long_map_r)
    #%%
    with open('plot_pulses_short_hist.p', 'wb') as fid:
        dat = (avg_hist_short_pulse, stats_short_pulse)
        pickle.dump(dat,fid)
    with open('plot_pulses_long_hist.p', 'wb') as fid:
        dat = (avg_hist_long_pulse, stats_long_pulse)
        pickle.dump(dat,fid)
    
    #%%
    #plot_pulses_short(hist_short_pulse, stats_short_pulse, short_map_r)
    #plot_pulses_long(avg_hist_long_pulse, stats_long_pulse)
    #%%
    
    #sorted([(k, x[1]) for k,x in stats_long_pulse['AQ2052'].items()], key = lambda x : x[1])
    
        
        