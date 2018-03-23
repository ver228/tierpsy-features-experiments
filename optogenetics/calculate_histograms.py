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

from center_data import delta4pulse, beforeT

all_feats = timeseries_feats_columns + ['d_' + x for x in timeseries_feats_columns]
all_feats = [x for x in all_feats if not 'path_curvature' in x]


def get_bins_ranges(bin_ranges_file, num_ybins = 26, xbins_args = (0, 580, 5)):
    
    bin_lims = pd.read_csv(bin_ranges_file, index_col=0)
    xbins = np.arange(*xbins_args)
    num_xbins = len(xbins) + 1
    
    ybins = {}
    for feat in all_feats:
        bot, top = bin_lims.loc[feat].values
        ybins[feat] = np.linspace(bot, top, num_ybins-1)
    
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

def get_JSD(_P, _Q):
    _M = 0.5 * (_P + _Q)
    JS =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))
    return JS

def get_all_JSD(binned_df, atr_inds, n_xbins, n_ybins, min_valid_counts = 500, skip_nbins = 0):
    
    
    
    is_atr_d = binned_df['exp_row'].isin(atr_inds).values
    x_digit = binned_df['timestamp'].values
    
    all_hist = []
    for feat in all_feats:
        y_digit = binned_df[feat].values
        
        #2d histograms binned by video and timestamp
        C_ctr, C_atr = _calc_histograms(x_digit, y_digit, is_atr_d, n_xbins, n_ybins)
        
        #distributions binned only by feature
        c_ctr = np.nansum(C_ctr[:, skip_nbins:], axis=1)
        h_ctr = c_ctr/np.sum(c_ctr)
        
        c_atr = np.nansum(C_atr[:, skip_nbins:], axis=1)
        h_atr = c_atr/np.sum(c_atr)
        js = get_JSD(h_atr, h_ctr)
        
        
        N_atr = C_atr.sum(axis=0)
        N_ctr = C_ctr.sum(axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H_atr = C_atr/N_atr
            H_ctr = C_ctr/N_ctr
        
        H_atr[:, N_atr<min_valid_counts] = np.nan
        H_ctr[:, N_ctr<min_valid_counts] = np.nan
        
        JS = get_JSD(H_atr, H_ctr)
        
        all_hist.append((feat, (JS, H_ctr, H_atr), (js, h_ctr, h_atr)))
    
    return all_hist


def permutation_tests(all_JSD, binned_data, irows, atr_N, num_xbins, num_ybins, skip_nbins = 0, desc=''):
    all_p_JSD = {}
    
    tot_comb = comb(len(irows), atr_N) #calculate the total number of combinations
    pbar = tqdm.tqdm(combinations(irows, atr_N), desc=desc, total=tot_comb)
    
    for atr_l_p in pbar:
        p_JSD = get_all_JSD(binned_data, atr_l_p, num_xbins, num_ybins)
        p_JSD = [(x[0], x[1][0], x[2][0]) for x in p_JSD]
        for feat, jsd_ts, jsd_feat in p_JSD:
            if not feat in all_p_JSD:
                all_p_JSD[feat] = []
            all_p_JSD[feat].append((jsd_ts, jsd_feat))
    
    
    all_p_JSD = {k:list(zip(*val)) for k, val in all_p_JSD.items()}
    
    #calculate the p values
    pvals_ts = {}
    pvals_feat = {}
    for dat in all_JSD:
        feat = dat[0]
        jsd_ts = dat[1][0]
        jsd_feat = dat[2][0]
        
        p_jsd_ts, p_jsd_feat = map(np.array, all_p_JSD[feat])
        
        
        nn = (p_jsd_ts > jsd_ts[None, :]).sum(axis=0)
        pvals_ts[feat] = (nn + 1)/(tot_comb + 1)
        pvals_ts[feat][np.isinf(jsd_ts)] = np.nan
        
        nn = (p_jsd_feat > jsd_feat).sum()
        pvals_feat[feat] = (nn + 1)/(tot_comb + 1)
    
    return pvals_ts, pvals_feat, all_p_JSD, tot_comb


def calculate_strain_JSD(strain_name, save_dir = './data', delT = 5):
    
    #bins used to skip the beggining of the video in the histograms averages
    skip_nbins = round(int(100/delT))
    
    bin_ranges_file = os.path.join(save_dir, 'bin_limits.csv')
    xbins, num_xbins, ybins, num_ybins = get_bins_ranges(bin_ranges_file)
    
    fname = os.path.join(save_dir, '{}.hdf5'.format(strain_name))
    
    if not os.path.exists(fname):
        return
    
    with pd.HDFStore(fname, 'r') as fid:
        centered_data = fid['centered_data']
        centered_data = centered_data.dropna(how='all')
    
    #get row indexes, and which ones belog to the ATR condition
    all_irows = centered_data['exp_row'].unique()
    dd = exp_df.loc[all_irows, 'exp_type']
    atr_irows = dd.index[dd=='atr'].values
    atr_N = len(atr_irows)
    
    binned_data = digitize_df(centered_data, xbins, ybins)
    
    all_JSD = get_all_JSD(binned_data, atr_irows, num_xbins, num_ybins, skip_nbins = skip_nbins)
    
    pvals_ts, pvals_feat, all_pJSD, tot_comb = permutation_tests(all_JSD, 
                                         binned_data, 
                                            all_irows.tolist(), 
                                            atr_N, 
                                            num_xbins, 
                                            num_ybins,
                                            skip_nbins = skip_nbins,
                                            desc = '{} Permutation Test'.format(strain_name))
    
    def best_JSD(dat):
        return pvals_feat[dat[0]]
    all_JSD = sorted(all_JSD, key = best_JSD)
    
    pJSD_ranges = {k:np.percentile(val[0], (5, 95), axis=0) for k, val in all_pJSD.items()}
    
    with open('{}.p'.format(strain_name), 'wb') as fid:
        dat = (all_JSD, pvals_ts, pvals_feat, all_pJSD, pJSD_ranges)
        pickle.dump(dat,fid)
    
    
    
    plot_input = strain_name, tot_comb, delT, pvals_ts, pvals_feat, all_JSD, pJSD_ranges, xbins, ybins
    
    return plot_input

def save_plots(strain_name, tot_comb, delT, pvals_ts, pvals_feat, all_JSD, pJSD_ranges, xbins, ybins):
    
    save_name = '{}_hist.pdf'.format(strain_name)
    
    with PdfPages(save_name) as fid_pdf:
        for feat, (JSD, H_ctr, H_atr), _ in all_JSD:
            
            H_atr[np.isnan(H_atr)] = 0
            H_ctr[np.isnan(H_ctr)] = 0
            
            
            fig,axs = plt.subplots(4,1, figsize = (14, 9), sharex=True)
            axs[0].imshow(H_atr, aspect='auto', interpolation='none', cmap="inferno")
            axs[0].invert_yaxis()
            axs[1].imshow(H_ctr, aspect='auto', interpolation='none', cmap="inferno")
            axs[1].invert_yaxis()
            
            
            
            bot, top = pJSD_ranges[feat]
            x = np.arange(bot.size)
            
            axs[2].fill_between(x, bot, top, color = 'gray', alpha = 0.1)
            
            axs[2].plot(JSD, 'r')
            
            yl = axs[2].get_ylim()
            axs[2].set_ylim((0, yl[1]))
            
            #axs[2].fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
            
            axs[3].plot(np.log10(pvals_ts[feat]))
            axs[3].set_ylim((np.log10(1/tot_comb) - 0.05, 0.05))
            
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
            
            
            
            tt = '{} - pval: {}'.format(feat, pvals_feat[feat])
            plt.suptitle(tt)
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
            

        
if __name__ == '__main__':
    
    save_dir = './data'
    
    import multiprocessing as mp
    def _process(strain):
        plot_input = calculate_strain_JSD(strain, save_dir = save_dir, delT = 5)
        save_plots(*plot_input)
        
    exp_df = pd.read_csv(os.path.join(save_dir, 'index.csv'), index_col=0)
    
    uStrains = exp_df['strain'].unique()
    
    
    #be careful with memory here...
    p = mp.Pool(len(uStrains))
    p.map(_process, uStrains)
    
    
    #for strain in tqdm.tqdm(exp_df['strain'].unique()):
        
         

