#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:12:43 2018

@author: ajaver
"""
from numba import jit
import pickle
import numpy as np
from scipy.stats import entropy
from optogenetics_grouped import all_feats

def _digital2histograms(dat, n_x, n_y, min_valid_counts = 500):
    tot = n_x*n_y
    all_hist = {}
    x_digit = dat['timestamp'].values
    for feat in all_feats:
        y_digit = dat[feat].values
        good = y_digit >= 0
        flat_digit = y_digit[good]*n_x + x_digit[good]
        flat_counts = np.bincount(flat_digit, minlength=tot)
            
        H = np.reshape(flat_counts, (n_y, n_x))
        N = H.sum(axis=0)
        P = H/N
        P[:, N<min_valid_counts] = np.nan
        all_hist[feat] = P
    
    return all_hist

def _getJSD(binned_df, atr_inds, n_xbins, n_ybins):
    good = binned_df['exp_row'].isin(atr_inds)
    
    atr_dat = binned_df[good]
    ctr_dat = binned_df[~good]
    
    aP_atr = _digital2histograms(atr_dat, n_xbins, n_ybins)
    aP_ctr = _digital2histograms(ctr_dat, n_xbins, n_ybins)
    
    all_JSD = []
    for feat in aP_atr:
        _P = aP_atr[feat]
        _Q = aP_ctr[feat]
        _M = 0.5 * (_P + _Q)
        JS =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))
        
        all_JSD.append((feat, JS, _P, _Q))
    return all_JSD


def get_JSD(_P, _Q):
    _M = 0.5 * (_P + _Q)
    JS =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))
    return JS

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

def _getJSD_opt(binned_df, atr_inds, n_xbins, n_ybins, all_feats):
    
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
        
        JS = get_JSD(H_atr, H_ctr)
        all_hist.append((feat, JS, H_ctr, H_atr))
    
    return all_hist
    


if __name__ == '__main__':
    DD = pickle.load( open( "hist_data.p", "rb" ) )
    (binned_data, all_irows, n_atr, num_xbins, num_ybins) = DD
    
    binned_data = binned_data[(binned_data>=0).all(axis=1)]
    #%%
    import tqdm
    for nn in tqdm.tqdm(range(100)):
        res = _getJSD_opt(binned_data, all_irows[:n_atr], num_xbins, num_ybins, all_feats)
    
    import matplotlib.pylab as plt
    plt.imshow(res['speed'][2])