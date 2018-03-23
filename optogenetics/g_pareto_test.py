#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:57:20 2018

@author: ajaver
"""

#https://brainder.org/tag/permutation-test/

import pickle
import tqdm
import pandas as pd
import numpy as np
from calculate_histograms import get_JSD
from scipy.interpolate import interp1d

def gpdpvals(x,a,k):
    #Modified from: https://github.com/andersonwinkler/PALM/blob/master/palm_pareto.m
    # Compute the p-values for a GPD with parameters a (scale)
    # and k (shape).
    
    eps = np.spacing(1)
    if np.abs(k) < eps:
        p = np.exp(-x/a)
    else:
        p = (1 - k*x/a)**(1/k)

    if k > 0:
        p[x > a/k] = eps;


    return p
#%%
def andersondarling(z,k):
    
    #Modified from: https://github.com/andersonwinkler/PALM/blob/master/palm_pareto.m
    # Compute the Anderson-Darling statistic and return an
    # approximated p-value based on the tables provided in:
    # * Choulakian V, Stephens M A. Goodness-of-Fit Tests
    #   for the Generalized Pareto Distribution. Technometrics.
    #   2001;43(4):478-484.
    
    # This is Table 2 of the paper (for Case 3, in which 
    # a and k are unknown, bold values only)
    ktable = np.array([0.9, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5]).T;
    ptable = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]);
    A2table = np.array([
            [0.3390, 0.4710, 0.6410, 0.7710, 0.9050, 1.0860, 1.2260, 1.5590], 
            [0.3560, 0.4990, 0.6850, 0.8300, 0.9780, 1.1800, 1.3360, 1.7070], 
            [0.3760, 0.5340, 0.7410, 0.9030, 1.0690, 1.2960, 1.4710, 1.8930], 
            [0.3860, 0.5500, 0.7660, 0.9350, 1.1100, 1.3480, 1.5320, 1.9660],
            [0.3970, 0.5690, 0.7960, 0.9740, 1.1580, 1.4090, 1.6030, 2.0640], 
            [0.4100, 0.5910, 0.8310, 1.0200, 1.2150, 1.4810, 1.6870, 2.1760], 
            [0.4260, 0.6170, 0.8730, 1.0740, 1.2830, 1.5670, 1.7880, 2.3140], 
            [0.4450, 0.6490, 0.9240, 1.1400, 1.3650, 1.6720, 1.9090, 2.4750],
            [0.4680, 0.6880, 0.9850, 1.2210, 1.4650, 1.7990, 2.0580, 2.6740], 
            [0.4960, 0.7350, 1.0610, 1.3210, 1.5900, 1.9580, 2.2430, 2.9220]
            ]);
    
    #% The p-values are already sorted
    k  = max(0.5, k);
    #z  = flipud(z)';
    n  = z.size;
    j  = np.arange(1, n + 1)
    
    #% Anderson-Darling statistic and p-value:
    A2 = -n -(1/n)*((2*j-1)*(np.log(z) + np.log(1-z[n-j])))
    i1 = np.array([interp1d(ktable, a2, bounds_error=False)(k) for a2 in A2table.T])
    i2 = interp1d(i1, ptable, bounds_error=False)(A2);
    A2pval = max(min(i2,1),0);
    
    return A2pval

#%%
if __name__ == '__main__':
    exp_df = pd.read_csv('./data/index.csv', index_col=0)
    
    hist_results = {}
    for strain in tqdm.tqdm(['AQ2050']):#3exp_df['strain'].unique()):
        fname = './results/{}.p'.format(strain)
        
        with open(fname, 'rb') as fid:
            all_JSD, pvals, all_pJSD = pickle.load(fid)
            hist_results[strain] = (all_JSD, pvals)
  
        #%%
        import matplotlib.pylab as plt
        for feat, JSD, _, _ in all_JSD:
            
            dat = np.array(all_pJSD[feat])
            n_repeats, n_bins = dat.shape 
            
            feat_pvals = []
            
            
            for nn in range(n_bins):
                dat_bin = dat[:, nn]  
                jsd = JSD[nn]
                
                if np.any(np.isinf(dat_bin) | np.isnan(dat_bin)):
                    feat_pvals.append(np.nan)
                    continue
                
                
                #get the number of samples in the permutation larger than N
                N = np.sum(dat_bin>jsd)
                
                
                if N >= 10:
                    feat_pvals.append(N/dat_bin.size)
                else:
                    
                    plt.figure()
                    plt.plot(np.sort(dat_bin))
                    
                    dat_bin_s = np.argsort(dat_bin)
                    
                    #try to fit to a pareto distribution since we do not have enough values to get a good estimate of the p-value
                    for tt in range(-250, 0, 10):
                        ytail = dat_bin_s[tt:]
                        
                        #% Estimate the distribution parameters. See §3.2 of Hosking &
                        #% Wallis (1987). Compared to the usual GPD parameterisation, 
                        #% here k = shape (xi), and a = scale.
                        x    = np.mean(ytail);
                        s2   = np.var(ytail);
                        apar = x*(x**2/s2 + 1)/2;
                        kpar =   (x**2/s2 - 1)/2;
                        
                        z = gpdpvals(ytail, apar, kpar)
                        A2pval = andersondarling(gpdpvals(ytail, apar, kpar), kpar)
                    
                        if A2pval > .05:
                            print('good :)')
                            break
                        #    cte = numel(Gtail)/nP;
                    #    Ptail = cte*gpdpvals(dat_bin_s, apar, kpar);
                    #%%
            break
    #%%
            
            
            #ECDF empirical cumulative distribution function
            #GPD generalized pareto distribution
            #ECDF = 1 + np.sum(jsd)
    
    #%%
    
    p_th = 1e-4
    for strain, (all_JSD, pvals) in hist_results.items():
        ff = [k for k,val in pvals.items() if np.sum(val<p_th) > 1]
        print(strain, len(ff))   
        
        
        pass