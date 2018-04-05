#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:07:50 2018

@author: ajaver
"""

import os
import matplotlib.pylab as plt
import seaborn as sns
import sys
sys.path.append('../../helper')
from misc import results_root_dir
from reader import read_feats

#%%



if __name__ == '__main__':
    is_reduced = True
    
    results_dir = os.path.join(results_root_dir, 'clustograms')
    
    
    
    if is_reduced:
        fname = os.path.join(results_root_dir, 'RFE', 'reduced_feats_SWDB.txt')
        with open(fname, 'r') as fid:
            feat_cols= fid.read().split('\n')    #experimental_dataset = 'CeNDR'
    
    for experimental_dataset in ['SWDB', 'Syngenta', 'CeNDR']:
        feat_data, col2ignore_r = read_feats(experimental_dataset)
        if 'all' in feat_data:
            del feat_data['all']
            del feat_data['OW']
        
        
        feats = feat_data['tierpsy']
        group_s = 'strain_description' if 'strain_description' in feats else 'strain'
        
        rr = [x for x in feat_cols if x in feats] 
        feats = feats[rr + [group_s]]
        df = feats.groupby(group_s).agg('mean')
        
        g = sns.clustermap(df, method = 'ward', figsize=(60, 70), robust=True)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        
        dd = os.path.join(results_dir, 'Clustogram_{}.pdf'.format(experimental_dataset))
        g.savefig(dd)