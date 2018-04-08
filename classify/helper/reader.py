#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:20 2018

@author: ajaver
"""
import os
import pandas as pd
import numpy as np

from misc import data_root_dir, col2ignore

def read_feats(experimental_dataset = 'SWDB', z_transform = True):
    save_dir = os.path.join(data_root_dir, experimental_dataset)
    if experimental_dataset == 'SWDB':
        feat_files = {
                'tierpsy' : 'F0.025_tierpsy_features_full_{}.csv'.format(experimental_dataset),
                'OW' : 'F0.025_ow_features_full_{}.csv'.format(experimental_dataset),
                }
    else:
        feat_files = {
                'tierpsy' : 'F0.025_tierspy_features_augmented_{}.csv'.format(experimental_dataset)
                }
    
    feat_data = {}
    for db_name, bn in feat_files.items():
        fname = os.path.join(save_dir, bn)
        feats = pd.read_csv(fname)
        
        ss = np.sort(feats['strain'].unique())
        s_dict = {s:ii for ii,s in enumerate(ss)}
        feats['strain_id'] = feats['strain'].map(s_dict)
        
        
            
        #maybe i should divided it in train and test, but cross validation should be enough...
        feats['set_type'] = ''
        feat_data[db_name] = feats
    
    if experimental_dataset == 'SWDB':
        #dirty fix for incorrectly named strains
        
        for db_name, df in feat_data.items():
            b1 = df.index[~df['base_name'].str.contains('N2 ') & (df['strain'] == 'N2')]
            
            #i didn't want to include males at this point neither...
            b2 = df.index[df['sex'] == 'male']
            
            #those are really to big to be worms and seems to be an error in the DB
            b3 = df.index[df['id'].isin([87, 14511, 14513, 14514, 14515, 14516, 
                          14517, 14518, 14519, 14520, 14521, 14524, 14525, 14526, 
                          14527, 14528, 14529, 14531, 14532, 14534, 14536, 14537, 
                          14679, 14680, 14681, 14682, 14683, 14684, 14686, 14687, 
                          14688, 14690, 14691, 14692, 14693, 14694, 14697])]
            
            bad_index = np.concatenate((b1, b2, b3))
            
            feat_data[db_name] = df.drop(bad_index)
    
    col2ignore_r = col2ignore + ['strain_id', 'set_type']
    
    if 'OW' in feat_data:
        # create a dataset with all the features
        feats = feat_data['OW']
        col_feats = [x for x in feats.columns if x not in col2ignore_r]
        feats = feats[col_feats + ['base_name']]
        feats.columns = [x if x == 'base_name' else 'ow_' + x for x in feats.columns]
        feat_data['all'] = feat_data['tierpsy'].merge(feats, on='base_name')
    
    if z_transform:
        # scale data (z-transform)
        for db_name, feats in feat_data.items(): 
            col_val = [x for x in feats.columns if x not in col2ignore_r]
            dd = feats[col_val]
            z = (dd-dd.mean())/(dd.std())
            feats[col_val] = z
    
            #drop in case there is any nan
            feats = feats.dropna(axis=1)
    
            feat_data[db_name] = feats
    
    return feat_data, col2ignore_r

