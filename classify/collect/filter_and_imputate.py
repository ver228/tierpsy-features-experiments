#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:20:36 2017

@author: ajaver
"""
import os
import pandas as pd

import sys
sys.path.append('../helper')
from misc import data_root_dir, col2ignore

def _set_to_file(set_type, feat_type):
    file_basenames = {
    'OW' : 'ow_features_full_{}.csv',
    'tierpsy' :'tierpsy_features_full_{}.csv',
    'tierpsy_augmented' : 'tierspy_features_augmented_{}.csv'
    }
    if not feat_type in file_basenames:
        raise ValueError('Feature type {} not recognized.'.format(feat_type))
        
    
    bn = file_basenames[feat_type].format(set_type)
    fname = os.path.join(data_root_dir, set_type, bn)
    return fname

def _get_args(set_type):
    if set_type == 'CeNDR':
        MIN_N_VIDEOS = 3
        feat_types = ['OW' , 'tierpsy', 'tierpsy_augmented']
    elif set_type == 'MMP':
        MIN_N_VIDEOS = 3
        feat_types = ['tierpsy_augmented']
    elif set_type == 'Syngenta':
        MIN_N_VIDEOS = 3
        feat_types = ['tierpsy_augmented']
    elif set_type == 'SWDB':
        MIN_N_VIDEOS = 10
        feat_types = ['OW' , 'tierpsy']
    elif set_type == 'Agging':
        MIN_N_VIDEOS = 10
        feat_types = ['OW' , 'tierpsy']
    else:
        raise ValueError('Input for "{}" not implemented.'.format(set_type))


    feat_files = {x:_set_to_file(set_type, x) for x in feat_types}
    return MIN_N_VIDEOS, feat_files

#%%
if __name__ == '__main__':
    MAX_FRAC_NAN = 0.025
    #MAX_FRAC_NAN = 0.05
    #experimental_dataset = 'MMP'
    #experimental_dataset = 'Syngenta'
    #experimental_dataset = 'SWDB'
    experimental_dataset = 'Agging'
    #experimental_dataset = 'CeNDR'
    
    MIN_N_VIDEOS, feat_files = _get_args(experimental_dataset)
    
    all_features = {}
    for db_name, feat_file in feat_files.items():
        print(db_name)
        feats = pd.read_csv(feat_file)
        
        if experimental_dataset == 'Syngenta':
            feats['strain'] = feats['base_name'].apply(lambda x : '_'.join(x.split('_')[2:4]))
        
        dd = feats.isnull().mean()
        
        col2remove =  dd.index[(dd>MAX_FRAC_NAN).values].tolist()
        feats = feats[[x for x in feats if x not in col2remove]]
        all_features[db_name] = feats
        
        print(db_name)
        print(col2remove)
       #%%
    if 'OW' in all_features:
        #make sure the same videos are selected in OW and tierpsy
        assert (all_features['OW']['base_name'].values == all_features['tierpsy']['base_name'].values).all()
    
    #%%
    val = next(iter(all_features.values()))
    #deal with augmented datset
    if 'id' in val:
        val = val.drop_duplicates('id')
    dd = val['strain'].value_counts()
    good_strains = dd.index[(dd>=MIN_N_VIDEOS).values].values
    #%%
    for db_name, feats in all_features.items():
        feats = feats[feats['strain'].isin(good_strains)]
        #Imputate missing values. I am using the global median to be more conservative
        all_features[db_name] = feats.fillna(feats.median())
    

    #%%
    #select files that are present in all the features sets
    valid_ind = None
    for db_name, feats in all_features.items():
        if valid_ind is None:
            valid_ind = set(feats['base_name'].values)
        else:
            valid_ind = valid_ind & set(feats['base_name'].values)
    
    for db_name, feats in all_features.items():
        all_features[db_name] = feats[feats['base_name'].isin(valid_ind)]
    
    
    #%%
    for db_name, feats in all_features.items():
        fname = feat_files[db_name]
        
        save_dir, bn = os.path.split(fname)
        
        fname = os.path.join(save_dir, 'F{:.3}_{}'.format(MAX_FRAC_NAN, bn))
        
        #check there is not any nan
        
        feat_cols = [x for x in feats.columns if x not in col2ignore]
        assert not feats[feat_cols].isnull().any().any()
        feats.to_csv(fname)
    
    