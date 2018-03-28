#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:20:36 2017

@author: ajaver
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

#col2ignore = ['Unnamed: 0', 'id', 'directory', 'base_name', 'exp_name']
col2ignore = ['Unnamed: 0', 'exp_name', 'id', 'base_name', 'date', 
              'original_video', 'directory', 'strain',
       'strain_description', 'allele', 'gene', 'chromosome',
       'tracker', 'sex', 'developmental_stage', 'ventral_side', 'food',
       'habituation', 'experimenter', 'arena', 'exit_flag', 'experiment_id',
       'n_valid_frames', 'n_missing_frames', 'n_segmented_skeletons',
       'n_filtered_skeletons', 'n_valid_skeletons', 'n_timestamps',
       'first_skel_frame', 'last_skel_frame', 'fps', 'total_time',
       'microns_per_pixel', 'mask_file_sizeMB', 'skel_file', 'frac_valid',
       'worm_index', 'n_frames', 'n_valid_skel', 'first_frame']


#MAIN_DIR = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/'
MAIN_DIR = '../data/'#'/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/'

def _get_args(set_type):
    if set_type == 'CeNDR':
        MIN_N_VIDEOS = 3
        save_dir = os.path.join(MAIN_DIR, 'CeNDR/')
        feat_files = {
                'OW' : 'ow_features_full_CeNDR.csv',
                'tierpsy' :'tierpsy_features_full_CeNDR.csv',
                'tierpsy_augmented' : 'tierspy_features_augmented_CeNDR.csv'
                }
    elif set_type == 'MMP':
        MIN_N_VIDEOS = 3
        save_dir = os.path.join(MAIN_DIR, 'MMP/')
        feat_files = {
                'tierpsy_augmented' : 'tierspy_features_augmented_MMP.csv'
                }
    elif set_type == 'SWDB':
        MIN_N_VIDEOS = 10
        save_dir = os.path.join(MAIN_DIR, 'SWDB/')
        feat_files = {
                'OW' : 'ow_features_full_SWDB.csv',
                'tierpsy' : 'tierpsy_features_full_SWDB.csv'
                }
    elif set_type == 'Agging':
        MIN_N_VIDEOS = 10
        save_dir = os.path.join(MAIN_DIR, 'SWDB/')
        feat_files = {
                'OW' : 'ow_features_full_SWDB.csv',
                'tierpsy' : 'tierpsy_features_full_SWDB.csv'
                }    
    elif set_type == 'Syngenta':
        MIN_N_VIDEOS = 3
        save_dir = os.path.join(MAIN_DIR, 'Syngenta/')
        feat_files = {
                'tierpsy_augmented' : 'tierspy_features_augmented_Syngenta.csv'
                }
    return MIN_N_VIDEOS, save_dir, feat_files

#%%
if __name__ == '__main__':
    MAX_FRAC_NAN = 0.025
    #MAX_FRAC_NAN = 0.05
    #experimental_dataset = 'MMP'
    #experimental_dataset = 'Syngenta'
    #experimental_dataset = 'SWDB'
    experimental_dataset = 'Agging'
    #experimental_dataset = 'CeNDR'
    
    MIN_N_VIDEOS, save_dir, feat_files = _get_args(experimental_dataset)
    #MIN_N_VIDEOS, save_dir, feat_files = _get_args('SWDB')
    #MIN_N_VIDEOS, save_dir, feat_files = _get_args('CeNDR')
    
    all_features = {}
    for db_name, feat_file in feat_files.items():
        print(db_name)
        feats = pd.read_csv(save_dir + feat_file)
        
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
        bn = feat_files[db_name]
        fname = os.path.join(save_dir, 'F{:.3}_{}'.format(MAX_FRAC_NAN, bn))
        
        #check there is not any nan
        
        feat_cols = [x for x in feats.columns if x not in col2ignore]
        assert not feats[feat_cols].isnull().any().any()
        feats.to_csv(fname)
    
    