#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:44:03 2018

@author: ajaver
"""
import os
import pandas as pd
import random
import math
import glob
import sys
import multiprocessing as mp
from tierpsy_features.summary_stats import get_n_worms_estimate, get_summary_stats

#def process_feat_tierpsy_file(fname, delta_time = 1/3):
#    #%%
#    fps = read_fps(fname)
#    timeseries_data = _get_timeseries_feats(fname, delta_time)
#    
#    with pd.HDFStore(fname, 'r') as fid:
#        blob_features = fid['/blob_features']
#    exp_feats = get_summary_stats(timeseries_data, fps,  blob_features, delta_time)


def _augment_data(fname, 
                 derivative_delta_time, 
                 n_folds, 
                 frac_worms_to_keep,
                 time_sample_seconds
                 ):
    with pd.HDFStore(fname, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features']
        
        fps = fid.get_node('/trajectories_data')._v_attrs['fps']
    
    #%%
    #fraction of trajectories to keep. I want to be proportional to the number of worms present.
    n_worms_estimate = get_n_worms_estimate(timeseries_data['timestamp'])
    frac_worms_to_keep_r = min(frac_worms_to_keep, (1-1/n_worms_estimate))
    if frac_worms_to_keep_r <= 0:
        #if this fraction is really zero, just keep everything
        frac_worms_to_keep_r = 1
    
    time_sample_frames = time_sample_seconds*fps
    
    ini_ts = timeseries_data['timestamp'].min()
    last_ts = timeseries_data['timestamp'].max()
    
    ini_sample_last = max(ini_ts, last_ts - time_sample_frames)
    
    augmented_data = []
    for i_fold in range(n_folds):
        ini = random.randint(ini_ts, ini_sample_last)
        fin = ini + time_sample_frames
        
        good = (timeseries_data['timestamp'] >= ini) & (timeseries_data['timestamp'] <= fin) 
        
        #select only a small fraction of the total number of trajectories present
        ts_sampled_worm_idxs = timeseries_data.loc[good, 'worm_index']
        available_worm_idxs = ts_sampled_worm_idxs.unique()
        random.shuffle(available_worm_idxs)
        n2select = math.ceil(len(available_worm_idxs)*frac_worms_to_keep_r)
        idx2select = available_worm_idxs[:n2select]
        
        good = good & ts_sampled_worm_idxs.isin(idx2select)
        
        timeseries_data_r = timeseries_data[good].reset_index(drop=True)
        blob_features_r = blob_features[good].reset_index(drop=True)
    
        exp_feats_r = get_summary_stats(timeseries_data_r, 
                                        fps,  
                                        blob_features_r, 
                                        derivative_delta_time,
                                        only_abs_ventral = True)
        
        augmented_data.append(exp_feats_r)
        
    return augmented_data

def _process_row(data_in):
    irow, row = data_in
    fname = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    
    print(irow+1, os.path.basename(fname))
    try:
        feat_stats = _augment_data(fname,
                                   derivative_delta_time, 
                                     n_folds, 
                                     frac_worms_to_keep,
                                     time_sample_seconds)
        
        
        feat_stats = pd.concat(feat_stats, axis=1).T
        feat_stats['experiment_id'] = int(row['id'])
        
        return feat_stats
    except:
        return None
    
if __name__ == '__main__':
    #fname = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_020617/N2_worms5_food1-10_Set1_Pos4_Ch5_02062017_115615_featuresN.hdf5'
    
    #exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    #save_name_f = 'tierspy_features_augmented_CeNDR.csv'
    
    exp_set_dir = '/Volumes/behavgenom_archive$/Adam/screening/Syngenta/'
    save_name_f = 'tierspy_features_augmented_Syngenta.csv'
    
    n_batch = 20
    
    n_folds = 20
    frac_worms_to_keep = 0.9
    time_sample_seconds = 10*60
    derivative_delta_time = 1/3
    
    save_dir = './'
    
    #sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    d_path = os.path.join(os.environ['HOME'], 'Github/process-rig-data/process_files')
    sys.path.append(d_path)
    from misc import get_rig_experiments_df

    
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')

    set_type = 'featuresN'
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    #filter temporary
    csv_files =  [x for x in csv_files if not os.path.basename(x).startswith('~')]
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments_df = get_rig_experiments_df(features_files, csv_files)
    experiments_df = experiments_df.sort_values(by='video_timestamp').reset_index()  
    
    experiments_df['id'] = experiments_df.index
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    
    #debug line...
    #experiments_df = experiments_df.iloc[:n_batch]
    
    p = mp.Pool(n_batch)
    all_stats = list(p.map(_process_row, experiments_df.iterrows()))
    #all_stats = list(map(_tierpsy_process_row, experiments_df.iterrows()))
    #%%
    all_stats_f = [x for x in all_stats if x is not None]
    all_stats_f = pd.concat(all_stats_f, ignore_index=True)
    
    all_stats_f.index = all_stats_f['experiment_id']
    del all_stats_f['experiment_id']
    #%%
    save_name = os.path.join(save_dir, save_name_f)
    
    #select experiments that where processed otherwise the concat will throw an error
    exp_df = experiments_df.loc[all_stats_f.index.unique()]
    dd = pd.concat((exp_df, all_stats_f), axis=1)
    dd.to_csv(save_name, index_label=False)
    
    