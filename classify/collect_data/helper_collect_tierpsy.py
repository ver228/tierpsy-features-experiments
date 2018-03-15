#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:04:47 2018

@author: ajaver
"""

import numpy as np
import pandas as pd
import tables

from tierpsy_features import get_timeseries_features
from tierpsy_features.summary_stats import get_n_worms_estimate, get_summary_stats

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name
from tierpsy.helper.params import read_fps, read_ventral_side


#%%
def _get_timeseries_feats(features_file, delta_time = 1/3):
    '''
    Get the all the time series features from the skeletons
    '''
    timeseries_features = []
    fps = read_fps(features_file)
    
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #only use data that was skeletonized
    #trajectories_data = trajectories_data[trajectories_data['skeleton_id']>=0]
    
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(features_file)
    tot_worms = len(trajectories_data_g)
    
    def _display_progress(n):
            # display progress
        dd = " Calculating tierpsy features. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())
    
    _display_progress(0)
    
    with tables.File(features_file, 'r') as fid:
        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]
        else:
            food_cnt = None
    
        #If i find the ventral side in the multiworm case this has to change
        ventral_side = read_ventral_side(features_file)
            
        timeseries_features = []
        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            with tables.File(features_file, 'r') as fid:
                skel_id = worm_data['skeleton_id'].values
                
                #deal with any nan in the skeletons
                good_id = skel_id>=0
                skel_id_val = skel_id[good_id]
                traj_size = skel_id.size

                args = []
                for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):
                    node = fid.get_node('/coordinates/' + p)
                    
                    dat = np.full((traj_size, *node.shape[1:]), np.nan)
                    if skel_id_val.size > 0:
                        if len(node.shape) == 3:
                            dd = node[skel_id_val, :, :]
                        else:
                            dd = node[skel_id_val, :]
                        dat[good_id] = dd
                    
                    args.append(dat)

                timestamp = worm_data['timestamp_raw'].values.astype(np.int32)
            
            feats = get_timeseries_features(*args, 
                                           timestamp = timestamp,
                                           food_cnt = food_cnt,
                                           fps = fps,
                                           ventral_side = ventral_side
                                           )
            #save timeseries features data
            feats = feats.astype(np.float32)
            feats['worm_index'] = worm_index
            #move the last fields to the first columns
            cols = feats.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            feats = feats[cols]
            
            feats['worm_index'] = feats['worm_index'].astype(np.int32)
            feats['timestamp'] = feats['timestamp'].astype(np.int32)
            
            timeseries_features.append(feats)
            _display_progress(ind_n)
        
        timeseries_features = pd.concat(timeseries_features, ignore_index=True)
    
    return timeseries_features



def process_feat_tierpsy_file(fname, delta_time = 1/3):
    #%%
    fps = read_fps(fname)
    timeseries_data = _get_timeseries_feats(fname, delta_time)
    
    with pd.HDFStore(fname, 'r') as fid:
        blob_features = fid['/blob_features']
    exp_feats = get_summary_stats(timeseries_data, fps,  blob_features, delta_time)
    #%%
    return exp_feats



if __name__ == '__main__':
    #fname =  '/Volumes/behavgenom_archive$/single_worm/finished/WT/AQ2947/food_OP50/XX/30m_wait/anticlockwise/483 AQ2947 on food R_2012_03_08__15_42_48___1___8_featuresN.hdf5'
    #fname = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/del-1(ok150)X@NC279/food_OP50/XX/30m_wait/clockwise/del-1 (ok150)X on food L_2012_03_08__15_16_22___1___7_featuresN.hdf5'
    fname = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/gpa-6(pk480)X@NL1146/food_OP50/XX/30m_wait/clockwise/gpa-6 (ph480)X on food L_2009_07_16__12_40__3_featuresN.hdf5'
    #fname = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_020617/N2_worms5_food1-10_Set1_Pos4_Ch5_02062017_115615_featuresN.hdf5'
    
    
    delta_time = 1/3
    exp_feats = process_feat_tierpsy_file(fname, delta_time)
    
    dd = sorted([x for x in exp_feats.index if ('angular' in x) and ('relative' in x)])
    print(dd)
    
    #make sure all the features are unique
    assert np.unique(exp_feats.index).size == exp_feats.size
    #%%
    #for x in exp_feats.index:
    #    print(x)