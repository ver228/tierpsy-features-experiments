#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:57:07 2017

@author: ajaver
"""

import os
import glob
import pandas as pd
import numpy as np
import tables
import numba

import matplotlib.pylab as plt
from tierpsy.helper.params import read_fps

n_pulses = 5
regions = ['before', 'after', 'long_pulse']
pulses_regions = ['short_pulse_{}'.format(ii+1) for ii in range(n_pulses)]
inter_pulses_regions = ['inter_pulses_{}'.format(ii+1) for ii in range(n_pulses)]
regions += pulses_regions + inter_pulses_regions

REGION_LABELS = {x:ii for ii,x in enumerate(regions)}
#%%
#make the inverse diccionary to get the name from the index
REGION_LABELS_I = {val:key for key, val in REGION_LABELS.items()}

@numba.jit
def fillfnan(arr):
    '''
    fill foward nan values (iterate using the last valid nan)
    I define this function so I do not have to call pandas DataFrame
    '''
    out = arr.copy()
    for idx in range(1, out.shape[0]):
        if np.isnan(out[idx]):
            out[idx] = out[idx - 1]
    return out

def read_light_data(mask_file, trajectories_data= None, n_sigmas=6):
    
    with tables.File(mask_file) as fid:
        mean_intensity = fid.get_node('/mean_intensity')[:]
        tot_imgs = fid.get_node('/mask').shape[0]
        
    #pad or eliminate elements if mean_intensity does not match the number of images
    if tot_imgs < mean_intensity.size:
        mean_intensity = mean_intensity[:tot_imgs]
    elif tot_imgs > mean_intensity.size:
        mean_intensity = np.pad(mean_intensity, (0, tot_imgs-mean_intensity.size), 'edge')
    assert mean_intensity.size == tot_imgs
    
    med = np.median(mean_intensity)
    mad = np.median(np.abs(mean_intensity-med))
    #the MAD is releated to sigma by the factor below as explained here:
    #wiki (https://en.wikipedia.org/wiki/Median_absolute_deviation#relation_to_standard_deviation)
    s = mad*1.4826 
    
    #... and since the median should be equal to the mean in a gaussian dist
    # we can use 6 sigma as our threshold
    light_on = mean_intensity >  med + s*n_sigmas
    
    if trajectories_data is not None:
        #fix the vector to match timestamp instead of frame number
        equiv_ts = trajectories_data[['frame_number', 'timestamp_raw']].drop_duplicates().reset_index(drop=True)
    
        max_frame_number, max_timestamp = equiv_ts.max().values
        
        light_on_c = np.full(max_timestamp + 1, np.nan)
        light_on_c[equiv_ts['timestamp_raw'].values] = light_on[equiv_ts['frame_number'].values]
        
        if max_timestamp > max_frame_number:
            #there are drop frames I should fill this
            #I am forcing the first frame to be 0
            light_on_c[0] = 0
            light_on_c = fillfnan(light_on_c)
        light_on = light_on_c.astype(light_on.dtype)
    

    return light_on

def get_pulses_indexes(light_on, window_size):
    switches = np.diff(light_on.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)
    
    assert turn_on.size == turn_off.size
    
    delP = turn_off - turn_on
    
    good = delP>window_size/2
    return turn_on[good], turn_off[good]



def define_regions(tot_frames, turn_on, turn_off, frames_w = 15):
    regions_lab = np.zeros(tot_frames, np.int)
    
    regions_lab[:turn_on[0]-frames_w] = REGION_LABELS['before']
    regions_lab[turn_off[-1] + frames_w :] = REGION_LABELS['after']
    
    regions_lab[turn_on[-1] + 1 :turn_off[-1]] = REGION_LABELS['long_pulse']
    
    for ii in range(n_pulses):
        ini = turn_on[ii]
        fin = turn_off[ii]
        regions_lab[ini + 1 : fin] = REGION_LABELS['short_pulse_{}'.format(ii+1)]
    
    for ii in range(n_pulses):
        ini = turn_off[ii]
        fin = turn_on[ii+1]
        regions_lab[ini + 1 : fin] = REGION_LABELS['inter_pulses_{}'.format(ii+1)]
    
        
    return regions_lab


def read_file_data(mask_file, feat_file, min_pulse_size_s=3, _is_debug=False):
    
    fps = read_fps(mask_file)
    min_pulse_size = fps*min_pulse_size_s
    
    #read features
    with pd.HDFStore(feat_file, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features']
        trajectories_data = fid['/trajectories_data']
    
    
    light_on = read_light_data(mask_file, trajectories_data)
    if np.nansum(light_on) < min_pulse_size:
        return
    
    turn_on, turn_off = get_pulses_indexes(light_on, min_pulse_size)
    region_lab = define_regions(light_on.size, turn_on, turn_off)
    region_size = np.bincount(region_lab)[1:]/fps
    
    if _is_debug:
        plt.figure()
        #plt.plot(region_lab)
        plt.plot(light_on)
        plt.plot(turn_on, light_on[turn_on], 'o')
        plt.plot(turn_off, light_on[turn_off], 'x')
        plt.title(os.path.basename(mask_file))
    
        
    timeseries_data['timestamp'] = timeseries_data['timestamp'].astype(np.int)
    #label if frame with the corresponding region
    timeseries_data['region_lab'] = region_lab[timeseries_data['timestamp']]
    
    
    with tables.File(mask_file, 'r') as fid:
        tot_images = fid.get_node('/mask').shape[0]
    
    
    return timeseries_data, blob_features, fps, region_size, tot_images, len(light_on)


#%%
def get_exp_data(mask_dir):
    
    fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)   
    col_names = ['day', 'strain', 'exp_type', 'mask_file', 'has_valid_light', 'video_duration']
    col_names +=  [REGION_LABELS_I[x] for x in sorted(REGION_LABELS_I.keys())]
    extra_cols_v =  [False] +  [np.nan]* (len(REGION_LABELS_I) + 1)
    
    data = []
    for fname in fnames:
        day_n = fname.split(os.sep)[-2].rpartition('-')[-1]
        base_name = os.path.basename(fname).replace('.hdf5', '')
        
        strain_n, _, dd = base_name.partition('-') 
        exp_type = dd.partition('_') [0]
        
        data.append([day_n, strain_n, exp_type, fname] + extra_cols_v)
    
    
    
    df = pd.DataFrame(data, columns=col_names)
    
    return df
   
if __name__ == '__main__':
    _is_debug = True
    
    mask_dir = '/Volumes/behavgenom_archive$/Lidia/MaskedVideos'
    exp_df = get_exp_data(mask_dir)
    
    #correct some issues
    
    wrongly_named_stains = {'HRB222':'HBR222'}
    bad_strains = ['AZ46', 'AZ60']
    
    exp_df = exp_df.replace({'strain':wrongly_named_stains})
    exp_df = exp_df[~exp_df['strain'].isin(bad_strains)]
    exp_df.index = np.arange(len(exp_df))
    
    exp_df['tot_images'] = np.nan
    exp_df['tot_timestamps'] = np.nan
    
    for irow, row in exp_df.iterrows():
        print(irow+1, len(exp_df))
        mask_file = row['mask_file']
        feat_file = mask_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_featuresN.hdf5')
        
        output = read_file_data(mask_file, feat_file, _is_debug = _is_debug)
        if output is None:
            continue
        else:
            timeseries_data, blob_features, fps, region_size, tot_images, tot_timestamps = output
        
        exp_df.loc[irow, 'has_valid_light'] = True
        fps = read_fps(mask_file)
        exp_df.loc[irow, 'video_duration'] = timeseries_data['timestamp'].max()/fps
        #add duration of each region
        for ii, val in enumerate(region_size):
            exp_df.loc[irow, REGION_LABELS_I[ii+1]] = val
        
        exp_df.loc[irow, 'tot_images'] = tot_images
        exp_df.loc[irow, 'tot_timestamps'] = tot_timestamps
        
        r_stats_l = []
        for r_lab, r_dat in timeseries_data.groupby('region_lab'):
            if r_lab not in REGION_LABELS_I:
                #likely 0 value corresponding a frames between regions
                continue
            
            lab = REGION_LABELS_I[r_lab]
            r_blob = blob_features.loc[r_dat.index]
    #%%
    #save results
    exp_df.to_csv('exp_info.csv')
    