#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:13:02 2018

@author: ajaver
"""
from tierpsy.analysis.ske_create.helperIterROI import getWormROI
import pandas as pd
import tables
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches


#%%
if __name__ == '__main__':
    save_dir = './'
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/paper_tierpsy_tracker/different_setups/'
    
    set_type = 'single_worm'
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/Ev_videos/N2_adults/MaskedVideos/N2_A_24C_L_5_2015_06_16__19_54_27__.hdf5'
    tt = 0
    delT = 800
    w_index = 1
    
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5').replace('MaskedVideos', 'Results')
    
    print(set_type)
    with tables.File(feat_file) as fid:
        fps = fid.get_node('/trajectories_data')._v_attrs['fps']
        time_units = fid.get_node('/trajectories_data')._v_attrs['time_units']
    
        microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']
        xy_units = fid.get_node('/trajectories_data')._v_attrs['xy_units']
        
        xlabel_str = 'Seconds' if time_units == 'seconds' else 'Frame Number'
    
    with pd.HDFStore(feat_file, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
    
    t_last = min(tt + delT, timeseries_data['timestamp'].max())
    
    with pd.HDFStore(feat_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #the data in timeseries_data is filter by skeleton
    good = trajectories_data['worm_index_joined'].isin(timeseries_data['worm_index'])
    trajectories_data = trajectories_data[good]
    
    traj_g = trajectories_data.groupby('frame_number')
    frame_data = traj_g.get_group(tt)
    
    ts_g = timeseries_data.groupby('timestamp')
    ts_frame_data = ts_g.get_group(tt)
    
    with tables.File(mask_file, 'r') as fid:
        img = fid.get_node('/mask')[tt]
        
        save_full = fid.get_node('/full_data')._v_attrs['save_interval']
        img_full = fid.get_node('/full_data')[tt//save_full]
    
    
    
    row = frame_data[frame_data['worm_index_joined'] == w_index].iloc[0]
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    #### ROI RAW ####
    worm_full, roi_corner = getWormROI(img_full, row['coord_x'], row['coord_y'], row['roi_size'])
    
    
    axs[0].imshow(worm_full, interpolation='none', cmap='gray')
    axs[0].axis('off')
    
    #### ROI + SKELETON ####
    worm_img, roi_corner = getWormROI(img, row['coord_x'], row['coord_y'], row['roi_size'])
    
    axs[1].imshow(worm_img, interpolation='none', cmap='gray')
    axs[1].axis('off')
    
    
    r_ind = ts_frame_data[ts_frame_data['worm_index'] == w_index].index[0]
    with tables.File(mask_file, 'r') as fid:
        stage_position_pix = fid.get_node('/stage_position_pix')[:]
    
    with tables.File(feat_file, 'r') as fid:
        cc1 = fid.get_node('/coordinates/dorsal_contours')[r_ind]/microns_per_pixel - roi_corner  - stage_position_pix[r_ind]
        cc2 = fid.get_node('/coordinates/ventral_contours')[r_ind]/microns_per_pixel - roi_corner   - stage_position_pix[r_ind]
        ss = fid.get_node('/coordinates/skeletons')[r_ind]/microns_per_pixel - roi_corner - 0.5  - stage_position_pix[r_ind]
        
        
    lw_roi = 1.5
    cc = np.vstack([cc1, cc2[::-1]])
    
    axs[2].plot(ss[:, 0], ss[:, 1], 'r', lw=lw_roi)
    axs[2].fill(cc[:, 0], cc[:, 1], color='salmon', alpha=0.7)
    
    
    axs[2].axis('off')
    plt.savefig(os.path.join(save_dir, set_type + '_skel.pdf'), 
                bbox_inches='tight')

    s_xlim = plt.xlim()
    s_ylim = plt.ylim()
    #%%
    plt.figure(figsize=(6, 1))
    for ii in range(0, 40, 5):
        tt = ii + 320
        with tables.File(feat_file, 'r') as fid:
            cc1 = fid.get_node('/coordinates/dorsal_contours')[tt]/microns_per_pixel - stage_position_pix[tt]
            cc2 = fid.get_node('/coordinates/ventral_contours')[tt]/microns_per_pixel   - stage_position_pix[tt]
            ss = fid.get_node('/coordinates/skeletons')[tt]/microns_per_pixel - 0.5  - stage_position_pix[tt]
            cc = np.vstack([cc1, cc2[::-1]])
            
        delx = (tt-320)*40
        plt.plot(ss[:, 0] + delx, ss[:, 1], 'r', lw=lw_roi)
        plt.fill(cc[:, 0] + delx, cc[:, 1], color='salmon', alpha=0.7)
        
    #plt.ylim((50, 350))
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'sequence_skel.pdf'), 
                bbox_inches='tight')