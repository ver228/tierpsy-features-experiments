#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:11:16 2018

@author: ajaver
"""
#from classify_CeNDR import divergent_set

import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pylab as plt
import pandas as pd
import os

import sys
sys.path.append('../../helper')
from misc import results_root_dir
results_dir = os.path.join(results_root_dir, 'simple')
    

def plot_confusion_matrix(cm, classes,
                      normalize=True,
                      cmap=plt.cm.OrRd):
    """
    This function prints and plots the confusion matrix.
    """
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)


    if False:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
#%%
if __name__ == '__main__':
    #experimental_dataset = 'CeNDR'
    experimental_dataset = 'SWDB'
    save_name = os.path.join(results_dir, 'model_results_{}.pkl'.format(experimental_dataset))
    with open(save_name, "rb" ) as fid:
        strain_dict, results = pickle.load(fid)
    #%%
    res_db = {}
    for (set_type, i_fold), dat in results:
        if set_type not in res_db:
            res_db[set_type] = []
        res_db[set_type].append(dat)
    
    for set_type, dat in res_db.items():
        res_db[set_type] = list(zip(*dat))
        
    for n, m_type in enumerate(['loss', 'acc', 'f1']):
        print(m_type)
        for set_type, dat in res_db.items():
            vv = dat[n]
            
            dd = '{} : {:.2f}±{:.2f}'.format(set_type, np.mean(dat[n]), np.std(dat[n]))
            print(dd)
    #%%
    for n, m_type in enumerate(['loss', 'acc', 'f1']):
        print(m_type)
        for set_type, dat in res_db.items():
            vv = dat[n]
            
            dd = '{} : {:.2f}±{:.2f}'.format(set_type, np.mean(dat[n]), np.std(dat[n]))
            print(dd)
    
    #%%
    ss_titles = {'all': 'All Features', 'reduced':'Reduced Set of Features'}
    
    
    _, s_sorted = zip(*sorted(strain_dict.items(), key=lambda x : x[0]))
    
    ss = ['all']#['reduced', 'all']
    for ii, set_type in enumerate(ss):
        y_pred = np.concatenate(res_db[set_type][-1])
        y_true = np.concatenate(res_db[set_type][-2])
        #np.concatenate(y_true)
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        #%%
        cm_df = pd.DataFrame(cm, index=s_sorted, columns=s_sorted)
        #%%
        #plt.subplot(1,2,ii+1)
        plt.figure(figsize=(60,50))
        plot_confusion_matrix(cm, s_sorted, 
                              normalize = True, 
                              cmap = plt.cm.magma_r)
        
        
        plt.title(ss_titles[set_type], fontsize=5)
    plt.tight_layout()
    
    fname = os.path.join(results_dir, '{}_confusion_matrix.pdf'.format(experimental_dataset))
    plt.savefig(fname)   