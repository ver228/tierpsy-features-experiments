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

def plot_confusion_matrix(cm, classes,
                      normalize=True,
                      cmap=plt.cm.OrRd):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

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
    experimental_dataset = 'CeNDR'
    save_name = 'model_results_{}.pkl'.format(experimental_dataset)
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
    
    plt.figure(figsize=(12,6))
    
    #ss_titles = {'all': 'All Features', 'motion':'Coarse-Grain Features'}
    ss_titles = {'all': 'All Features', 'reduced':'Reduced Set of Features'}
    for ii, set_type in enumerate(['reduced', 'all']):
        y_pred = np.concatenate(res_db[set_type][-1])
        y_true = np.concatenate(res_db[set_type][-2])
        #np.concatenate(y_true)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.subplot(1,2,ii+1)
        plot_confusion_matrix(cm, strain_dict, 
                              normalize = True, 
                              #cmap = plt.cm.inferno_r)
                              #cmap = plt.cm.plasma_r)
                              #cmap = plt.cm.viridis_r)
                              cmap = plt.cm.magma_r)
        
        
        plt.title(ss_titles[set_type], fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf')
        