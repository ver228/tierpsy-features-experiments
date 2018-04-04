#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:20:28 2018S

@author: ajaver
"""

import numpy as np

from torch.autograd import Variable
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score

import tqdm

class SimpleNet(nn.Module):
    '''
    Modified from: 
    https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/2_logistic_regression.py
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class TrainerSimpleNet():
    def __init__(self, n_classes, n_features, 
                 n_epochs = 250, 
                 batch_size = 250, 
                 cuda_id = None,
                 lr = 0.001,
                 momentum = 0.9,
                 only_metrics = True):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        
        self.model = SimpleNet(n_features, n_classes)

        if self.cuda_id is not None:
            self.model = self.model.cuda(self.cuda_id)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum)
        self.criterion = F.nll_loss
        
    def fit(self, input_train, target_train):
        dataset = TensorDataset(input_train, target_train)
        loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        
        pbar = tqdm.trange(self.n_epochs)
        for i in pbar:
            #Train model
            self.model.train()
            train_loss = 0.
            for k, (xx, yy) in enumerate(loader):
                train_loss += self._train_step(xx, yy)
            train_loss /= len(loader)
            
            d_str = "train loss = %f" % (train_loss)
            pbar.set_description(d_str)
    
    def _train_step(self, input_v, target_v):

        if self.cuda_id is not None:
            target_v = target_v.cuda(self.cuda_id)
            input_v = input_v.cuda(self.cuda_id)
            
        input_v = Variable(input_v, requires_grad=False)
        target_v = Variable(target_v, requires_grad=False)
    
        # Reset gradient
        self.optimizer.zero_grad()
    
        # Forward
        output = self.model(input_v)
        loss = self.criterion(output, target_v)
    
        # Backward
        loss.backward()
    
        # Update parameters
        self.optimizer.step()
    
        return loss.data[0]
    
    def evaluate(self, input_v, target_v):
        return self._evaluate(self.model, input_v, target_v)
        
    def _evaluate(self, model, input_v, target_v):
        model.eval()
        output = model(input_v)
        
        loss = self.criterion(output, target_v).data[0]
        
        _, y_pred = output.max(dim=1)
        acc = (y_pred == target_v).float().mean().data[0]*100
        
        y_test_l, y_pred_l = target_v.cpu().data.numpy(), y_pred.cpu().data.numpy()
        f1 = f1_score(y_test_l, y_pred_l, average='weighted')
        
        return loss, acc, f1, y_test_l, y_pred_l

    def get_feat_importance(self, input_v, target_v):
        n_features = self.model.fc.in_features
        n_classes = self.model.fc.out_features
        
        model_reduced = SimpleNet(n_features-1, n_classes)
        model_reduced.eval()
        
        inds = list(range(n_features))
        res_selection = []
        for ii in range(n_features):
            ind_r = inds[:ii] + inds[ii+1:]
            model_reduced.fc.weight.data = self.model.fc.weight[:, ind_r].data
            input_r = input_v[:, ind_r]
            
            res = self._evaluate(model_reduced, input_r, target_v)
            loss, acc, f1, y_test_l, y_pred_l = res
            
            res_selection.append((loss, acc, f1))
        
        loss, acc, f1 = map(np.array, zip(*res_selection))
    
    
        return dict(loss = loss, acc = acc, f1 = f1)

