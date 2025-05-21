# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:43:20 2025

@author: rosam
"""

import torch
import numpy as np

#function for performance evaluation metrics

def recall(pred,target):
    tp = (pred*target).sum().item()
    fn = (np.maximum(target-pred,np.zeros_like(pred))).sum().item()
    
    recall = tp/(tp+fn)
    return recall

def precision(pred,target):
    tp = (pred*target).sum().item()
    fp = (np.maximum(pred-target,np.zeros_like(pred))).sum().item()
    
    precision = tp/(tp+fp)
    return precision

def accuracy(pred,target):
    tp = (pred*target).sum().item()
    fp = (np.maximum(pred-target,np.zeros_like(pred))).sum().item()
    fn = (np.maximum(target-pred,np.zeros_like(pred))).sum().item()
    tn = len(pred)*19 - (tp + fp + fn)
    
    accuracy = (tp + tn) / (len(pred)*19)
    return accuracy

def F1(pred,target):
    F1 = (2*precision(pred,target)*recall(pred,target))/(precision(pred,target) + recall(pred,target))
    return F1

def compute_metrics(pred,target):
    metrics = dict(F1_score=F1(pred,target),accuracy=accuracy(pred,target),precision=precision(pred,target),recall=recall(pred,target))
    return metrics