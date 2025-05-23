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
    if tp == 0:
        return 0
    else:
        rec = tp/(tp+fn)
    return rec

def precision(pred,target):
    tp = (pred*target).sum().item()
    fp = (np.maximum(pred-target,np.zeros_like(pred))).sum().item()
    if tp == 0:
        return 0
    else:
        prec = tp/(tp+fp)
    return prec

def accuracy(pred,target):
    tp = (pred*target).sum().item()
    fp = (np.maximum(pred-target,np.zeros_like(pred))).sum().item()
    fn = (np.maximum(target-pred,np.zeros_like(pred))).sum().item()
    size = 1
    for dim in np.shape(pred.squeeze()): size *= dim
    tn = size - (tp + fp + fn)
    if tp + tn == 0:
        return 0
    else:
        acc = (tp + tn) / (size)
    return acc

def F1(pred,target):
    prec = precision(pred,target)
    rec = recall(pred,target)
    if prec + rec == 0:
        return 0
    else:
        F1 = (2*prec*rec)/(prec+rec)
    return F1

def compute_metrics(pred,target):
    metrics = dict(F1_score=F1(pred,target),accuracy=accuracy(pred,target),precision=precision(pred,target),recall=recall(pred,target))
    return metrics