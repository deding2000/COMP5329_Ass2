# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:43:20 2025

@author: rosam
"""

import torch

#function for performance evaluation metrics

def recall(pred,target):
    tp = (pred*target).sum().item()
    fn = (torch.maximum(target-pred,torch.zeros_like(pred))).sum().item()
    
    recall = tp/(tp+fn)
    return recall

def precision(pred,target):
    tp = (pred*target).sum().item()
    fp = (torch.maximum(pred-target,torch.zeros_like(pred))).sum().item()
    
    precision = tp/(tp+fp)
    return precision

def accuracy(pred,target):
    tp = (pred*target).sum().item()
    fp = (torch.maximum(pred-target,torch.zeros_like(pred))).sum().item()
    fn = (torch.maximum(target-pred,torch.zeros_like(pred))).sum().item()
    tn = len(pred) - (tp + fp + fn)
    
    accuracy = (tp + tn) / len(pred)
    return accuracy

def F1(pred,target):
    F1 = (2*precision(pred,target)*recall(pred,target))/(precision(pred,target) + recall(pred,target))
    return F1

