# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:08:22 2025

@author: rosam
"""

import numpy as np
import matplotlib.pyplot as plt

def pos_weight(df_train):

    n = len(df_train)
    count=np.zeros(19)
    
    for i in range(n):
        labels = list(map(int,df_train.iloc[i,1].split(" ")))
        for j in range(19):
            count[j] = count[j]+labels.count(j+1)
    
    print(count)
    names = np.arange(1,20)
    
    # calculate weights for the loss function
    pos_weight  = (n - count) / count
    print(pos_weight)
    
    plt.bar(names,count)





