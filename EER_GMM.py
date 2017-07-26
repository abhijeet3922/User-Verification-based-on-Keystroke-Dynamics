# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:05:22 2017

@author: Admin
"""
import numpy as np
np.set_printoptions(suppress = True)

def evaluateEERGMM(user_scores, imposter_scores):
    thresholds = range(20,51)
    array = np.zeros((len(thresholds),3))
    i = 0
    for th in thresholds:
        g_i = 0
        i_g = 0
        for score in user_scores:
            if score < th:
                g_i = g_i + 1
        for score in imposter_scores:    
            if score > th:
                i_g = i_g + 1

        FA = float(i_g) / len(imposter_scores) 
        FR = float(g_i) / len(user_scores)
        array[i, 0] = th
        array[i, 1] = FA
        array[i, 2] = FR
        i = i + 1
    
    for j in range(array.shape[0]):
        if array[j,1] < array[j,2]:
            thresh = (array[j,0] + array[j - 1, 0]) / 2
            break
    g_i = 0
    i_g = 0
    for score in user_scores:
        if score < thresh:
            g_i = g_i + 1
    for score in imposter_scores:    
        if score > thresh:
            i_g = i_g + 1

    FA = float(i_g) / len(imposter_scores) 
    FR = float(g_i) / len(user_scores)
    return (FA + FR) /2
    