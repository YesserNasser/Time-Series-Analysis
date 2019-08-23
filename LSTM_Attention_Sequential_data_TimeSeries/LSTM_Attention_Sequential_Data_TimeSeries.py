# -*- coding: utf-8 -*-
"""
Created on Tue Jul  16 12:44:50 2019
@author: Yesser H. Nasser
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import matplotlib.pyplot as plt

SEQ_LENGTH = 20
TARGET_TO_PREDICT = 'Ticker'
PERIOD_TO_PREDICT = 3

def classify(current,future):
    if float(future)>float(current):
        return 1
    else:
        return 0

# pre process data scalling /balancing / shuffling data
def pre_process(df):
    df = df.drop('future',1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_period = deque(maxlen=SEQ_LENGTH)
    
    for i in df.values:
        prev_period.append([n for n in i[:-1]])
        if len(prev_period) == SEQ_LENGTH:
            sequential_data.append([np.array(prev_period), i[-1]])
    
    random.shuffle(sequential_data)  
    
    
''' ========================================================================'''
''' ======== To learn more about the code please get in touch =============='''
'''====================== yesser.nasser@icloud.com ========================='''
''' ========================================================================'''
