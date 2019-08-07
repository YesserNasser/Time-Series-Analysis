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
    
    # balance the data make there are buys as many sells   
    buys=[]
    sells=[]   
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq, target])           
    random.shuffle(buys)
    random.shuffle(sells)
    
    # look equal number of sells and buys
    lower = min(len(buys), len(sells))   
    buys = buys[:lower]
    sells = sells[:lower]   
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    
    # seperate seq-X, target - y
    X = []
    y = []   
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)       
    return np.array(X), y
# =====================================================
def load_data(DIR,TARGET_TO_PREDICT,c):
    data = pd.read_csv(DIR)
    data.index  = data.DATE
    main_data = data.drop('DATE', 1)
    main_data_corr = main_data.corr()
    target_corr = main_data_corr[TARGET_TO_PREDICT]
    target_corr = abs(target_corr).sort_values(ascending=False)
    tickers_corr = list(target_corr.index[target_corr>c])
    all_tickers = list(main_data.columns[0:])
    tickers_to_drop = list(set(all_tickers) - set(tickers_corr))
    for i in range(len(tickers_to_drop)):
        main_data.drop(tickers_to_drop[i], axis=1, inplace=True)
    return main_data

DIR = 'data.csv'
c = 0.9
main_df = load_data(DIR,TARGET_TO_PREDICT,c)    
# ====================================================    

main_df['future'] = main_df[TARGET_TO_PREDICT].shift(-PERIOD_TO_PREDICT)
main_df['target'] = list(map(classify, main_df[TARGET_TO_PREDICT], main_df['future']))

times = sorted(main_df.index.values)
last_20pct = times[-int(0.2*len(times))]

main_df_validation = main_df[(main_df.index >= last_20pct)]
main_df_train = main_df[(main_df.index < last_20pct)]

X_train, y_train =  pre_process(main_df_train)
X_validation, y_validation =  pre_process(main_df_validation)

print(f'train_data: {len(X_train)}, validation_data: {len(X_validation)}')
print(f'buys_training_data: {y_train.count(1)}, sells_training_data: {y_train.count(0)}')
print(f'buys_validation_data: {y_validation.count(1)}, sells_validation_data: {y_validation.count(0)}')

# building the RNN - LSTM with attention
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Input, Conv2D, MaxPool2D
from keras.models import Sequential, Model
from attention import Attention

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape = (X_train.shape[1:]), return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#
#model.add(LSTM(128, activation='tanh', input_shape = (X_train.shape[1:]), return_sequences = True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
#
model.add(LSTM(128, activation='tanh', input_shape = (X_train.shape[1:]), return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Attention())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax'))

model.summary()


batch_size = 20
training_epochs = 20

model.compile(optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-5), loss = keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
Sequential_prediction_with_attention_sp500 = model.fit(X_train, y_train, batch_size = batch_size, epochs = training_epochs, validation_data=(X_validation, y_validation))
# 
model.save('Sequential_prediction_with_attention_sp500.h5py')

accuracy = Sequential_prediction_with_attention_sp500.history['acc']
val_accuracy = Sequential_prediction_with_attention_sp500.history['val_acc']

loss = Sequential_prediction_with_attention_sp500.history['loss']
val_loss = Sequential_prediction_with_attention_sp500.history['val_loss']

Epochs = range(len(accuracy))

plt.figure()
plt.subplot(1,2,1)
plt.plot(Epochs, accuracy, 'go', label = 'training accuracy')
plt.plot(Epochs, val_accuracy, 'g', label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(1)

plt.subplot(1,2,2)
plt.plot(Epochs, loss, 'ro', label = 'training loss')
plt.plot(Epochs, val_loss, 'r', label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(1)