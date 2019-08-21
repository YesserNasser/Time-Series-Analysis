# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:06:10 2019

@author: Yesser H. Nasser
"""

import math
import os
import json
import numpy as np
import datetime as dt
import pandas as pd
import time
import keras
import matplotlib.pyplot as plt


from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import newaxis

# =============================================================================
class Timer():
	def __init__(self):
		self.start_dt = None
	def start(self):
		self.start_dt = dt.datetime.now()
	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))      
 
'''========================================================================='''
'''================ loading data and config files  ========================='''
'''========================================================================='''

configs = json.load(open('config_param.json', 'r'))
df = pd.read_csv(configs["data"]["filename"])
cols = configs["data"]["columns"]
s = configs["data"]["train_test_split"]
seq_len = configs["data"]["sequence_length"]

split = int(len(df)*s)
normalise = True

'''========================================================================='''
'''===================== Data Preprocessing  ==============================='''
'''========================================================================='''

# ========================= Normalize the data ================================
def normalise_windows(window_data, single_window=False):
    '''Normalise window with a base value of zero'''
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T 
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

# ==================Prepare data features X, target y =========================
def get_data(data):
    data_windows = []
    
    for i in range(len(data) - seq_len):
        data_windows.append(data[i:i+seq_len])
    
    data_windows_1 = np.array(data_windows).astype(float)
    data_windows_1 = normalise_windows(data_windows_1, single_window=False) if normalise else data_windows_1
    
    x = data_windows_1[:, :-1]
    y = data_windows_1[:,-1,[0]]
    return x, y

# =============================================================================
def next_window(i, seq_len, normalise):
    window = data_train[i:i+seq_len]
    window = normalise_windows(window, single_window=True)[0] if normalise else window
    x = window[:-1]
    y = window[-1, [0]]
    return x, y
# ======================= Generate train batchs ===============================
def generate_train_batch(data_train, configs):
    len_train = len(data_train)
    seq_len = configs["data"]["sequence_length"]
    batch_size = configs["training"]["batch_size"]
    normalise = configs["data"]["normalise"]
    i = 0
    while i < (len_train - seq_len):
        x_batch = []
        y_batch = []
        for b in range(batch_size):
            if i >= (len_train - seq_len):
                # stop-condition for a smaller final batch if data doesn't divide evenly
                yield np.array(x_batch), np.array(y_batch)
                i = 0
            x, y = next_window(i, seq_len, normalise)
            x_batch.append(x)
            y_batch.append(y)
            i += 1
        yield np.array(x_batch), np.array(y_batch)
        
# ============ split data into Train and test data ============================
data_train = df.get(cols).values[:split]
data_test = df.get(cols).values[split:]
len_train = len(data_train)

x_train, y_train = get_data(data_train)

x_test, y_test = get_data(data_test)
'''========================================================================='''
'''================= Build ing the Model RNN (LSTM) ========================'''
'''========================================================================='''
def build_model(configs):
    model = Sequential()
    for layer in configs["model"]["layers"]:
        neurons = layer["neurons"] if "neurons" in layer else None
        dropout_rate = layer["rate"] if "rate" in layer else None
        activation = layer["activation"] if "activation" in layer else None
        return_seq  = layer["return_seq"] if "return_seq" in layer else None
        input_timesteps = layer["input_timesteps"] if "input_timesteps" in layer else None 
        input_dim = layer["input_dim"] if "input_dim" in layer else None
        
        if layer["type"] == "dense":
            model.add(Dense(neurons, activation = activation))
        if layer["type"] == "lstm":
            model.add(LSTM(neurons, input_shape = (input_timesteps, input_dim), return_sequences = return_seq))
        if layer["type"] == "dropout":
            model.add(Dropout(dropout_rate))  
    model.compile(optimizer = configs["model"]["optimizer"], loss = configs["model"]["loss"])
    
    return model, model.summary()
model, summary = build_model(configs)

'''========================================================================='''
'''======================== Training the model ============================='''
'''========================================================================='''
def train_model_generator(model, data, configs, steps_per_epoch):
    timer = Timer()
    timer.start()
    
    epochs = configs["training"]["epochs"]
    batch_size = configs["training"]["batch_size"]
    save_dir = configs["model"]["save_dir"] 
    
    print('[Model] training started')
    print('[Model] %s epochs, %s batch_size %s steps_per_epoch' %(epochs, batch_size, steps_per_epoch))
    
    
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    
    callbacks = [
            EarlyStopping(monitor = "val_loss" , patience = 2),
            ModelCheckpoint(filepath = save_fname, monitor = 'val_loss', save_best_only = True)
            ]
    
    model.fit_generator(data_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs = epochs,
                        callbacks = callbacks,
                        workers=1)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.save(save_fname)
    print('[Model] Training completed. Model saved as %s' %save_fname)
    timer.stop()
# =============================================================================
data_train_gen= generate_train_batch(data_train, configs)
len_train = len(data_train)
steps_per_epoch = math.ceil((len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])   
train_model_generator(model, data_train_gen, configs, steps_per_epoch)

'''========================================================================='''
'''=========================== Predictions ================================='''
'''========================================================================='''

# ======================= predict point by point ==============================
def predict_point_by_point(model,data):
    print('[Model] Predicting Point-by-Point')
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
# ======================= predict multiple sequences ==========================
def predict_sequences_multiple(model, data, window_size, prediction_len):
    print('[Model] predicting sequences multiple')
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

# ===================== Prediction using test data ============================
predictions_multi_sequences = predict_sequences_multiple(model, x_test, 
                                                               configs['data']['sequence_length'], 
                                                               configs['data']['sequence_length'])   
predictions_point_by_point = predict_point_by_point(model, x_test) 

# ============================= Plot predictions ==============================  
# ========================== plot multi sequences =============================
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    fig = plt.figure(figsize =(12,6))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    plt.show()
    
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    fig = plt.figure(figsize =(12,6))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.05),
               fancybox=True, shadow=True)
    plt.show()

plt.figure()
plot_results_multiple(predictions_multi_sequences, y_test, configs['data']['sequence_length'])

plt.figure()
plot_results(predictions_point_by_point, y_test)   
