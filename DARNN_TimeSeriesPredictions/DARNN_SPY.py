# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:33:51 2019
@author: Yesser H. Nasser
"""
import torch
from torch import nn
import typing
import pandas as pd
import numpy as np
import collections
from torch.autograd import Variable

from torch import optim

from sklearn.preprocessing import StandardScaler
from typing import Tuple

from modules import Encoder, Decoder
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def data_processing(data,target_list) -> Tuple[TrainingData,StandardScaler]:
    scale = StandardScaler().fit(raw_data)
    proc_dat = scale.transform(raw_data)
    sep = np.ones(proc_dat.shape[1],dtype=bool)
    dat_cols = list(raw_data.columns)
    for col_name in target_list:
        sep[dat_cols.index(col_name)]=False
    features = proc_dat[:, sep]
    targets = proc_dat[:,~ sep]
    
    return TrainingData(features, targets), scale
# ============================= Build DA-RNN ==================================
def darnn(train_data: TrainingData, 
           n_targets: int, 
           encoder_hidden_size: int, 
           decoder_hidden_size: int,
           T: int, 
           learning_rate=0.002, 
           batch_size=32):
    train_cfg = TrainingConfig(T, int(train_data.features.shape[0] * 0.7), batch_size, nn.MSELoss())
    print(f"Training size: {train_cfg.train_size:d}.")
    enc_kwargs = {"input_size": train_data.features.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,"decoder_hidden_size": decoder_hidden_size, "T": T, "out_features": n_targets}
    decoder = Decoder(**dec_kwargs).to(device)
    encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad],lr=learning_rate)
    decoder_optimizer = optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad],lr=learning_rate)
    da_rnn_net = Darnn_Net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    return train_cfg, da_rnn_net
# ========================== prep_train_data ==================================
def prep_train_data(batch_idx: np.ndarray, 
                    t_cfg: TrainingConfig, 
                    train_data: TrainingData):
    features = np.zeros((len(batch_idx), t_cfg.T -1, train_data.features.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T -1, train_data.targets.shape[1]))
    y_target = train_data.targets[batch_idx + t_cfg.T]
    
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx+t_cfg.T - 1)
        features[b_i, :, :] = train_data.features[b_slc,:]
        y_history[b_i, :] = train_data.targets[b_slc]
        
    return features, y_history, y_target

''' ========================================================================'''
''' ========== For the rest of the code please get in touch ================'''
'''========================================================================='''
''' ========================================================================'''
