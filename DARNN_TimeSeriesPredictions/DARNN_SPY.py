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
# =============================================================================
class TrainingData(typing.NamedTuple):
    features: np.ndarray
    targets: np.ndarray
# =============================================================================    
class TrainingConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable
# =============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# =============================================================================
Darnn_Net = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])
# =============================================================================
def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))

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
# =========================== Train_iteration =================================
def train_iteration(t_net: Darnn_Net, 
                    loss_func: typing.Callable, 
                    X, 
                    y_history, 
                    y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()
    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))
    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()
    t_net.enc_opt.step()
    t_net.dec_opt.step()
    return loss.item()
# =========================== adjust learning =================================
def adjust_learning_rate(t_net:Darnn_Net, n_iter: int):
    if n_iter % 200 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(t_net.enc_opt.param_groups,t_net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr']*0.9
            dec_params['lr'] = dec_params['lr']*0.9     
# =========================== Predict Function ================================
def predict(t_net: Darnn_Net, t_dat: TrainingData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targets.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.features.shape[0] - train_size, out_size))
    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.features.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targets.shape[1]))
        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)
            X[b_i, :, :] = t_dat.features[idx, :]
            y_history[b_i, :] = t_dat.targets[idx]
        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()
    return y_pred
# ===================== Building Training Function ============================
def train(Network: Darnn_Net, train_data: TrainingData, t_cfg: TrainingConfig, n_epochs=10):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    print(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")
    n_iter = 0
    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            features, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)
            loss = train_iteration(Network, t_cfg.loss_func, features, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            n_iter += 1
            adjust_learning_rate(Network, n_iter)
        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        y_test_pred = predict(Network, train_data,t_cfg.train_size, t_cfg.batch_size, t_cfg.T, on_train=False)
        val_loss = y_test_pred - train_data.targets[t_cfg.train_size:]
        print(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
        y_train_pred = predict(Network, train_data, t_cfg.train_size, t_cfg.batch_size, t_cfg.T, on_train=True)
        if e_i % 1 == 0:      
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targets)), train_data.targets, label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred, label= f'Predicted - Train_epoch_{e_i}')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targets) + 1), y_test_pred, label= f'Predicted - Test_epoch_{e_i}')
            plt.legend(loc='upper left')
            plt.grid()
    return iter_losses, epoch_losses

# =============================================================================
raw_data = pd.read_csv('df_data_Btwn_corr_SPY.csv')
target_list = ('SPY',)   
    
data, scale =  data_processing(raw_data,target_list)   

darnn_args = {'batch_size':32,'T':6, 'encoder_hidden_size': 64, 'decoder_hidden_size': 64} 
lr = 0.002   
config, model = darnn(data, n_targets=len(target_list), learning_rate=lr, **darnn_args)      
iter_loss, epoch_loss = train(model, data, config, n_epochs = 71)
final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)
# =============================================================================

#
plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss, label='iter_loss')
plt.legend(loc='upper right')
plt.grid()

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss, label='epoch loss')
plt.legend(loc='upper right')
plt.grid()

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targets[config.train_size:], label='True')
plt.legend(loc='upper left')
plt.grid()