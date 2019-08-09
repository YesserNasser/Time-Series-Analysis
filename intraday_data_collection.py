# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:30:27 2019

@author: Yesser
"""

from alpha_vantage.timeseries import TimeSeries
import time
import bs4 as bs
import pickle
import requests
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import os
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
style.use('ggplot')

api_key = 'RK9Z1YLQR1VHFTZ5'
ts = TimeSeries(key=api_key, output_format = 'pandas')
#=================================== get Tickers===============================
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        if (ticker != 'BRK.B'and ticker!= 'BF.B' and ticker!= 'CTVA'):
            tickers.append(ticker)
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers,f)   
    print(tickers)  
    return tickers
tickers = save_sp500_tickers()

# ============================ get stock data from Alpha Vantage========================
def get_data_from_AV(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')       
    for i in range(0,502,2):
        tickers_inter = tickers[i:i+2]
        for ticker in tickers_inter:
            print(ticker)
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                data, meta_data = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')
                data.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))
        time.sleep(30) 
get_data_from_AV()
time.sleep(62) 
#============== get addtional Indexies from Yahoo (test SPY)===================
def get_additional_index_1(symbols):
    for symbol in symbols:
        print(symbol)
        if not os.path.exists('stock_dfs/{}.csv'.format(symbol)):
            data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
            data.to_csv('stock_dfs/{}.csv'.format(symbol))
        else:
            print('Already have {}'.format(symbol))
        if symbol not in tickers:
            tickers.append(symbol)
symbols_1 = ['SPY','IWM','DIA','IEF','TLT',]         
get_additional_index_1(symbols_1)
time.sleep(62) 
# =============================================================================
def get_additional_index_2(symbols):
    for symbol in symbols:
        print(symbol)
        if not os.path.exists('stock_dfs/{}.csv'.format(symbol)):
            data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
            data.to_csv('stock_dfs/{}.csv'.format(symbol))
        else:
            print('Already have {}'.format(symbol))
        if symbol not in tickers:
            tickers.append(symbol)
symbols_2 = ['GLD','SLV','USD',]           
get_additional_index_2(symbols_2)
#=====================compile data in one dataframe============================
def compile_data():
    with open('sp500tickers.pickle','rb') as f:       
        main_df = pd.DataFrame()
        for count,ticker in enumerate (tickers):
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('date', inplace=True)
            
            df.rename(columns={'4. close': ticker}, inplace=True)
            df.drop(['1. open', '2. high', '3. low', '5. volume'], 1, inplace=True)
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
            
            if count % 10 == 0:
                print(count)
        print(main_df.head())
        main_df.to_csv('sp500_joined_intrady_1min_closes_02_08_august2019.csv')
compile_data() 
# ========================= upload data =======================================
data_df = pd.read_csv('sp500_joined_intrady_1min_closes_02_08_august2019.csv') 
data_df.index = data_df.date
data_df = data_df.drop('date', 1)
data_df.isnull().sum()  
for col in data_df.columns:
    data_df[col].fillna(method='ffill', inplace=True)
data_df.isnull().sum()  
corr = data_df.corr()