# -*- coding: utf-8 -*-
"""
@author: Sergio GARCIA-VEGA
sergio.garcia-vega@postgrad.manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_LSTM.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from os import listdir
from keras.layers import LSTM
from keras.layers import Dense
import sklearn.metrics as metrics
from keras.models import Sequential
from sklearn.metrics import mean_squared_error



tic = time.time()

# fix random seed for reproducibility
np.random.seed(7)

#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd         = os.getcwd()
data_dir_trte  = os.path.join(crr_wd,'Data')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\' + str(trte_stock), 'rb'))
    
keys_stocks = list(TrainTest.keys())


count = 1

for stock in keys_stocks:
    
    
    
    #==========================================================================
    #                          Filter Configuration
    #==========================================================================
    N_tr       = TrainTest[stock]['Parameters']['NTrS']
    N_te       = TrainTest[stock]['Parameters']['NTeS']
    time_delay = 10
    
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[stock]['Train']['X_tr'] 
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    y_train = TrainTest[stock]['Train']['y_tr']
    
    X_test = TrainTest[stock]['Test']['X_te']
    X_test  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test = TrainTest[stock]['Test']['y_te']
    
    
    
    #==========================================================================
    #                                 LSTM
    #==========================================================================
    batch_size = 4000
    n          = 20
    e          = 70
    
    model = Sequential()
    model.add(LSTM(n, input_shape=(1, time_delay)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=e, batch_size=batch_size, verbose=0)
    y_pred           = model.predict(X_test, batch_size=batch_size) 
    mse              = mean_squared_error(y_test, y_pred)        
    mae_test         = metrics.mean_absolute_error(y_test, y_pred)    
    
    
    print('Stock: ', count,'/24', ' -- MSE:', mse)
    
    count    = count + 1
    
    #==========================================================================
    #                               Saving Results
    #==========================================================================
    Results_LSTM                                  = {}
    Results_LSTM['Regression']                    = {}
    Results_LSTM['Regression']['Desired']         = y_test 
    Results_LSTM['Regression']['Prediction']      = y_pred
    
    Results_LSTM['Measures']                      = {}
    Results_LSTM['Measures']['MSE']               = mse
    Results_LSTM['Measures']['MAE']               = mae_test
    
    Results_LSTM['Parameters']                    = {}
    Results_LSTM['Parameters']['Number Training'] = N_tr
    Results_LSTM['Parameters']['Number Testing']  = N_te
    Results_LSTM['Parameters']['Time Delay']      = time_delay
    Results_LSTM['Parameters']['Batch Size']      = batch_size 
    Results_LSTM['Parameters']['Neurons']         = n
    Results_LSTM['Parameters']['Epochs']          = e
    
    pickle.dump(Results_LSTM, open('Results\\LSTM\\Results_LSTM_' + stock + '.pkl', 'wb'))
    

toc = time.time()
print(toc-tic, 'sec Elapsed')


