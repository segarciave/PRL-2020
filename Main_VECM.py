# -*- coding: utf-8 -*-
"""
@author: Sergio GARCIA-VEGA
sergio.garcia-vega@postgrad.manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_VECM.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from os import listdir
import sklearn.metrics as metrics
from statsmodels.tsa.api import VECM
from sklearn.metrics import mean_squared_error



#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd        = os.getcwd()
data_dir_trte = os.path.join(crr_wd,'Data')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df            = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\' + str(trte_stock), 'rb'))
keys_stocks = list(TrainTest.keys())

data = {}
for stock in keys_stocks:
    data[stock] = TrainTest[stock]['Train']['Set']
data = pd.DataFrame(data)
data = np.array(data)



#==============================================================================
#                     Vector Error Correction Model (VECM)
#==============================================================================
data_test = np.zeros([280,10,24])
count     = 0    

for stock in keys_stocks:    
    #======================================================================
    #                            Data Embedding
    #======================================================================
    X_test = TrainTest[stock]['Test']['X_te']
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    for i in range(280):
        data_test[i, :, count] = X_test[i, :, 0]
    count = count + 1


    #======================================================================
    #                            Predictions
    #======================================================================
pred = np.zeros([280, 1, 24]) 
for i in range(280):
    vecm          = VECM(endog = np.concatenate((data, data_test[i, :, :])), k_ar_diff = 3, coint_rank = 0, deterministic = 'ci')
    res           = vecm.fit()
    pred[i, :, :] = res.predict(steps=1)


    #==========================================================================
    #                               Saving Results
    #==========================================================================
count = 0  
for stock in keys_stocks:
    y_test   = TrainTest[stock]['Test']['y_te']
    y_pred   = pred[:, 0, count][:, np.newaxis]
    mse      = mean_squared_error(y_test, y_pred)    
    mae_test = metrics.mean_absolute_error(y_test, y_pred)
    
    Results_VECM                             = {}
    Results_VECM['Regression']               = {}
    Results_VECM['Regression']['Desired']    = y_test
    Results_VECM['Regression']['Prediction'] = y_pred
    Results_VECM['Measures']                 = {}
    Results_VECM['Measures']['MSE']          = mse
    Results_VECM['Measures']['MAE']          = mae_test
    
    
    count = count + 1
    
    pickle.dump(Results_VECM, open('Results\\VECM\\Results_VECM_' + stock + '.pkl', 'wb'))


