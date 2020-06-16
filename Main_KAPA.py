# -*- coding: utf-8 -*-
"""
@author: Sergio GARCIA-VEGA
sergio.garcia-vega@postgrad.manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_KAPA.py
"""

import os
import pickle
import numpy as np
from os import listdir
import sklearn.metrics as metrics
import Functions.Aux_funcs as aux



#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd = os.getcwd()
data_dir_trte  = os.path.join(crr_wd,'Data')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\' + str(trte_stock), 'rb'))
    
keys_stocks = list(TrainTest.keys())


count = 1


for model in keys_stocks:
    #==========================================================================
    #                          Filter Configuration
    #==========================================================================
    N_tr       = TrainTest[model]['Parameters']['NTrS']                         #Number of training samples
    N_te       = TrainTest[model]['Parameters']['NTeS']                         #Number of testing samples    
    time_delay = TrainTest[model]['Parameters']['TD']                           #Time delay (embedding length)    
    horizon    = TrainTest[model]['Parameters']['H']                            #Prediction horizon    
    eta        = TrainTest[model]['Parameters']['LR']                           #Learning rate
    sigma      = TrainTest[model]['Parameters']['KS']                           #Kernel bandwidth
    L          = 15                                                             #Most recent observations
    
    
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[model]['Train']['X_tr'].T 
    y_train = TrainTest[model]['Train']['y_tr'] 
    
    X_test = TrainTest[model]['Test']['X_te'].T
    y_test = TrainTest[model]['Test']['y_te']


    #==========================================================================
    #                   Kernel Affine Projection Algorithm (KAPA)
    #==========================================================================
    
    "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
    [inputDimension, trainSize] = np.shape(X_train)
    testSize                    = len(y_test)
    
    expansionCoefficient        = np.zeros([trainSize, 1])
    networkOutput               = np.zeros([L,1])
    
    expansionCoefficient[0][0]  = eta*y_train[0][0]
    
    for n in range(1,L):
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)
        
        ii      = list(range(0,n))
        u_train = X_train[:,n][np.newaxis, :]
        dicti   = X_train[:, ii[:]].T
        
        networkOutput[L-1][0]      = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))
        predictionError            = y_train[n][0] - networkOutput[L-1][0]
        expansionCoefficient[n][0] = eta*predictionError
        
        
    for n in range(L,trainSize):
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)
        
        ii      = list(range(0,n))    
        for kk in range(0,L):
            u_train              = X_train[:,n+kk-L+1][np.newaxis, :]
            dicti                = X_train[:, ii[:]].T
            networkOutput[kk][0] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))
            
        aprioriErr                      = y_train[n-L+1:n+1] - networkOutput
        expansionCoefficient[n-L+1:n+1] = expansionCoefficient[n-L+1:n+1] + eta*aprioriErr
            
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_te = np.zeros([testSize, 1])
    
    for jj in range(0,testSize):
        u_test = X_test[:,jj][np.newaxis, :]
        dicti  = X_train[:, ii[:]].T
        
        y_te[jj] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_test, dicti, sigma).T))
        
    count    = count + 1
    
    
    #==========================================================================
    #                            Performance Measures
    #==========================================================================
    err_test = y_test - y_te                                                    #Mean Squared Error (MSE)
    mse_test = np.mean(err_test**2)    
    mae_test = metrics.mean_absolute_error(y_test, y_te)                        #Mean Absolute Error (MAE)
    
    
    #==========================================================================
    #                              Saving Results
    #==========================================================================
    Results_KAPA = {}
    
    Results_KAPA['Regression']               = {}
    Results_KAPA['Regression']['Desired']    = y_test 
    Results_KAPA['Regression']['Prediction'] = y_te
    
    Results_KAPA['Measures']                 = {}
    Results_KAPA['Measures']['MSE']          = mse_test
    Results_KAPA['Measures']['MAE']          = mae_test
    
    
    pickle.dump(Results_KAPA, open('Results\\KAPA\\Results_KAPA_' + model + '.pkl', 'wb'))                             


