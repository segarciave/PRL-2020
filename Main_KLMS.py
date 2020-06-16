# -*- coding: utf-8 -*-
"""
@author: Sergio GARCIA-VEGA
sergio.garcia-vega@postgrad.manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_KLMS.py
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
    
    
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[model]['Train']['X_tr'].T 
    y_train = TrainTest[model]['Train']['y_tr'] 
    
    X_test = TrainTest[model]['Test']['X_te'].T
    y_test = TrainTest[model]['Test']['y_te']



    #==========================================================================
    #                       Kernel Least-Mean-Square (KLMS)
    #==========================================================================
    err_train    = np.zeros((N_tr,1))                                           #Vector with errors (training)
    y_train_pred = np.zeros((N_tr,1))                                           #Vector with predictions (training)
    ev_pred      = np.zeros((N_tr,N_te))                                        #Matrix with predictions (testing)
    mse_test     = np.zeros((N_tr,1))                                           #Vector with 'MSE' values (testing)


    #==========================================================================
    #                              Initialization
    #==========================================================================
    err_train[0]    = y_train[0]                                                #Init: error (training)
    y_train_pred[0] = 0                                                         #Init: prediction (training)
    dicti           = X_train[:, 0][np.newaxis, :]                              #Init: dictionary (training)
    alphas          = eta*y_train[0]                                            #Init: weights (training)
    mse_test[0]     = np.mean(y_test**2)                                        #Init: MSE (testing)
    
    Dicti_size    = np.zeros((N_tr,1))
    Dicti_size[0] = len(dicti)
    
    n=1
    while n < N_tr:
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)
        
        "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
        
        "Input vector"
        u_train         = X_train[:,n][np.newaxis, :] 
        
        #Prediction (training)
        y_train_pred[n] = np.float(np.dot(aux.gaus_kernel(u_train, dicti, sigma), alphas))
        
        #Compute error (training)
        err_train[n]    = y_train[n] - y_train_pred[n]
        
        #Assign a new center
        dicti           = np.append(dicti,u_train,axis=0)
        alphas          = np.append(alphas,eta*err_train[n])
        
        Dicti_size[n] = len(dicti)
        
        n = n + 1
        
        
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_test_pred = np.zeros((N_te,1))
    i = 0
    
    while i < N_te:    
        "Input vector"
        u_test           = X_test[:, i][np.newaxis, :]
        
        #Prediction (testing)
        y_test_pred[i] = np.float(np.dot(aux.gaus_kernel(u_test,dicti,sigma), alphas))
        i = i + 1  
    
    count    = count + 1
    

    #==========================================================================
    #                            Performance Measures
    #==========================================================================
    err_test = y_test - y_test_pred                                             #Mean Squared Error (MSE)
    mse_test = np.mean(err_test**2)    
    mae_test = metrics.mean_absolute_error(y_test, y_test_pred)                #Mean Absolute Error (MAE)
    
    
    #==========================================================================
    #                              Saving Results
    #==========================================================================
    Results_KLMS = {}
    
    Results_KLMS['Regression']               = {}
    Results_KLMS['Regression']['Desired']    = y_test 
    Results_KLMS['Regression']['Prediction'] = y_test_pred
    
    Results_KLMS['Measures']                 = {}
    Results_KLMS['Measures']['MSE']          = mse_test
    Results_KLMS['Measures']['MAE']          = mae_test
    
    
    pickle.dump(Results_KLMS, open('Results\\KLMS\\Results_KLMS_' + model + '.pkl', 'wb'))           
                             


