# -*- coding: utf-8 -*-
"""
@author: Sergio GARCIA-VEGA
sergio.garcia-vega@postgrad.manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_KMEE.py
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



#==============================================================================
#                   Finding Kernel width for density estimation
#==============================================================================

count = 1
for model in keys_stocks:
    
    sig_kde    = np.linspace(0.1,1.5,10)
    err_sigmas = np.zeros(len(sig_kde))
    count_sig  = 0
    
    for sigma_kde in sig_kde:
        
        print('Finding KDE bandwidth: ' + str(count_sig+1) + '/' + str(len(err_sigmas)) + ' <-- Stock ' + str(count) + '/24')
    
        #======================================================================
        #                          Filter Configuration
        #======================================================================
        N_tr       = TrainTest[model]['Parameters']['NTrS']                     #Number of training samples
        N_te       = TrainTest[model]['Parameters']['NTeS']                     #Number of testing samples    
        time_delay = TrainTest[model]['Parameters']['TD']                       #Time delay (embedding length)    
        horizon    = TrainTest[model]['Parameters']['H']                        #Prediction horizon    
        eta        = TrainTest[model]['Parameters']['LR']                       #Learning rate
        sigma      = TrainTest[model]['Parameters']['KS']                       #Kernel bandwidth
        L          = 15                                                         #Most recent observations
        #======================================================================
        #                              Data Embedding
        #======================================================================
        X_train = TrainTest[model]['Train']['X_tr'].T 
        y_train = TrainTest[model]['Train']['y_tr'] 
        X_test  = TrainTest[model]['Test']['X_te'].T
        y_test  = TrainTest[model]['Test']['y_te']
        #======================================================================
        #                Kernel Minimum Error Entropy Algorithm (KMEE)
        #======================================================================
        "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
        [inputDimension, trainSize] = np.shape(X_train)
        testSize                    = len(y_test)
        expansionCoefficient        = np.zeros([trainSize, 1])
        networkOutput               = np.zeros([L,1])
        errors                      = np.zeros([L,1])
        expansionCoefficient[0][0]  = eta*y_train[0][0]
        for n in range(1,L):
            ii      = list(range(0,n))
            u_train = X_train[:,n][np.newaxis, :]
            dicti   = X_train[:, ii[:]].T
            networkOutput[L-1][0]      = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))
            predictionError            = y_train[n][0] - networkOutput[L-1][0]
            errors[n][0]               = eta*predictionError
            kde_a                      = (1)/(L*np.sqrt(2*np.pi)*sigma_kde**3)
            kde_b                      = np.exp(-(errors[n][0]-errors[ii[:],:])**2/(2*sigma_kde**2)) 
            kde_c                      = errors[n][0]-errors[ii[:],:]
            kde_d                      = aux.gaus_kernel(u_train, dicti, sigma).T
            expansionCoefficient[n][0] = eta*kde_a*np.sum(kde_b*kde_c*kde_d)
        for n in range(L,trainSize):
            ii      = list(range(0,n))    
            for kk in range(0,L):
                u_train              = X_train[:,n+kk-L+1][np.newaxis, :]
                dicti                = X_train[:, ii[:]].T
                networkOutput[kk][0] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))
            aprioriErr                      = y_train[n-L+1:n+1] - networkOutput        
            kde_a                           = (1)/(L*np.sqrt(2*np.pi)*sigma_kde**3)
            kde_b                           = np.exp(-(aprioriErr[-1]-aprioriErr[-L:])**2/(2*sigma_kde**2))        
            kde_c                           = aprioriErr[-1]-aprioriErr[-L:]
            kde_d                           = aux.gaus_kernel(u_train, dicti[-L:], sigma).T        
            expansionCoefficient[n-L+1:n+1] = expansionCoefficient[n-L+1:n+1] + eta*kde_a*np.sum(kde_b*kde_c*kde_d)
        "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
        y_te = np.zeros([testSize, 1])
        for jj in range(0,testSize):
            u_test = X_test[:,jj][np.newaxis, :]
            dicti  = X_train[:, ii[:]].T
            y_te[jj] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_test, dicti, sigma).T))
        #======================================================================
        #                            Performance Measures
        #======================================================================
        err_test              = y_test - y_te                                   #Mean Squared Error (MSE)
        mse_test              = np.mean(err_test**2)
        err_sigmas[count_sig] = mse_test
        count_sig             = count_sig + 1
        mae_test              = metrics.mean_absolute_error(y_test, y_te)       #Mean Absolute Error (MAE)
    
        




    #==========================================================================
    #                                 MAIN KMEE
    #==========================================================================  
    N_tr       = TrainTest[model]['Parameters']['NTrS']                         #Number of training samples
    N_te       = TrainTest[model]['Parameters']['NTeS']                         #Number of testing samples    
    time_delay = TrainTest[model]['Parameters']['TD']                           #Time delay (embedding length)    
    horizon    = TrainTest[model]['Parameters']['H']                            #Prediction horizon    
    eta        = TrainTest[model]['Parameters']['LR']                           #Learning rate
    sigma      = TrainTest[model]['Parameters']['KS']                           #Kernel bandwidth
    sigma_kde  = sig_kde[np.argmin(err_sigmas)]                                 #Kernel width for density estimation
    L          = 15                                                             #Most recent observations
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[model]['Train']['X_tr'].T 
    y_train = TrainTest[model]['Train']['y_tr'] 
    X_test  = TrainTest[model]['Test']['X_te'].T
    y_test  = TrainTest[model]['Test']['y_te']
    #==========================================================================
    #                Kernel Minimum Error Entropy Algorithm (KMEE)
    #==========================================================================
    "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
    [inputDimension, trainSize] = np.shape(X_train)
    testSize                    = len(y_test)
    expansionCoefficient        = np.zeros([trainSize, 1])
    networkOutput               = np.zeros([L,1])
    errors                      = np.zeros([L,1])
    expansionCoefficient[0][0]  = eta*y_train[0][0]
    for n in range(1,L):
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)    
        ii      = list(range(0,n))
        u_train = X_train[:,n][np.newaxis, :]
        dicti   = X_train[:, ii[:]].T    
        networkOutput[L-1][0]      = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))
        predictionError            = y_train[n][0] - networkOutput[L-1][0]
        errors[n][0]               = eta*predictionError    
        kde_a                      = (1)/(L*np.sqrt(2*np.pi)*sigma_kde**3)
        kde_b                      = np.exp(-(errors[n][0]-errors[ii[:],:])**2/(2*sigma_kde**2)) 
        kde_c                      = errors[n][0]-errors[ii[:],:]
        kde_d                      = aux.gaus_kernel(u_train, dicti, sigma).T
        expansionCoefficient[n][0] = eta*kde_a*np.sum(kde_b*kde_c*kde_d)
    for n in range(L,trainSize):
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)    
        ii      = list(range(0,n))    
        for kk in range(0,L):
            u_train              = X_train[:,n+kk-L+1][np.newaxis, :]
            dicti                = X_train[:, ii[:]].T
            networkOutput[kk][0] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_train, dicti, sigma).T))        
        aprioriErr                      = y_train[n-L+1:n+1] - networkOutput        
        kde_a                           = (1)/(L*np.sqrt(2*np.pi)*sigma_kde**3)
        kde_b                           = np.exp(-(aprioriErr[-1]-aprioriErr[-L:])**2/(2*sigma_kde**2))        
        kde_c                           = aprioriErr[-1]-aprioriErr[-L:]
        kde_d                           = aux.gaus_kernel(u_train, dicti[-L:], sigma).T        
        expansionCoefficient[n-L+1:n+1] = expansionCoefficient[n-L+1:n+1] + eta*kde_a*np.sum(kde_b*kde_c*kde_d)
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_te = np.zeros([testSize, 1])
    for jj in range(0,testSize):
        u_test = X_test[:,jj][np.newaxis, :]
        dicti  = X_train[:, ii[:]].T
        y_te[jj] = float(np.dot(expansionCoefficient[ii].T, aux.gaus_kernel(u_test, dicti, sigma).T))
    #==========================================================================
    #                            Performance Measures
    #==========================================================================
    err_test = y_test - y_te                                                    #Mean Squared Error (MSE)
    mse_test = np.mean(err_test**2)
    mae_test = metrics.mean_absolute_error(y_test, y_te)                        #Mean Absolute Error (MAE)
    #==========================================================================
    #                              Saving Results
    #==========================================================================
    Results_KMEE = {}
    Results_KMEE['Regression']                        = {}
    Results_KMEE['Regression']['Desired']             = y_test 
    Results_KMEE['Regression']['Prediction']          = y_te
    Results_KMEE['Measures']                          = {}
    Results_KMEE['Measures']['MSE']                   = mse_test
    Results_KMEE['Measures']['MAE']                   = mae_test
    Results_KMEE['Parameters']                        = {}
    Results_KMEE['Parameters']['Number Training']     = N_tr
    Results_KMEE['Parameters']['Number Testing']      = N_te
    Results_KMEE['Parameters']['Time Delay']          = time_delay
    Results_KMEE['Parameters']['Horizon']             = horizon
    Results_KMEE['Parameters']['Learning Rate']       = eta 
    Results_KMEE['Parameters']['Recent Observations'] = L
    Results_KMEE['Parameters']['Bandwidth Kernel']    = sigma
    Results_KMEE['Parameters']['Bandwidth KDE']       = sigma_kde
    Results_KMEE['Parameters']['KDE']                 = {}
    Results_KMEE['Parameters']['KDE']['Sigmas']       = sig_kde
    Results_KMEE['Parameters']['KDE']['MSE']          = err_sigmas

    pickle.dump(Results_KMEE, open('Results\\KMEE\\Results_KMEE_' + model + '.pkl', 'wb'))
    
    count    = count + 1                        


