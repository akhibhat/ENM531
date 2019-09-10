# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:01:09 2019

@author: akhil
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from LinearBasisModel import BayesianLinearRegression
from BasisFunctions import BasisFunctions as bf

if __name__ == "__main__":
    
    # N - no of points
    N = 10
    
    # no. of features
    M = 8
    
    # initializing gaussian noise
    noise_mean = 0
    noise_var = 0.5
    noise = np.reshape(np.random.normal(noise_mean,noise_var,N),(N,1))
    
    alpha = 5
    beta = 0.1
    
    x = 2*lhs(1,N)
    #x = np.reshape(np.linspace(0,2,N),[500,1])
    y = np.exp(x)*np.sin(2*np.pi*x) + noise
    
    # Defining the basis set
    phi = bf(x,N,M)
    phi_id = np.reshape(phi.identity_basis(),[N,1])
    phi_mon = phi.monomial_basis()
    phi_fou = phi.fourier_basis()
    phi_leg = phi.legendre_basis()
    
    # Defining the model
    blr_id = BayesianLinearRegression(phi_id,y,alpha,beta)
    blr_mon = BayesianLinearRegression(phi_mon, y, alpha, beta)
    blr_fou = BayesianLinearRegression(phi_fou,y,alpha,beta)
    blr_leg = BayesianLinearRegression(phi_leg,y,alpha,beta)
    
    # FIt identity MLE and MAP estimates for w
    w_MLE_id = blr_id.fit_MLE()
    w_MAP_id, Lambda_inv_id = blr_id.fit_MAP()
    
    # Fit monomial MLE and MAP estimates for w
    w_MLE_mon = blr_mon.fit_MLE()
    w_MAP_mon, Lambda_inv_mon = blr_mon.fit_MAP()
    
    # Fit fourier MLE and MAP estimates for w
    w_MLE_fou = blr_fou.fit_MLE()
    w_MAP_fou, Lambda_inv_fou = blr_fou.fit_MAP()
    
    # Fit legendre MLE and MAP estimates for w
    w_MLE_leg = blr_leg.fit_MLE()
    w_MAP_leg, Lambda_inv_leg = blr_leg.fit_MAP()
    
    # Using the results to predict at a set of points
    X_star = np.linspace(0,2,N)[:,None]
    
    phi_star = bf(X_star,N,M)
    phi_star_id = np.reshape(phi_star.identity_basis(),[N,1])
    phi_star_mon = phi_star.monomial_basis()
    phi_star_fou = phi_star.fourier_basis()
    phi_star_leg = phi_star.legendre_basis()
    
    y_MLE_id = np.matmul(phi_star_id,w_MLE_id)
    y_MAP_id = np.matmul(phi_star_id,w_MAP_id)
    
    y_MLE_mon = np.matmul(phi_star_mon,w_MLE_mon)
    y_MAP_mon = np.matmul(phi_star_mon,w_MAP_mon)
    
    y_MLE_fou = np.matmul(phi_star_fou,w_MLE_fou)
    y_MAP_fou = np.matmul(phi_star_fou,w_MAP_fou)
    
    y_MLE_leg = np.matmul(phi_star_leg,w_MLE_leg)
    y_MAP_leg = np.matmul(phi_star_leg,w_MAP_leg)
    
    # Predictive destribution
    num_samples = 500
    mean_star_id, var_star_id = blr_id.predictive_distribution(phi_star_id)
    samples_id = np.random.multivariate_normal(mean_star_id.flatten(),var_star_id,num_samples)
    
    mean_star_mon, var_star_mon = blr_mon.predictive_distribution(phi_star_mon)
    samples_mon = np.random.multivariate_normal(mean_star_mon.flatten(),var_star_mon,num_samples)
    
    mean_star_fou, var_star_fou = blr_fou.predictive_distribution(phi_star_fou)
    samples_fou = np.random.multivariate_normal(mean_star_fou.flatten(),var_star_fou,num_samples)
    
    mean_star_leg, var_star_leg = blr_leg.predictive_distribution(phi_star_leg)
    samples_leg = np.random.multivariate_normal(mean_star_leg.flatten(),var_star_leg,num_samples)
    
    # PLotting all fits
    f1 = plt.figure()
    plt.plot(x,y,'.')
    plt.plot(X_star,y_MAP_fou,'r')
    plt.plot(X_star,y_MLE_fou,'k')
    for i in range(0,num_samples):
        plt.plot(X_star, samples_fou[i,:],'m',linewidth = 0.05,alpha = 0.4)
    plt.show()

    f2 = plt.figure()
    plt.plot(x,y,'.')
    plt.plot(X_star,y_MLE_mon,'r')
    plt.plot(X_star,y_MAP_mon,'k')
    for i in range(0,num_samples):
        plt.plot(X_star, samples_mon[i,:],'m',linewidth = 0.05,alpha = 0.4)
    plt.show()
    
    f3 = plt.figure()
    plt.plot(x,y,'.')
    plt.plot(X_star,y_MLE_id,'r')
    plt.plot(X_star,y_MAP_id,'k',linewidth = 0.5)
    for i in range(0,num_samples):
        plt.plot(X_star, samples_id[i,:],'m',linewidth = 0.05,alpha = 0.4)
    plt.show()
    
    f4 = plt.figure()
    plt.plot(x,y,'.')
    plt.plot(X_star,y_MLE_leg,'r')
    plt.plot(X_star,y_MAP_leg,'k')
    for i in range(0,num_samples):
        plt.plot(X_star,samples_leg[i,:],'m',linewidth = 0.05,alpha = 0.4)
    plt.show()
    
    # Plotting the basis
    
    plt.subplot(2,2,1)
    plt.plot(X_star,phi_star_id)
    
    plt.subplot(2,2,2)
    plt.plot(X_star,phi_star_mon)
    
    plt.subplot(2,2,3)
    plt.plot(X_star,phi_star_fou)
    
    plt.subplot(2,2,4)
    plt.plot(X_star,phi_star_leg)
    
    plt.show()
    