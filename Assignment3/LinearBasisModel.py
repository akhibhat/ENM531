# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 23:57:22 2019

@author: akhil
"""

import numpy as np

class BayesianLinearRegression:
  """   
    Linear regression model: y = (w.T)*phi + epsilon
    w ~ N(0,beta^(-1)I)
    P(y|phi,w) ~ N(y|(w.T)*phi,alpha^(-1)I)
  """
  def __init__(self, phi, y, alpha = 1.0, beta = 1.0):
           
      self.X = phi
      self.y = y
      
      self.alpha = alpha
      self.beta = beta
      
      self.jitter = 1e-8
      
      
  def fit_MLE(self):
      phiTphi_inv = np.linalg.inv(np.matmul(self.X.T,self.X) + self.jitter)
      phiTy = np.matmul(self.X.T, self.y)
      
      w_MLE = np.matmul(phiTphi_inv,phiTy)
      
      self.w_MLE = w_MLE
      
      return w_MLE
  
  def fit_MAP(self):
      Lambda = np.matmul(self.X.T,self.X) + (self.beta/self.alpha)*np.eye(self.X.shape[1])
      Lambda_inv  = np.linalg.inv(Lambda)
      phiTy = np.matmul(self.X.T,self.y)
      mu = np.matmul(Lambda_inv,phiTy)
      
      self.w_MAP = mu
      self.Lambda_inv = Lambda_inv
      
      return mu, Lambda_inv
  
  def predictive_distribution(self,x_star):
      
      mean_star = np.matmul(x_star, self.w_MAP)
      var_star = 1/self.alpha + np.matmul(x_star, np.matmul(self.Lambda_inv, x_star.T))
      
      return mean_star, var_star
  
      
          