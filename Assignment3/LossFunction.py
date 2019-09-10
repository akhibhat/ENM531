# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:29:12 2019

@author: akhil
"""

import numpy as np

 
def __init__(self, phi, y, theta, N):
       
  self.x = phi
  self.y = y
  self.theta = theta
  self.N = N

def gausslhood(self):
    
    thetalen = self.theta.size
    weights = self.theta[:thetalen-1]
    var = self.theta[thetalen-1]
    
    diff = self.y - np.matmul(self.x,weights)
    
    
    nll = (self.N/2)*np.log(2*np.pi*var) + (1/2*np.square(var))*np.matmul(diff.T,diff)
            
    return nll


