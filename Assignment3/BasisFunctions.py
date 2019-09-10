# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:11:57 2019

@author: akhil
"""

import numpy as np
from scipy.special import legendre


class BasisFunctions:
    
    def __init__(self,x,N,M):
        
        self.x = x
        self.N = N
        self.M = M
        
    def identity_basis(self):
        phi = self.x
        
        return phi
    
    def monomial_basis(self):
        
        phi = np.reshape(np.ones(self.N),self.x.shape)
        
        phi_mon = phi;
        
        for i in range(1,(self.M)+1):
            phi_mon = np.concatenate((phi_mon,self.x**i),axis = 1)
        
        return phi_mon
    
    def fourier_basis(self):
        phi0 = np.reshape(np.zeros(self.N),self.x.shape)
        phi1 = np.reshape(np.ones(self.N),self.x.shape)
        
        phi_fou = np.concatenate((phi0,phi1),axis = 1)
        #phi_fou = phi1
        
        for i in range(1,(self.M)+1):
            psin = np.sin(i*np.pi*self.x)
            pcos = np.cos(i*np.pi*self.x)
            phi_fou = np.concatenate((phi_fou,psin,pcos),axis = 1)
            
        return phi_fou
    
    def legendre_basis(self):
        
        leg1 = legendre(0)
        phi_leg = leg1(self.x)
        
        for i in range(1,(self.M)+1):
            leg1 = legendre(i)
            phi_leg = np.concatenate((phi_leg,leg1(self.x)),axis = 1)
            
        return phi_leg
            