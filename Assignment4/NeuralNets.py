# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:46:13 2019

@author: akhil
"""
"""
Class NeuralNets to implement a neural network with l layers, each layer having n neurons
Input is of dimension indim
Output is of dimension outdim
N is the size of data
"""
import numpy as np

class NeuralNets:
    
    def __init__(self,N,layers,indim,neurons,outdim):
        
        self.N = N
        self.layers = layers
        self.indim = indim
        self.neurons = neurons
        self.outdim = outdim
    
    def gen_params_firstlast(self):
        
        weights_1 = np.random.normal(0,(2/(self.indim + self.neurons)),(self.indim,self.neurons))        
        bias_1 = np.zeros((1,self.neurons))
        
        weights_end = np.random.normal(0,(2/(self.neurons + self.outdim)),(self.neurons,self.outdim))
        bias_end = np.zeros((1,self.outdim))
        
        return weights_1, bias_1, weights_end, bias_end
    
    def gen_params_hidden(self):
        
        weight = np.random.normal(0,(1/self.neurons),(self.neurons,self.neurons))
        bias = np.zeros((1,self.neurons))
        
        return weight,bias
    
    