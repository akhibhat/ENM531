# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:49:17 2019

@author: akhil
"""

import numpy as np
import tensorflow as tf

class TFNets:
    
    
    def __init__(self,N,layers,indim,neurons,outdim):
        
        self.N = N
        self.layers = layers
        self.indim = indim
#        self.x = x
        self.neurons = neurons
        self.outdim = outdim
        
    
    def gen_flparameters(self):
        
        xavier_stddev1 = 1. / np.sqrt((self.indim + self.neurons) / 2.)
        xavier_stddevend = 1. / np.sqrt((self.neurons + self.outdim) / 2.)
        w1 = tf.Variable(tf.random_normal([self.indim, self.neurons], dtype=tf.float64) * xavier_stddev1, dtype=tf.float64)
        w_end = tf.Variable(tf.random_normal([self.neurons,self.outdim], dtype=tf.float64) * xavier_stddevend, dtype=tf.float64)
        
        b1 = tf.Variable(tf.zeros([1,self.neurons], dtype=tf.float64), dtype=tf.float64)
        b_end = tf.Variable(tf.zeros([1,self.outdim], dtype=tf.float64), dtype=tf.float64)
        
        return w1, w_end, b1, b_end
    
    def gen_hiddenparameters(self):
        
        xavier_stddevmid = 1. / np.sqrt((self.neurons + self.neurons) / 2.)
        w = tf.Variable(tf.random_normal([self.neurons,self.neurons], dtype=tf.float64) * xavier_stddevmid, dtype=tf.float64)
        
        b = tf.Variable(tf.zeros([1,self.neurons], dtype=tf.float64), dtype=tf.float64)
        
        return w, b
        
        