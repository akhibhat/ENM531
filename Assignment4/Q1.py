# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:56:37 2019

@author: akhil
"""

import numpy as np
from pyDOE import lhs
#import torch

N = 500

x = 50 + 4*lhs(1,N)
y = 50 + 4*lhs(1,N)
xstar = (x-np.mean(x))/np.std(x)
ystar = (y-np.mean(y))/np.std(y)

fxy = np.cos(np.pi*xstar)*np.cos(np.pi*ystar)

layers = 2
dim = 2
nsperlayer = 50

def generate_params_layer1():
    inputs = 2
    weights = np.random.normal(0,2/(inputs + nsperlayer),(dim,nsperlayer))
    
    bias1 = np.random.random_sample((1,nsperlayer))
    bias = np.repeat(bias1,N,axis = 0)
    return weights, bias

def actifunc(w,b,x):
    y = np.matmul(x,w) + b
    y = np.tanh(y)
    return y

def generate_params_layer2():
    inputs = 50
    weights = np.random.normal(0,2/(inputs + nsperlayer),(inputs,nsperlayer))

    bias1 = np.random.random_sample((1,nsperlayer))
    bias = np.repeat(bias1,N,axis = 0)
    return weights, bias

def generate_params_outputlayer():
    inputs = 50
    outputs = 1
    weights = np.random.normal(0,2/(inputs + outputs),(inputs,outputs))
    bias1 = np.random.random_sample((1,1))
    bias = np.repeat(bias1,N,axis = 0)
    return weights, bias

def sgdbatch(x,batchsize):
    
    index = np.random.randint(N)
    if batchsize < N:
        batch = x[index:index+batchsize,:]
    else:
        batch = x[:,:]
    return batch, index

# Are we supposed to standardize the data?
x_star = np.linspace(-2,2,N)[:,None]
#x = (xstar-np.mean(xstar))/np.std(xstar)

y_star = np.linspace(-2,2,N)[:,None]
#y = (ystar-np.mean(ystar))/np.std(ystar)

xy = np.hstack((x_star,y_star))

weights,bias = generate_params_layer1()
H1 = actifunc(weights,bias,xy)

weights2, bias2 = generate_params_layer2()
H2 = actifunc(weights2,bias2,H1)

weights3, bias3 = generate_params_outputlayer()
xyout = np.matmul(H2,weights3) + bias3

def loss():
    lam = 0.1
    batchsize = 1
    batch, index = sgdbatch(fxy,batchsize)
    xydata = xyout[index:index+batchsize,:]
    
    diff = batch-xydata
    loss = np.matmul(diff.T,diff) + lam*np.matmul(weights3.T,weights3)
    
    return loss
    