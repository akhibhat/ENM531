# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:48:08 2019

@author: akhil
"""
import autograd.numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from autograd import grad, hessian, elementwise_grad
from NeuralNets import NeuralNets as nn

def gen_data():
    Nu_out = np.zeros((Nu,1))
    
    x = 2*lhs(1,Nf) - 1
    x = np.append(x,1)[:,None]
    x = np.append(x,-1)[:,None]
    
    fx = -((np.pi*np.pi) - lam)*np.sin(np.pi*x)
    
    return x,fx,Nu_out

def actifunc(w,b,x):
    
    y = np.matmul(x,w) + b
    y = np.tanh(y)
    return y

def sgdbatch(x,y,batchsize):
    
    index = np.random.randint(N)
    if batchsize < N:
        batchtrain = x[index:index+batchsize,:]
        batchout = y[index:index+batchsize,:]
    else:
        batchtrain = x[:,:]
        batchout = y[:,:]
    return batchtrain, batchout, index

def gen_u(x):
    
    batchsize = N
    batchx,batchy,index = sgdbatch(x,fx,batchsize)
    H1 = actifunc(w1,b1,batchx)
    H2 = actifunc(w2,b2,H1)
    u = np.matmul(H2,w_end) + b_end
    
#    gu = grad(u,x)
    
    return u

def loss(w1,w2,w_end,b1,b2,b_end):
    
    loss = 1
    return loss


N = 500
Nu = 2
Nf = N-Nu
lam = 1

layers = 2
indim = 1
outdim = 1
neurons = 50

x,fx,Nu_out = gen_data()

net = nn(N,layers,indim,neurons,outdim)
w1, b1, w_end, b_end = net.gen_params_firstlast()
w2, b2 = net.gen_params_hidden()

gu = grad(gen_u,0)
g2u = grad(gu)

gx = gu(x)