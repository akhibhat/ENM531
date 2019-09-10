# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:59:09 2019

@author: akhil
"""

import autograd.numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from autograd import grad


N = 500

def generate_traindata():
    x = 50 + 4*lhs(1,N)
    y = 50 + 4*lhs(1,N)
    
    data = np.hstack((x,y))
    
    xy = np.cos(np.pi*x)*np.cos(np.pi*y)
    
    return xy,data

def normalize(i):
    
    meani = np.mean(i)
    stdi = np.std(i)
    
    normi = (i - meani)/stdi
    
    return normi

def generate_params(ninputs,noutputs):
    
    nxweights = np.random.normal(0,(2/(ninputs+noutputs)),(ninputs,noutputs))
    bias = np.zeros((1,noutputs))
    
    nxbias = np.repeat(bias,N,axis = 0)
    
    return nxweights, nxbias

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

def reg_term(w):
    
    lam = 4
    r = lam*np.matmul(w.T,w)
    return r

def l2loss(w1,w2,w3,b1,b2,b3):
    
    batchsize = 1
    
    layer1 = actifunc(w1,b1,data)
    layer2 = actifunc(w2,b2,layer1)
    outputxy = actifunc(w3,b3,layer2)
#    outputxy = normalize(outputxy_unnorm)
    fx,y,i = sgdbatch(xy,outputxy,batchsize)
    diff = fx-y
    
    loss = np.matmul(diff.T,diff)
    
    return loss

xy,data = generate_traindata()

#xy = normalize(xy_unnorm)
#
#data = np.zeros((N,2))
#data[:,0] = normalize(data_unnorm[:,0])
#data[:,1] = normalize(data_unnorm[:,1])

w1, b1 = generate_params(2,50)
w2, b2 = generate_params(50,50)
w3, b3 = generate_params(50,1)

#outputxy = get_output(data,w1,w2,w3,b1,b2,b3)

gw1 = grad(l2loss,0)
gw2 = grad(l2loss,1)
gw3 = grad(l2loss,2)
gb1 = grad(l2loss,3)
gb2 = grad(l2loss,4)
gb3 = grad(l2loss,5)


it = 1000
eta = 0.05

b1 = 0.9
b2 = 0.999
e = 1e-8

lossfunction = np.array([0])

mw1 = 0
vw1 = 0
mw2 = 0
vw2 = 0
mw3 = 0
vw3 = 0
mb1 = 0
vb1 = 0
mb2 = 0
vb2 = 0
mb3 = 0
vb3 = 0

#def update_params(param,i,gradient,m,v):
#    gradient_sq = np.power(gradient,2)
#    m = b1*m + (1-b1)*gradient_sq
#    v = b2*v + (1-b2)*gradient_sq
#    
#    m_hat = m/(1-np.power(b1,i))
#    v_hat = v/(1-np.power(b2,i))
#    
#    param = param - (eta*m_hat)/(np.sqrt(v_hat) + e)   
    

for i in range(1,it+1):
    gradw1 = gw1(w1,w2,w3,b1,b2,b3)
    gradw2 = gw2(w1,w2,w3,b1,b2,b3)
    gradw3 = gw3(w1,w2,w3,b1,b2,b3)
    gradb1 = gb1(w1,w2,w3,b1,b2,b3)
    gradb2 = gb2(w1,w2,w3,b1,b2,b3)
    gradb3 = gb3(w1,w2,w3,b1,b2,b3)
    
    gradw1_2 = np.power(gradw1,2)
    gradw2_2 = np.power(gradw2,2)
    gradw3_2 = np.power(gradw3,2)
    gradb1_2 = np.power(gradb1,2)
    gradb2_2 = np.power(gradb2,2)
    gradb3_2 = np.power(gradb3,2)
    
    mw1 = b1*mw1 + (1-b1)*gradw1
    vw1 = b2*vw1 + (1-b2)*gradw1_2
    m_hatw1 = mw1/(1-np.power(b1,i))
    v_hatw1 = vw1/(1-np.power(b2,i))
    
    mw2 = b1*mw2 + (1-b1)*gradw2
    vw2 = b2*vw2 + (1-b2)*gradw2_2
    m_hatw2 = mw2/(1-np.power(b1,i))
    v_hatw2 = vw2/(1-np.power(b2,i))
    
    mw3 = b1*mw3 + (1-b1)*gradw3
    vw3 = b2*vw3 + (1-b2)*gradw3_2
    m_hatw3 = mw3/(1-np.power(b1,i))
    v_hatw3 = vw3/(1-np.power(b2,i))
    
    mb1 = b1*mb1 + (1-b1)*gradb1
    vb1 = b2*vb1 + (1-b2)*gradb1_2
    m_hatb1 = mb1/(1-np.power(b1,i))
    v_hatb1 = vb1/(1-np.power(b2,i))
    
    mb2 = b1*mb2 + (1-b1)*gradb2
    vb2 = b2*vb2 + (1-b2)*gradb2_2
    m_hatb2 = mb2/(1-np.power(b1,i))
    v_hatb2 = vb2/(1-np.power(b2,i))
    
    mb3 = b1*mb3 + (1-b1)*gradb3
    vb3 = b2*vb3 + (1-b2)*gradb3_2
    m_hatb3 = mb3/(1-np.power(b1,i))
    v_hatb3 = vb3/(1-np.power(b2,i))
    
    w1 = w1 - (eta*m_hatw1)/(np.sqrt(v_hatw1) + e)
    w2 = w2 - (eta*m_hatw2)/(np.sqrt(v_hatw2) + e)
    w3 = w3 - (eta*m_hatw3)/(np.sqrt(v_hatw3) + e)
    
    b1 = b1 - (eta*m_hatb1)/(np.sqrt(v_hatb1) + e)
    b2 = b2 - (eta*m_hatb2)/(np.sqrt(v_hatb2) + e)
    b3 = b3 - (eta*m_hatb3)/(np.sqrt(v_hatb3) + e)
    
#    w1 = w1 - eta*gradw1
#    w2 = w2 - eta*gradw2
#    w3 = w3 - eta*gradw3
#    b1 = b1 - eta*gradb1
#    b2 = b2 - eta*gradb2
#    b3 = b3 - eta*gradb3


    H1 = actifunc(w1,b1,data)
    H2 = actifunc(w2,b2,H1)
    H_out = actifunc(w3,b3,H2)
    
    diff = xy - H_out
    
    loss = np.matmul(diff.T,diff) + reg_term(w3)
    lossfunction = np.append(lossfunction, loss)

#it = np.linspace(0,iter-1,iter)
#plt.plot(it,lossfunction[1:])