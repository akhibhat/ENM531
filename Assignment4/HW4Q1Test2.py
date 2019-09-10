# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:41:32 2019

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
    
#    nxbias = np.repeat(bias,N,axis = 0)
    
    return nxweights, bias

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

def l2loss(w1,w2,w3,b1,b2,b3):
    
    batchsize = N
    
    layer1 = actifunc(w1,b1,data)
    layer2 = actifunc(w2,b2,layer1)
#    outputxy = actifunc(w3,b3,layer2)
    outputxy = np.matmul(layer2,w3) + b3
#    outputxy = normalize(outputxy_unnorm)
    fx,y,i = sgdbatch(xy,outputxy,batchsize)
    diff = fx-y
    
    loss = np.matmul(diff.T,diff)
    
    return loss

xy_un,data_un = generate_traindata()

xy = normalize(xy_un)
data = normalize(data_un)

w1, b1 = generate_params(2,50)
w2, b2 = generate_params(50,50)
w3, b3 = generate_params(50,1)

#b3 = 0

gw1 = grad(l2loss,0)
gw2 = grad(l2loss,1)
gw3 = grad(l2loss,2)
gb1 = grad(l2loss,3)
gb2 = grad(l2loss,4)
gb3 = grad(l2loss,5)

it = 10000
eta = 0.001

b11 = 0.9
b21 = 0.999
e = 1e-8

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

lossfunction = np.array([0])

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
    
    mw3 = b11*mw3 + (1-b11)*gradw3
    vw3 = b21*vw3 + (1-b21)*gradw3_2
    
    mw2 = b11*mw2 + (1-b11)*gradw2
    vw2 = b21*vw2 + (1-b21)*gradw2_2
    
    mw1 = b11*mw1 + (1-b11)*gradw1
    vw1 = b21*vw1 + (1-b21)*gradw1_2
    
    mb3 = b11*mb3 + (1-b11)*gradb3
    vb3 = b21*vb3 + (1-b21)*gradb3_2
    
    mb2 = b11*mb2 + (1-b11)*gradb2
    vb2 = b21*vb2 + (1-b21)*gradb2_2
    
    mb1 = b11*mb1 + (1-b11)*gradb1
    vb1 = b21*vb1 + (1-b21)*gradb1_2
    
    m_hatw3 = mw3/(1-(np.power(b11,i)))
    v_hatw3 = vw3/(1-(np.power(b21,i)))
    
    m_hatw2 = mw2/(1-(np.power(b11,i)))
    v_hatw2 = vw2/(1-(np.power(b21,i)))
    
    m_hatw1 = mw1/(1-(np.power(b11,i)))
    v_hatw1 = vw1/(1-(np.power(b21,i)))
    
    m_hatb3 = mb3/(1-(np.power(b11,i)))
    v_hatb3 = vb3/(1-(np.power(b21,i)))
    
    m_hatb2 = mb2/(1-(np.power(b11,i)))
    v_hatb2 = vb2/(1-(np.power(b21,i)))
    
    m_hatb1 = mb1/(1-(np.power(b11,i)))
    v_hatb1 = vb1/(1-(np.power(b21,i)))
    
    w3 = w3 - (eta*m_hatw3)/(np.sqrt(v_hatw3) + e)
    w2 = w2 - (eta*m_hatw2)/(np.sqrt(v_hatw2) + e)
    w1 = w1 - (eta*m_hatw1)/(np.sqrt(v_hatw1) + e)
    
    b3 = b3 - (eta*m_hatb3)/(np.sqrt(v_hatb3) + e)
    b2 = b2 - (eta*m_hatb2)/(np.sqrt(v_hatb2) + e)
    b1 = b1 - (eta*m_hatb1)/(np.sqrt(v_hatb1) + e)
    
    H1 = actifunc(w1,b1,data)
    H2 = actifunc(w2,b2,H1)
    H_out = np.matmul(H2,w3) + b3
    
    diff = xy - H_out
    
    loss = np.matmul(diff.T,diff)
    lossfunction = np.append(lossfunction, loss)

iterations = np.linspace(0,it-1,it)
plt.plot(iterations,lossfunction[1:])

fxy, xy_star = generate_traindata()

xy_norm = (xy_star-np.mean(xy_star))/np.std(xy_star)
H1 = actifunc(w1,b1,xy_norm)
H2 = actifunc(w2,b2,H1)
fxy_nn = np.matmul(H2,w3) + b3

fxyout = fxy_nn * np.std(fxy_nn) + np.mean(fxy_nn)

finaldiff = fxy - fxyout
approxer = np.matmul(finaldiff.T,finaldiff)/(np.matmul(fxy.T,fxy))