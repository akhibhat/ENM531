# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:13:17 2019

@author: akhil
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.metrics import confusion_matrix

def getdata():
    
    my_data = np.genfromtxt('Data_for_UCI_named.csv', delimiter=',')
    my_data = my_data[1:10001,:]
    
    mydatasize = np.size(my_data,0)
    
    bias = np.ones((mydatasize,1))
    
    my_data = np.hstack((bias,my_data))
    
    tdata_size = (2*mydatasize)//3
    
    t_data = my_data[0:tdata_size,:]
    
    x_star = t_data[:,0:13]

    y = np.ceil(t_data[:,13][:,None])
    
    shape = x_star.shape
    N = shape[0]
    
    return x_star, y, N, my_data, tdata_size

def sgdbatch(x,batchsize):
    
    index = np.random.randint(np.size(x,0))
    if batchsize < N:
        batch = x[index:index+batchsize,:]
    else:
        batch = x[:,:]
    return batch, index

def sigmoid(xs,w,index,batchsize):
    y_hat = np.matmul(xs,w)
    a = (1/(1 + np.exp(-y_hat)))
    return a

def bernlhood(weights):
    
    batchsize = 32
    xs,i = sgdbatch(x_star,batchsize)
    a = sigmoid(xs,weights,i,batchsize)
    loga = np.log(a)
    log_a = np.log(1-a)
    yd = y[i:i+batchsize,:]
    bceloss = -np.matmul(yd.T,loga) - np.matmul((1-yd).T, log_a)
    return bceloss

def generate_params(M):
    weights = np.random.random_sample((M,1))
    
    return weights


M = 13

b1 = 0.9
b2 = 0.999
e = 1e-8
eta = 1e-3

x_star, y, N, my_data, tdatasize = getdata()
weights = generate_params(M)

grad_bceloss_w = grad(bernlhood)

iter = 20000

mw = np.zeros((M,1))
vw = np.zeros((M,1))

lossfunction = 0

for t in range(1,iter):
    
    gtw = grad_bceloss_w(weights)
    gt2w = np.power(gtw,2)
    
    mw = b1*mw + (1-b1)*gtw
    vw = b2*vw + (1-b2)*gt2w
    m_hatw = mw/(1-np.power(b1,t))
    v_hatw = vw/(1-np.power(b2,t))
    weights = weights - (eta*m_hatw)/(np.sqrt(v_hatw) + e)

    bcel = bernlhood(weights)
    lossfunction = np.append(lossfunction,bcel)

iterations = np.linspace(1,20000,20000)
plt.plot(iterations, lossfunction)

traindata = my_data[tdatasize:-1,0:13]
trainy = np.ceil(my_data[tdatasize:-1,13][:,None])
ypred = np.matmul(traindata,weights)
apred = (1/(1 + np.exp(-ypred)))

ypredshape = apred.shape[0]

for ys in range(ypredshape):
    if apred[ys] >= 0.5:
        apred[ys] = 1
    else:
        apred[ys] = 0

cm = confusion_matrix(trainy,apred)
den = cm.sum()
num = cm.trace()
claAcc = (100*num)/den
