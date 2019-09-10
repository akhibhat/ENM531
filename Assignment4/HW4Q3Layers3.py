# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:46:30 2019

@author: akhil
"""


import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.metrics import confusion_matrix
from NeuralNets import NeuralNets as nn

def getdata():
    
    my_data = np.genfromtxt('Data_for_UCI_named.csv', delimiter=',')
    my_data = my_data[1:10001,:]
    
    mydatasize = np.size(my_data,0)
    
    tdata_size = (2*mydatasize)//3
    
    t_data = my_data[0:tdata_size,:]
    
    x_star = t_data[:,0:12]

    y = np.ceil(t_data[:,12][:,None])
    
    shape = x_star.shape
    N = shape[0]
    
    return x_star, y, N, my_data, tdata_size

def sgdbatch(x,batchsize,N):
    
    index = np.random.randint(np.size(x,0)-32)
    if batchsize < N:
        batch = x[index:index+batchsize,:]
    else:
        batch = x[:,:]
    return batch, index

def sigmoid(xs,w,b):
    y_hat = np.matmul(xs,w) + b
    a = (1/(1 + np.exp(-y_hat)+e))
    return a

def relu(xs,w,b):
    y_hat = np.matmul(xs,w) + b
    a = np.maximum(y_hat,0)
    
    return a

def bceloss(w1,w2,w3,w_end,b1,b2,b3,b_end):
    
    batchsize = 32
    batchx, i = sgdbatch(trainx,batchsize,N)
    
    H1 = relu(batchx,w1,b1)
    H2 = relu(H1,w2,b2)
    H3 = relu(H2,w3,b3)
    y_out = sigmoid(H3,w_end,b_end)
    
    logy_out = np.log(y_out)
    log_yout = np.log(1-y_out)
    
    y_exact = trainy[i:i+batchsize,:]
    bceloss = -np.matmul(y_exact.T,logy_out) - np.matmul((1-y_exact).T,log_yout)
    
    return bceloss

trainx, trainy, N, my_data, trainsize = getdata()
layers = 3
indim = 12
neurons = 100
outdim = 1

it = 20000
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
mw_end = 0
vw_end = 0
mb1 = 0
vb1 = 0
mb2 = 0
vb2 = 0
mb3 = 0
vb3 = 0
mb_end = 0
vb_end = 0

batchsize = 32

net = nn(batchsize,layers,indim,neurons,outdim)
w1, b1, w_end, b_end = net.gen_params_firstlast()
w2, b2 = net.gen_params_hidden()
w3, b3 = net.gen_params_hidden()

gw1 = grad(bceloss,0)
gw2 = grad(bceloss,1)
gw3 = grad(bceloss,2)
gw_end = grad(bceloss,3)
gb1 = grad(bceloss,4)
gb2 = grad(bceloss,5)
gb3 = grad(bceloss,6)
gb_end = grad(bceloss,7)

lossfunction = np.array([0])

for i in range(1,it+1):
    gradw1 = gw1(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradw2 = gw2(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradw3 = gw3(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradw_end = gw_end(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradb1 = gb1(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradb2 = gb2(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradb3 = gb3(w1,w2,w3,w_end,b1,b2,b3,b_end)
    gradb_end = gb_end(w1,w2,w3,w_end,b1,b2,b3,b_end)
    
    gradw1_2 = np.power(gradw1,2)
    gradw2_2 = np.power(gradw2,2)
    gradw3_2 = np.power(gradw3,2)
    gradw_end_2 = np.power(gradw_end,2)
    gradb1_2 = np.power(gradb1,2)
    gradb2_2 = np.power(gradb2,2)
    gradb3_2 = np.power(gradb3,2)
    gradb_end_2 = np.power(gradb_end,2)
    
    mw_end = b11*mw_end + (1-b11)*gradw_end
    vw_end = b21*vw_end + (1-b21)*gradw_end_2
    
    mw3 = b11*mw3 + (1-b11)*gradw3
    vw3 = b21*vw3 + (1-b21)*gradw3_2
    
    mw2 = b11*mw2 + (1-b11)*gradw2
    vw2 = b21*vw2 + (1-b21)*gradw2_2
    
    mw1 = b11*mw1 + (1-b11)*gradw1
    vw1 = b21*vw1 + (1-b21)*gradw1_2
    
    mb_end = b11*mb_end + (1-b11)*gradb_end
    vb_end = b21*vb_end + (1-b21)*gradb_end_2
    
    mb3 = b11*mb3 + (1-b11)*gradb3
    vb3 = b21*vb3 + (1-b21)*gradb3_2
    
    mb2 = b11*mb2 + (1-b11)*gradb2
    vb2 = b21*vb2 + (1-b21)*gradb2_2
    
    mb1 = b11*mb1 + (1-b11)*gradb1
    vb1 = b21*vb1 + (1-b21)*gradb1_2
    
    m_hatw_end = mw_end/(1-(np.power(b11,i)))
    v_hatw_end = vw_end/(1-(np.power(b21,i)))
    
    m_hatw3 = mw3/(1-(np.power(b11,i)))
    v_hatw3 = vw3/(1-(np.power(b21,i)))
    
    m_hatw2 = mw2/(1-(np.power(b11,i)))
    v_hatw2 = vw2/(1-(np.power(b21,i)))
    
    m_hatw1 = mw1/(1-(np.power(b11,i)))
    v_hatw1 = vw1/(1-(np.power(b21,i)))
    
    m_hatb_end = mb_end/(1-(np.power(b11,i)))
    v_hatb_end = vb_end/(1-(np.power(b21,i)))
    
    m_hatb3 = mb3/(1-(np.power(b11,i)))
    v_hatb3 = vb3/(1-(np.power(b21,i)))
    
    m_hatb2 = mb2/(1-(np.power(b11,i)))
    v_hatb2 = vb2/(1-(np.power(b21,i)))
    
    m_hatb1 = mb1/(1-(np.power(b11,i)))
    v_hatb1 = vb1/(1-(np.power(b21,i)))
    
    w_end = w_end - (eta*m_hatw_end)/(np.sqrt(v_hatw_end) + e)
    w3 = w3 - (eta*m_hatw3)/(np.sqrt(v_hatw3) + e)
    w2 = w2 - (eta*m_hatw2)/(np.sqrt(v_hatw2) + e)
    w1 = w1 - (eta*m_hatw1)/(np.sqrt(v_hatw1) + e)
    
    b_end = b_end - (eta*m_hatb_end/(np.sqrt(v_hatb_end) + e))
    b3 = b3 - (eta*m_hatb3)/(np.sqrt(v_hatb3) + e)
    b2 = b2 - (eta*m_hatb2)/(np.sqrt(v_hatb2) + e)
    b1 = b1 - (eta*m_hatb1)/(np.sqrt(v_hatb1) + e)
    
    loss = bceloss(w1,w2,w3,w_end,b1,b2,b3,b_end)
    lossfunction = np.append(lossfunction, loss)

tdatasize = 6666
testdata = my_data[tdatasize:-1,0:12]
testy = np.ceil(my_data[tdatasize:-1,12][:,None])

layer1 = relu(testdata,w1,b1)
layer2 = relu(layer1,w2,b2)
layer3 = relu(layer2,w3,b3)
outputlayer = sigmoid(layer3,w_end,b_end)

outputshape = outputlayer.shape[0]

for ys in range(outputshape):
    if outputlayer[ys]>=0.5:
        outputlayer[ys] = 1
    else:
        outputlayer[ys] = 0

cm = confusion_matrix(testy,outputlayer)
den = cm.sum()
num = cm.trace()
claAcc = (100*num)/den
