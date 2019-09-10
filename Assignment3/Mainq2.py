# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 00:08:12 2019

@author: akhil
"""
import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from autograd import grad
from BasisFunctions import BasisFunctions as bf

N = 500

M = 16

a = 0.001
b1 = 0.9
b2 = 0.999
e = 1e-8
eta = 1e-3

def sgdbatch(x,batchsize):
    
    index = np.random.randint(N)
    if batchsize < N:
        batch = x[index:index+batchsize,:]
    else:
        batch = x[:,:]
    return batch, index

def loss(xs,w,index,batchsize,y):
    y_hat = np.matmul(xs,w)
    
    if batchsize < N:
        diff = y[index:index+batchsize,:] - y_hat
    else:
        diff = y - y_hat
    return diff

def gausslhood(weights,noise_var):
    
    batchsize = 1
    xs, i = sgdbatch(phi_fou,batchsize)
    diff = loss(xs,weights,i,batchsize,y)
    nll = (N*np.log(2*np.pi*noise_var*noise_var))/2 + ((np.matmul(diff.T,diff))/(2*noise_var*noise_var))
    return nll

def initializevar(N,M):
    
    x = (2*lhs(1,N)) - 1

    y = 2*np.sin(2*np.pi*x) + np.sin(8*np.pi*x) + 0.5*np.sin(16*np.pi*x)
    
    y = y + 0.2*np.std(y)*np.random.randn(N,1)  
    phi = bf(x,N,M)
    phi_fou = phi.fourier_basis()

    return phi_fou,y,x

def generate_params(M):

    noise_var = np.random.random_sample((1,1))
    weights = np.random.random_sample((2*M+2,1))
    
    return weights, noise_var

weights,noise_var = generate_params(M)
phi_fou,y,x = initializevar(N,M)

grad_nll_w = grad(gausslhood,0)
grad_nll_v = grad(gausslhood,1)

iter = 10000

mw = 0
vw = 0
ms = 0
vs = 0

lossfunction = np.array([0])

for t in range(1,iter):
    
    gtw = np.reshape(grad_nll_w(weights,noise_var),(2*M+2,1))
    gt2w = np.power(gtw,2)
    
    gts = np.reshape(grad_nll_v(weights,noise_var),(1,1))
    gt2s = np.power(gts,2)
    
    mw = b1*mw + (1-b1)*gtw
    vw = b2*vw + (1-b2)*gt2w
    m_hatw = mw/(1-np.power(b1,t))
    v_hatw = vw/(1-np.power(b2,t))
    
    ms = b1*ms + (1-b1)*gts
    vs = b2*vs + (1-b2)*gt2s
    m_hats = ms/(1-np.power(b1,t))
    v_hats = vs/(1-np.power(b2,t))
    
    noise_var = noise_var - (eta*m_hats)/(np.sqrt(v_hats) + e)
    weights = weights - (eta*m_hatw)/(np.sqrt(v_hatw) + e)
    
    diff = y - np.matmul(phi_fou,weights)
    nll = (N*np.log(2*np.pi*noise_var*noise_var))/2 + ((np.matmul(diff.T,diff))/(2*noise_var*noise_var))
    
    lossfunction = np.append(lossfunction,nll)

x_star = np.linspace(-1,1,500)[:,None]
y_star = 2*np.sin(2*np.pi*x_star) + np.sin(8*np.pi*x_star) + 0.5*np.sin(16*np.pi*x_star)
phi_star = bf(x_star,N,M)
phi_star_fou = phi_star.fourier_basis()

yt = np.matmul(phi_star_fou,weights)
plt.plot(x_star,yt,'.')
plt.plot(x,y,'.')
plt.plot(x_star,y_star)
plt.show()

iterations = np.linspace(1,9999,9999)
lossfunction = lossfunction[1:]
plt.plot(iterations,lossfunction)
plt.show()