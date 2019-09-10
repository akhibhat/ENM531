# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:20:23 2019

@author: akhil
"""
import numpy as np
import tensorflow as tf
from pyDOE import lhs
import time
from matplotlib import pyplot as plt


def xavier_weights(size):
    indim = size[0]
    outdim = size[1]
    
    std = np.sqrt(2/(indim+outdim))
    
    xw = tf.Variable(tf.random_normal([indim, outdim], stddev = std), dtype = tf.float32)
    
    return xw

def initialize_neuralnets(layers):
    weights = []
    biases = []
    nlayers = len(layers)
    
    for i in range(0,nlayers-1):
        w = xavier_weights(size = [layers[i], layers[i+1]])
        b = tf.Variable(tf.zeros([1, layers[i+1]], dtype = tf.float32), dtype = tf.float32)
        weights.append(w)
        biases.append(b)
    return weights, biases

def forward_pass(X, weights, bias):
    layers = len(weights) + 1
    H = 2.0 * (X - lb)/(ub - lb) - 1.0
    
    for i in range(0, layers-2):
        W = weights[i]
        b = bias[i]
        H = tf.tanh(tf.add(tf.matmul(H,W), b))
    W = weights[-1]
    b = bias[-1]
    output = tf.add(tf.matmul(H,W), b)
    
    return output

def nnet_u(x):
    out_u = forward_pass(x, weights, bias)
    
    return out_u

def nnet_f(x):
    out = nnet_u(x)
    grad_out = tf.gradients(out,x)[0]
    grad2_out = tf.gradients(grad_out,x)[0]
    
    out_f = grad2_out - out + (np.power(pi,2) + 1)*tf.sin(pi*x)
    
    return out_f

def train_NNet(iterations):
    
    tfdict = {xu_ph: x_u, yu_ph: y_u, xf_ph: x_f}
    
    start_time = time.time()
    for i in range(iterations):
        sess.run(train_AdamOptimizer, tfdict)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            loss_value = sess.run(loss, tfdict)
            print('Iteration: %d, Loss: %.3e, Time: %.2f' %(i, loss_value, elapsed))
            start_time = time.time()

def predict_output(x_star):
    tfdict = {xu_ph: x_star}
    u_star = sess.run(xu_pred, tfdict)
    return u_star
            

iterations = 10000
N= 2000
lb = -1
ub = 1
layers = [1, 50, 50, 1]

pi = tf.constant(np.pi)
    
x_u = np.array([[-1], [1]])
y_u = np.array([[0], [0]])

x_f = 2*lhs(1,N) - 1

weights, bias = initialize_neuralnets(layers)

sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True))

lam = tf.Variable([1], dtype=tf.float32)

xu_ph = tf.placeholder(tf.float32, shape = [None, x_u.shape[1]])
xf_ph = tf.placeholder(tf.float32, shape = [None, x_f.shape[1]])
yu_ph = tf.placeholder(tf.float32, shape = [None, y_u.shape[1]])

xu_pred = nnet_u(xu_ph)
xf_pred = nnet_f(xf_ph)

loss = tf.reduce_mean(tf.square(yu_ph - xu_pred)) + tf.reduce_mean(tf.square(xf_pred))
optimizer = tf.train.AdamOptimizer()
train_AdamOptimizer = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
train_NNet(iterations)

x_star = np.linspace(-1,1,N)[:,None]
u_star = np.sin(np.pi*x_star)
u_star_predict = predict_output(x_star)
l2error = np.linalg.norm(u_star-u_star_predict,2)/np.linalg.norm(u_star,2)
print('Error: %e' %(l2error))
plt.figure()
plt.scatter(u_star, u_star_predict)
plt.show()