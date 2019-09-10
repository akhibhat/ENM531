# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:16:29 2019

@author: akhil
"""

import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from NeuralNetsTF import TFNets as tfnn
import tensorflow as tf
import timeit

Nu = 2
Nf = 498
N = Nu + Nf
lam = 1

layers = 2
neurons = 50
indim = 1
outdim = 1

pi = tf.constant(np.pi, dtype=tf.float64)

x1 = tf.Variable(lhs(1,Nf))
x2 = tf.Variable([[-1],[1]], dtype = tf.float64)

x = tf.concat((x1,x2),axis = 0)

fx = -(tf.pow(pi,2) + lam)*tf.sin(pi*x)
fx_exact = fxout = tf.slice(fx,[0,0],[498, fx.get_shape()[1]])
u_exact = tf.gather(fx,[498,499])

nnets = tfnn(N,layers,indim,neurons,outdim)

w1, w_end, b1, b_end = nnets.gen_flparameters()
w2,b2 = nnets.gen_hiddenparameters()

H1 = tf.tanh(tf.add(tf.matmul(x, w1), b1))
H2 = tf.tanh(tf.add(tf.matmul(H1,w2), b2))
output = tf.add(tf.matmul(H2,w_end), b_end)


grad_u = tf.gradients(output,x)[0]
ggrad_u = tf.gradients(grad_u,x)[0]

fxnn = ggrad_u - lam*output
fxout = tf.slice(fxnn,[0,0],[498, fxnn.get_shape()[1]]) #use for MSEf (nn part)

u_out = tf.gather(fxnn,[498,499]) #use for MSEu (nn part)


mseudiff = tf.subtract(u_out,u_exact)
msefdiff = tf.subtract(fxout,fx_exact)

init = tf.global_variables_initializer()

mseuloss = tf.scalar_mul(1/Nu,tf.matmul(tf.transpose(mseudiff),mseudiff))
msefloss = tf.scalar_mul(1/Nf,tf.matmul(tf.transpose(msefdiff),msefdiff))

mse = tf.add(mseuloss,msefloss)

train_op = tf.train.AdamOptimizer(1e-3).minimize(mse)
#init_op = tf.global_variables_initializer()

sess = tf.Session()
var = sess.run(init)

with tf.Session() as sess:
    sess.run(init)
    v = sess.run(fxout)
    w = sess.run(fxnn)
    z = sess.run(mseuloss)
    print(mseuloss)  # will show you your variable.
    sess.close()

