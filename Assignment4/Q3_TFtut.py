# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:04:01 2019

@author: akhil
"""

import tensorflow as tf
import numpy as np
from pyDOE import lhs

Nu = 2
Nf = 498
N = Nf + Nu
lam = 1

indim = 1
outdim = 1
nodes_hl1 = 50
nodes_hl2 = 50

x = tf.placeholder('float64', [None, 1])
y = tf.placeholder('float64', [None, 1])

pi = tf.constant(np.pi, dtype=tf.float64)

x1 = tf.Variable(lhs(1,Nf))
#x2 = tf.Variable([[-1],[1]], dtype = tf.float64)

#xin = tf.Variable(tf.concat((x1,x2),axis = 0))

fx = tf.Variable(-(tf.pow(pi,2) + lam)*tf.sin(pi*x1))
fx_exact = fxout = tf.slice(fx,[0,0],[498, fx.get_shape()[1]])
u_exact = tf.Variable([[0],[0]])

def neural_network_model(data):
    
    xavier_stddevhl1 = 1. / np.sqrt((indim + nodes_hl1) / 2.)
    xavier_stddevhl2 = 1. / np.sqrt((nodes_hl1 + nodes_hl2) / 2.)
    xavierstddevout = 1. / np.sqrt((nodes_hl2 + outdim) / 2.)
    
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([indim, nodes_hl1], dtype=tf.float64)*xavier_stddevhl1, dtype=tf.float64),
                     'bias': tf.Variable(tf.zeros([1,nodes_hl1], dtype=tf.float64), dtype=tf.float64)}
    
    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2], dtype=tf.float64)*xavier_stddevhl2, dtype=tf.float64),
                     'bias': tf.Variable(tf.zeros([1,nodes_hl2], dtype=tf.float64), dtype=tf.float64)}
    
    outputlayer = {'weights': tf.Variable(tf.random_normal([nodes_hl2, outdim], dtype=tf.float64)*xavierstddevout, dtype=tf.float64),
                     'bias': tf.Variable(tf.zeros([1,outdim], dtype=tf.float64), dtype=tf.float64)}
    
    l1 = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['bias'])
    l1 = tf.nn.tanh(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_layer2['weights']), hidden_layer2['bias'])
    l2 = tf.nn.tanh(l2)
        
    output = tf.matmul(l2,outputlayer['weights']) + outputlayer['bias']
    
    gradu = tf.gradients(output,data)[0]
    ggradu = tf.gradients(gradu,data)[0]
    
    return output, ggradu

#def train_neural_network(x):
prediction, ggradu = neural_network_model(x1)


#fxnn = tf.Variable(tf.squeeze(tf.subtract(ggradu,prediction))[:,None])

prediction = tf.Variable(tf.squeeze(prediction))
ggradu = tf.Variable(tf.squeeze(ggradu))

fxnn = tf.subtract(prediction,ggradu)[:,None]
##    fxout = tf.slice(fxnn,[0,0],[498, fxnn.get_shape()[1]]) #use for MSEf (nn part)
##    
##    u_out = tf.gather(prediction,[498,499]) #use for MSEu (nn part)
##    
##    mseudiff = tf.subtract(u_out,u_exact)
##    msefdiff = tf.subtract(fxout,fx_exact)
##    
##    mseuloss = tf.scalar_mul(1/Nu,tf.matmul(tf.transpose(mseudiff),mseudiff))
##    msefloss = tf.scalar_mul(1/Nf,tf.matmul(tf.transpose(msefdiff),msefdiff))
#
msediff = tf.Variable(tf.subtract(fxnn,fx))
cost = tf.Variable(tf.scalar_mul(1/N,tf.matmul(tf.transpose(msediff),msediff)))
#
##    cost = tf.add(mseuloss,msefloss)
#
##    cost = tf.metrics.mean_squared_error(fx,fxnn)
#predu
optimizer = tf.train.AdamOptimizer().minimize(cost)
#

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    v = sess.run(fxout)
#    w = sess.run(fxnn)
#    z = sess.run(cost)
#    print(cost)  # will show you your variable.

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    loss=0
    for iterations in range(10000):
        batch_x = x1
        batch_y = fx
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                            y: batch_y})
        loss +=c
        print(loss)

#output, ggradu = neural_network_model(xin)
        
train_neural_network(x)