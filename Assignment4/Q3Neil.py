# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:51:08 2019

@author: akhil
"""

import tensorflow as tf
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import time


np.random.seed(1234)
tf.set_random_seed(1234)
tfpi = tf.constant(np.pi)

iter = 1000
data = 100          #n
l1 = -1
l2 = 1
l_1=[1,50,50,1]
x1=np.array([[-1],[1]])
u1=np.array([[0],[0]])


x2=2*lhs(1,data)-1
weights = []
biases = []
num_l_1 = len(l_1)

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net( X, weights, biases):
    num_l_1 = len(weights) + 1

    H = 2.0 * (X - l1) / (l2 - l1) - 1.0
    for l in range(0, num_l_1 - 2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def net_f(x):
    u = net_u(x)
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f = u_xx - u + (tfpi**2 +1)*tf.sin(tfpi*x)
    return f

def net_u( x):
    u = neural_net(x, weights, biases)
    return u

for l in range(0, num_l_1 - 1):
    W = xavier_init(size=[l_1[l], l_1[l + 1]])
    b = tf.Variable(tf.zeros([1, l_1[l + 1]], dtype=tf.float32), dtype=tf.float32)
    weights.append(W)
    biases.append(b)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

x1_tf = tf.placeholder(tf.float32, shape=[None, x1.shape[1]])
x2_tf = tf.placeholder(tf.float32, shape=[None, x2.shape[1]])
u1_tf = tf.placeholder(tf.float32, shape=[None, u1.shape[1]])



u1_pred = net_u(x1_tf)
f_pred = net_f(x2_tf)


loss = tf.reduce_mean(tf.square(u1_tf - u1_pred)) + tf.reduce_mean(tf.square(f_pred))

optimizer_Adam = tf.train.AdamOptimizer()

train_op_Adam = optimizer_Adam.minimize(loss)


init = tf.global_variables_initializer()
sess.run(init)

def predict(X_star):
        tf_dict = {x1_tf: X_star}
        u_star = sess.run(u1_pred, tf_dict)
        # f_star = self.sess.run(self.f_pred, tf_dict)
        return u_star

def train(nIter):
    tf_dict = {x1_tf: x1, u1_tf: u1, x2_tf : x2}

    start_time = time.time()
    for it in range(nIter):
        sess.run(train_op_Adam, tf_dict)

        # Print
        if it % 10 == 0:
            elapsed = time.time() - start_time
            loss_value = sess.run(loss, tf_dict)
            print('It: %d, Loss: %.3e, Time: %.2f' %
                  (it, loss_value, elapsed))
            start_time = time.time()




train(iter)
x_star = np.atleast_2d(np.linspace(-1, 1, 100)).T
u = np.sin(np.pi*x_star)
u_hat = predict(x_star)
error_u = np.linalg.norm(u - u_hat, 2) / np.linalg.norm(u_hat, 2)
print('Error u: %e' % (error_u))
plt.figure()
plt.scatter(u, u_hat)
plt.show()