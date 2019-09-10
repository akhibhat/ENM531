#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# In[2]:
# Initialize the number of samples

n = 5000

# In[2]:
# Samples of x

mean = np.array([0,0])
sigma = np.array([[1, 0.9],[0.9,1]])

px1x2 = np.random.multivariate_normal(mean,sigma,n)
x1 = px1x2[:,0]
x2 = px1x2[:,1]

# In[3]:
# Transform x to y

a = 1.15
b = 0.5
c = x1**2

[y1,y2] =  [a*x1, x2/a + b*(a**2 + x1**2)]

y = np.zeros([n,2])

y[:,0] = y1
y[:,1] = y2

# In[4]:
# Set up for plotting

X,Y = np.meshgrid(y1,y2)

X1 = np.reshape(X,(np.square(n),1))
Y1 = np.reshape(Y,(np.square(n),1))

pos = np.zeros([2,np.square(n)])
pos[0,:] = X1.T
pos[1,:] = Y1.T

# In[5]:
# Gaussain estimate

gy = stats.gaussian_kde(y.T)

pY = gy.evaluate(pos)
pYr = np.reshape(pY,(n,n))

# In[6]:
# Plot the gaussian estimate

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X,Y,pYr)
plt.show()

# In[7]:
# Empirical mean of y1 and y2

y1mean = np.mean(y1)
y2mean = np.mean(y2)