#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Import all required libraries

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# In[2]:
# No. of samples

n = 100;

# In[3]:
# For p(x,y)

mean = np.array([0,2])
sigma = np.array([[0.3,-1],[-1,5]])
pxy = np.random.multivariate_normal(mean,sigma,n)
x = pxy[:,0]
y = pxy[:,1]
X,Y = np.meshgrid(x,y)
X1 = np.reshape(X,(np.square(n),1))
Y1 = np.reshape(Y,(np.square(n),1))
posk = np.zeros([2,np.square(n)])
posk[0,:] = X1.T
posk[1,:] = Y1.T

# In[4]:
# Scatter plot of Gaussian estimate for p(x,y)

Zxy = stats.gaussian_kde(pxy.T)
Gxy = Zxy.evaluate(posk)
GxyR = np.reshape(Gxy,(n,n))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X1,Y1,GxyR)
plt.show()

# In[5]:
# Scatter plot of Analytical solution for p(x,y)

pos = np.empty(X.shape+(2,))
pos[:,:,0] = X
pos[:,:,1] = Y
ana = stats.multivariate_normal(mean,sigma)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X,Y,ana.pdf(pos))
plt.show()

# In[6]:
# For p(x)

meanx = np.array([0])
sigmax = np.array([0.3])
px = np.random.normal(meanx,np.sqrt(sigmax),n)

# In[7]:
# Plot of gaussian estimate for p(x)

Zx = stats.gaussian_kde(px.T)
Gx = Zx.evaluate(px.T)
plt.plot(px,Gx,'.')
plt.show()

# In[8]:
# Plot of analytical solution for p(x)

marx = stats.multivariate_normal(meanx,sigmax)
mx = marx.pdf(px)
plt.plot(px,mx,'.')
plt.show()

# In[9]:
# For p(x|y=-1)

meanxy1 = np.array([0.6])
sigmaxy1 = np.array([0.1])
pxy1 = np.random.normal(meanxy1,np.sqrt(sigmaxy1),n)

# In[10]:
# Plot of gaussian estimate for p(x|y=-1)

Zxy1 = stats.gaussian_kde(pxy1)
Gxy1 = Zxy1.evaluate(pxy1)
plt.plot(pxy1,Gxy1,'.')
plt.show()

# In[11]:
# Plot of analytical solution for p(x|y=-1)

marxy1 = stats.multivariate_normal(meanxy1,sigmaxy1)
mxy1 = marxy1.pdf(pxy1)
plt.plot(pxy1,mxy1,'.')
plt.show()