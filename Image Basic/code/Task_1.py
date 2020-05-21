# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:47:03 2020

@author: Fengdan Cui
"""

import numpy as np

a = np.array([[2, 4, 5],[5, 2, 200]])
print(a)

b = a[0,:] 
print(b)

f = np.random.randn(500, 1) 
print(f.shape)

g = f[f < 0] 
print(g)

x = np.zeros((1, 100)) + 0.35
print(x)

y = 0.6 * np.ones([1, len(x.transpose())])  
print(y)
 
z = x - y 
print(z)

a = np.linspace(1, 50) 
print(a)

b = a[: :-1] 
print(b)

b[b <= 50] = 0
print(b)
