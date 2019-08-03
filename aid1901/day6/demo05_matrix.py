# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_matrix.py  矩阵
"""
import numpy as np

ary = np.arange(1, 10).reshape(3, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=False)
print(m, type(m))
ary[0,0] = 999
print(m, type(m))

m2 = np.mat('1 2 3; 5 6 4')
print(m2)

# 测试矩阵的乘法
print('-'*45)
print(m)
print(m * m)

# 测试矩阵的逆矩阵
b = np.mat('4 6 7; 2 3 7; 8 4 2')
print(b)
print(np.linalg.inv(b))
print(b * b.I)

c = np.mat('4 6 7; 2 3 7')
print(c)
print(c.I)
print(c * c.I)

# 解应用题
print('='*45)
prices = np.mat('3 3.2; 3.5 3.6')
totals = np.mat('118.4; 135.2')
#解法１
x = np.linalg.lstsq(prices, totals)[0]
#解法２
np.linalg.solve(prices,totals)

print('x:',x)

persons = prices.I * totals
print('prices:',prices)
print('prices.I:',prices.I)
print(persons)
print('='*45)

# 测试
m = np.mat('1 1; 1 0')
print(m**20)


