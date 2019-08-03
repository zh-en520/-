# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_poly.py 多项式操作
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-20, 20, 1000)
y = 4*x**3 + 3*x**2 - 1000*x + 1

# 求多项式函数的导函数
P = [4, 3, -1000, 1]
Q = np.polyder(P)
print('Q:',Q)
xs = np.roots(Q)
print('xs:',xs)
# 把x坐标带入原函数，求得函数值
ys = np.polyval(P, xs)
print('ys:',ys)
# 绘制
mp.plot(x, y)
mp.scatter(xs, ys, marker='D',
    color='red', s=60, zorder=3)
mp.show()


