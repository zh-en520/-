# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_inter.py  插值
"""
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as mp
# 造一些散点数据
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 15)
dis_y = np.sinc(dis_x)
print(dis_y)
# 绘图
mp.scatter(dis_x, dis_y, c='orangered',
    s=60, marker='o')

# 基于这些离散数据，使用插值获得连续函数
linear=si.interp1d(dis_x, dis_y, kind='linear')
# 绘制linear函数图像
x = np.linspace(min_x, max_x, 1000)
y = linear(x)
mp.plot(x, y)

# 三次样条插值器
cubic=si.interp1d(dis_x, dis_y, kind='cubic')
# 绘制linear函数图像
y = cubic(x)
mp.plot(x, y)

mp.show()