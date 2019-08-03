# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_vectorize.py  矢量化
"""
import math as m
import numpy as np

def f(a, b):
    return m.sqrt(a**2 + b**2)
print(f(3, 4))

a = np.array([3, 4, 5, 6])
b = np.array([4, 5, 6, 7])
# 函数矢量化 改造f函数，使之可以处理矢量数据
# print(f(a, b))
f_vec = np.vectorize(f)  # 返回矢量函数
print(f_vec(a, b))
print(np.vectorize(f)(a, 5))

# 使用frompyfunc函数实现函数矢量化
# 2：f函数接收2个参数
# 1：f函数有1个返回值
f_fpf = np.frompyfunc(f, 2, 1)
print(f_fpf(a, b))







