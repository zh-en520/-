# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_add.py  通用运算符
"""
import numpy as np
a = np.arange(1, 7)
print(a)
print(np.add(a, a))
print(np.add.reduce(a))
print(np.add.accumulate(a))
print(np.prod(a))
print(np.cumprod(a))

# 外和
print(np.add.outer([10,20,30], a))
# 外积
print(np.outer([10,20,30], a))

# 测试除法与取整通用函数
a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])

print(np.divide(a, b))
print(np.floor_divide(a, b))
print(np.ceil(a/b))
print(np.round(a/b))
print(np.trunc(a/b))

