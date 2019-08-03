# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_bitwise.py 位运算通用函数
"""
import numpy as np
a = np.array([0, -1, 2, 3, 4, -5])
b = np.array([1, 1, 2, 3, 4, 5])
c = a ^ b
print(c)
print(np.where(c < 0))
