# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_piecewise.py  数组处理函数
"""
import numpy as np

a = np.array([50, 65, 78, 82, 99])
# 数组处理，分类分数级别
r = np.piecewise(a, 
    [a>=85, (a>=60) & (a<85), a<60], 
    [1, 2, 3])
print(r)