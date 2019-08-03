# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_sort.py 排序
"""
import numpy as np

prods = np.array(['Apple', 'Huawei', 'Mi',
                  'Oppo', 'Vivo'])
prices = [8000, 4999, 2999, 3999, 3999]
volumns = np.array([40, 80, 50, 35, 40])

indices = np.lexsort((-volumns, prices))
print(indices)

# 插入排序
a = np.array([1, 2, 3, 6, 9])
b = np.array([5, 8])
indices = np.searchsorted(a, b)
print(indices)
# 把b元素按照indices的索引位置，插入a数组
d = np.insert(a, indices, b)
print(d)
