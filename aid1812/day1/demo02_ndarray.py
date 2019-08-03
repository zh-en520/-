"""
demo02_ndarray.py  ndarray示例
"""
import numpy as np

a = np.array([1,2,3,4,5,6])
b = np.arange(7, 13, 1)
print(a)
print(b)
c = np.zeros(6)
d = np.ones(6)
print(c)
print(d / 5)
e = np.array([[1, 2, 3], [4, 5, 6]])
print(e)
print(np.ones_like(e))




