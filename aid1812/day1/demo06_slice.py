"""
demo06_slice.py 数组索引操作
"""
import numpy as np

a = np.arange(1, 10)
print(a[:3])
print(a[3:6])
print(a[6:])
print(a[::-1])
print(a[:-4:-1])
print(a[::])
print(a[::3])
print(a[1::3])
print(a[2::3])

# 针对高维数组的切片
a = a.reshape(3, 3)
print(a, a.shape)
print(a[:2, :2])
print(a[:2, 0])
