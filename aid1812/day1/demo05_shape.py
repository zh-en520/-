"""
demo05_shape.py
"""
import numpy as np

a = np.arange(1, 7)
print(a, a.shape)

# 测试视图变维 reshape()  ravel()
b = a.reshape(2, 3)
print(b, b.shape)
a[0] = 999
print(b, b.shape)
c = b.ravel()
print(c, c.shape)

# 复制变维  flatten()   copy()
d = b.flatten()  # 扁平化
print(b, b.shape)
print(d, d.shape)
d[0] = 1
print(b, b.shape)
print(d, d.shape)

# 就地变维
print(b, b.shape)
b.resize(3, 2)
print(b, b.shape)
