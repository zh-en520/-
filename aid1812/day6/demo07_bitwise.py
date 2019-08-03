"""
demo07_bitwise.py  位运算
"""
import numpy as np

a = np.array([-1, -1, 2, -5, -3])
b = np.array([-6, 1, 2, -5, 3])

c = np.bitwise_xor(a, b)
# where方法用于寻找符合要求的索引序列
print(np.where(c<0)[0])


# 1000以内2的幂
a = np.arange(1, 1000)
mask = np.bitwise_and(a, a-1)==0
print(a[mask])

