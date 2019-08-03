"""
demo08_stack.py  数组的组合与拆分
"""
import numpy as np

a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(2, 3)
print(a)
print(b)
# 垂直方向
c = np.vstack((a, b))
print(c)
a, b = np.vsplit(c, 2)
print(a, b, sep='\n')

# 水平方向
c = np.hstack((a, b))
print(c)
a, b = np.hsplit(c, 2)
print(a, b, sep='\n')

# 深度方向
c = np.dstack((a, b))
print(c)
a, b = np.dsplit(c, 2)
print(a, b, sep='\n')

# 简单一位数组的组合方案
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]
print(np.row_stack((x, y)))
print(np.column_stack((x, y)))






