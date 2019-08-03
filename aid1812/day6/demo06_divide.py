"""
demo06_divide.py  通用函数
"""
import numpy as np

a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])

print(np.divide(a, b))
print(np.floor_divide(a, b))
print(np.ceil(a / b))
print(np.floor(a / b))
print(np.round(a / b))
print(np.trunc(a / b))  # 截断除



