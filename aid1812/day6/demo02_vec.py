"""
demo02_vec.py  函数矢量化
"""
import numpy as np
import math as m

def f (a, b):
	return m.sqrt(a**2 + b**2)

print(f(3, 4))
a = np.array([3,4,5,6])
b = np.array([4,5,6,7])
# print(f(a, b))   报错 
# 矢量化过后的函数  f_vec
f_vec = np.vectorize(f)
print(f_vec(a, b))
print(f_vec(a, 5))

# np.frompyfunc() 也可以实现函数矢量化
f_fpf = np.frompyfunc(f, 2, 1)
print(f_fpf(a, b))

