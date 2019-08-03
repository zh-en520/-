"""
demo11_integrate.py  积分
"""
import numpy as np
import scipy.integrate as si

def f(x):
	return 2 * x**2 + 3*x + 4

val = si.quad(f, -5, 5)
print(val)
# val[0]：积分值    val[1]：积分误差
