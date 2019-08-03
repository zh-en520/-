"""
demo01_sin.py   测试傅里叶定理
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.zeros(1000)
n = 1000
for i in range(1, n+1):
	y += 4/((2*i-1)*np.pi)*np.sin((2*i-1)*x)

mp.plot(x, y, label='n=4')
mp.legend()
mp.show()
