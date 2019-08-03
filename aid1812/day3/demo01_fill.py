"""
demo01_fill.py 填充
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
x = np.linspace(0, 8*np.pi, n)
sinx = np.sin(x)
cosx = np.cos(x/2) / 2 

mp.figure('Fill', facecolor='lightgray')
mp.title('Fill', fontsize=14)
mp.xlabel('X', fontsize=12)
mp.ylabel('Y', fontsize=12)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
mp.plot(x, sinx, c='dodgerblue', 
	label=r'$y=sin(x)$')
mp.plot(x, cosx, c='orangered', 
	label=r'$y=\frac{1}{2}cos(\frac{x}{2})$')

mp.fill_between(x, sinx, cosx, sinx>cosx,
	color='dodgerblue', alpha=0.5)
mp.fill_between(x, sinx, cosx, sinx<cosx,
	color='orangered', alpha=0.5)


mp.legend()
mp.show()

