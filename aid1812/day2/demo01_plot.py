"""
demo01_plot.py  基本绘图API
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([34,12,34,12,3,34])
mp.plot(x, y)
mp.hlines(15, 1, 6)
mp.vlines([3, 4], 5, 35)
mp.show()