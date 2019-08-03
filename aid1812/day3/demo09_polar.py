"""
demo09_polar.py 极坐标系
"""
import matplotlib.pyplot as mp
import numpy as np

mp.figure('Polar',facecolor='lightgray')
mp.title('Polar')
mp.gca(projection='polar')
mp.grid(linestyle=':')
# 绘制曲线
t = np.linspace(0, 4*np.pi, 1000)
r = 0.8 * t
mp.plot(t, r)
#
x = np.linspace(0, 6*np.pi, 1000)
y = 3*np.sin(6*x)
mp.plot(x, y)
mp.show()