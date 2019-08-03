"""
demo05_imshow.py 热成像图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
# 构建网格点坐标矩阵
x, y = np.meshgrid(np.linspace(-3,3,n),
				   np.linspace(-3,3,n))
# 根据每个坐标点的x与y计算高度值z
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制热成像图
mp.figure('Imshow', facecolor='lightgray')
mp.title('Imshow', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.imshow(z, cmap='jet', origin='lower')
mp.colorbar()
mp.show()