"""
demo04_contour.py 等高线图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
# 构建网格点坐标矩阵
x, y = np.meshgrid(np.linspace(-3,3,n),
				   np.linspace(-3,3,n))
# 根据每个坐标点的x与y计算高度值z
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制等高线
mp.figure('Coutour', facecolor='lightgray')
mp.title('Coutour', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
cntr=mp.contour(x, y, z, 8, colors='black',
	linewidths=0.5)
# 为每个等高线绘制高度标签
mp.clabel(cntr, inline_spacing=1, 
	fmt='%.1f', fontsize=10)
# 为等高线区间填充颜色
mp.contourf(x, y, z, 8, cmap='jet')
mp.show()