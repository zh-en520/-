"""
demo06_3dscatter.py   三维散点图
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 500
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
z = np.random.normal(0, 1, n)

mp.figure('3D Scatter')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x', fontsize=14)
ax3d.set_ylabel('y', fontsize=14)
ax3d.set_zlabel('z', fontsize=14)
mp.tick_params(labelsize=10)
d = x**2 + y**2 + z**2
ax3d.scatter(x, y, z, s=30, c=d,
	alpha=0.7, cmap='jet_r')
mp.show()



