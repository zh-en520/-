import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 500
x = np.random.normal(0,1,n)
y = np.random.normal(0,1,n)
z = np.random.normal(0,1,n)
mp.figure('3D Scatter',facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x',fontsize=12)
ax3d.set_ylabel('y',fontsize=12)
ax3d.set_zlabel('z',fontsize=12)
ax3d.scatter(x,y,z,s=60,marker='o',alpha=0.6)
mp.show()