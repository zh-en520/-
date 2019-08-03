import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 1000
x,y = np.meshgrid(np.linspace(-3,3,n),np.linspace(-3,3,n))
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)

mp.figure('3D Surface', facecolor='lightgray')
mp.tick_params(labelsize=10)
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x',fontsize=12)
ax3d.set_ylabel('y',fontsize=12)
ax3d.set_zlabel('z',fontsize=12)
ax3d.plot_surface(x,y,z,rstride=30,cstride=30,cmap='jet')
mp.show()