import numpy as np
import matplotlib.pyplot as mp

n = 1000
x,y = np.meshgrid(np.linspace(-3,3,n),np.linspace(-3,3,n))
z = (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
#绘制等高线
mp.figure('Coutour', facecolor='lightgray')
mp.title('Coutour', fontsize=18)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
cntr = mp.contour(x,y,z,30,colors='black',linewidths=0.5)
#为等高线图添加高度标签
mp.clabel(cntr,fmt='%.1f',fontsize=8,inline_spacing=1)
#填充等高线图
mp.contourf(x,y,z,30,cmap='jet')
mp.show()