#散点图
import matplotlib.pyplot as mp
import numpy as np

n = 300
x = np.random.normal(175,5,n)
y = np.random.normal(65,5,n)

#画图
mp.figure('Persons',facecolor='lightgray')
mp.title('Persons',fontsize=16)
mp.xlabel('Height',fontsize=12)
mp.ylabel('Weight',fontsize=12)
d = (x-175)**2 + (y-65)**2
mp.scatter(x,y,s=60,alpha=0.8,c=d,cmap='jet_r',marker='*',label='Persons')
mp.legend()
mp.show()