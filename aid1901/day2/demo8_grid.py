#刻度网格线
import matplotlib.pyplot as mp

y = [1,10,100,1000,100,10,1]

mp.figure('GridLine',facecolor='lightgray')
mp.title('GridLine',fontsize=14)
#设置水平和垂直方向的刻度定位器
ax = mp.gca()
ax.xaxis.set_major_locator(mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(mp.MultipleLocator(50))
#设置刻度网格线
ax.grid(which='major',axis='both',linestyle='-',linewidth=0.75,color='orange')
ax.grid(which='minor',axis='both',linestyle='-',linewidth=0.25,color='orange')
mp.plot(y)
mp.show()