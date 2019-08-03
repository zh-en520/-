#绘制正弦曲线
#y = sin(x)
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-np.pi,np.pi,1000)
y = np.sin(x)

#设置坐标轴刻度
mp.xticks([-np.pi,-np.pi/2.0,0,np.pi/2,np.pi])
mp.yticks([-1.0,-0.5,0.5,1.0])

#设置坐标轴
ax = mp.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))

#绘制曲线，设置线型，线宽，颜色
mp.plot(x,y,linestyle='--',linewidth=3,color='dodgerblue',alpha=0.7,label=r'$y=sin(x)$')

y2 = 1/2*np.cos(x)
mp.plot(x,y2,linestyle=':',linewidth=3,color='orangered',alpha=0.7,label=r'$y=\frac{1}{2}cos(x)$')

#绘制特殊点
mp.scatter([np.pi/2,np.pi/2],[0,1],
           marker='o',s=60,facecolor='steelblue',
           edgecolors='red',zorder=3)

#添加备注
mp.annotate(
    r'$[\frac{\pi}{2},1]$',
    xycoords='data',
    xy=(np.pi/2,1),
    textcoords='offset points',
    xytext=(30,50),
    fontsize=10,
    arrowprops=dict(
        arrowstyle='->',
        connectionstyle='angle3'
    )
)

#自动显示图例
mp.legend(loc='upper left')
mp.show()