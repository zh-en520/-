import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma

n = 100
balls = np.zeros(n,dtype=[
    ('position','f4',2),
    ('size','f4',1),
    ('growth','f4',1),
    ('color','f4',4)
])
#初始化,随机生成100个泡泡
balls['position'] = np.random.uniform(0,1,(n,2))
balls['size'] = np.random.uniform(40,70,n)
balls['growth'] = np.random.uniform(10,20,n)
balls['color'] = np.random.uniform(0,1,(n,4))
#画图
mp.figure('Animation',facecolor='lightgray')
mp.title('Animation',fontsize=18)
sc = mp.scatter(
    balls['position'][:,0],
    balls['position'][:,1],
    balls['size'],
    color=balls['color'])
#每隔30毫秒更新每个泡泡的大小

def update(number):
    balls['size'] += balls['growth']
    #每次都让一个泡泡重新随机属性
    index = number % n
    balls[index]['size'] = np.random.uniform(40,70,1)
    balls[index]['position'] = np.random.uniform(0,1,(1,2))
    #重新绘制所有点
    sc.set_sizes(balls['size'])
    sc.set_offsets(balls['position'])

anim = ma.FuncAnimation(mp.gcf(),update,interval=30)
mp.show()