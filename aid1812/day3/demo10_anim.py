"""
demo10_anim.py 动画
"""
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma

# 随机生成100个泡泡
n = 100
balls = np.zeros(n, dtype=[
		('position', float, 2),
		('size', float, 1),
		('growth', float, 1),
		('color', float, 4)])

# 初始化泡泡的属性
balls['position']=np.random.uniform(0,1,(n,2))
balls['size']=np.random.uniform(30,70,n)
balls['growth']=np.random.uniform(10,20,n)
balls['color']=np.random.uniform(0,1,(n,4))

mp.figure('Animation', facecolor='lightgray')
mp.title('Animation', fontsize=14)
mp.xticks([])
mp.yticks([])
sc = mp.scatter(balls['position'][:,0], 
		balls['position'][:,1],
		c=balls['color'],
		s=balls['size'])

# 没30毫秒 更新泡泡大小
def update(number):
	balls['size']+=balls['growth']
	ind = number % n
	balls[ind]['size'] = \
			np.random.uniform(30, 70, 1)
	balls[ind]['position'] = \
			np.random.uniform(0, 1, (1,2))
	# 修改属性后重新绘制界面
	sc.set_sizes(balls['size'])
	sc.set_offsets(balls['position'])

anim=ma.FuncAnimation(mp.gcf(), update, 
	interval=30)

mp.show()
