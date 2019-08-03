"""
demo02_plot.py  基本绘图API
"""
import numpy as np
import matplotlib.pyplot as mp

# 绘制正弦曲线  y=sin(x)
x = np.linspace(-np.pi, np.pi, 1000)
sinx = np.sin(x)
# 绘制余弦曲线  y=1/2 * cos(x)
cosx = np.cos(x) / 2
# 设置 可视区域
# mp.xlim(0, np.pi)
# mp.ylim(0, 1)

# 设置坐标刻度
vals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
texts = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', 
		  r'$\frac{\pi}{2}$', r'$\pi$']
mp.xticks(vals, texts)
mp.yticks([-1, -0.5, 0, 0.5, 1], 
		  ['-1.0', '-0.5', '0.0', '0.5', '1.0'])

# 设置坐标轴
ax = mp.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

mp.plot(x, sinx, linestyle='--', linewidth=2, 
	color='dodgerblue', alpha=0.8, 
	label=r'$y=sin(x)$')
mp.plot(x, cosx, linestyle=':', linewidth=3, 
	color='orangered', alpha=0.8, 
	label=r'$y=\frac{1}{2}cos(x)$')

# 绘制特殊点
px = [np.pi/2, np.pi/2]
py = [1, 0]
mp.scatter(px, py, s=100, marker='o',
	edgecolor='steelblue', zorder=3,
	facecolor='deepskyblue')
# 添加备注
mp.annotate(r'$[\frac{\pi}{2}, 1]$',
	xycoords='data',
	xy=(np.pi/2, 1),
	textcoords='offset points',
	xytext=(30, 10),
	fontsize=12,
	arrowprops=dict(
		arrowstyle='-|>',
		connectionstyle='angle3'
	)
)

mp.annotate(r'$[\frac{\pi}{2}, 0]$',
	xycoords='data',
	xy=(np.pi/2, 0),
	textcoords='offset points',
	xytext=(-60, -40),
	fontsize=12,
	arrowprops=dict(
		arrowstyle='-|>',
		connectionstyle='angle3'
	)
)


mp.legend(loc=2)
mp.show()
