"""
demo06_locator.py 刻度定位器
"""
import matplotlib.pyplot as mp

locators = ['mp.NullLocator()',
			'mp.MultipleLocator(1)', 
			'mp.MaxNLocator(nbins=4)',
			'mp.AutoLocator()']

mp.figure('Locator', facecolor='lightgray')

for i, locator in enumerate(locators):
	mp.subplot(4, 1, i+1)
	ax = mp.gca()
	# 干掉上左右轴
	ax.spines['top'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	mp.xlim(0, 10)
	mp.ylim(-1, 1)
	ax.spines['bottom'].set_position(('data',0))
	mp.yticks([])
	# 设置主刻度定位器  每隔1就显示一个主刻度
	major_locator = eval(locator)
	ax.xaxis.set_major_locator(major_locator)
	minor_locator = mp.MultipleLocator(0.1)
	ax.xaxis.set_minor_locator(minor_locator)

mp.show()