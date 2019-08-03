"""
demo02_bar.py 柱状图
"""
import numpy as np
import matplotlib.pyplot as mp

# 整理苹果12个月销量
apples=[92,34,75,32,96,52,36,10,23,41,22,35]
oranges=[23,43,46,58,23,74,56,72,38,95,63,9]
x = np.arange(len(apples))
mp.figure('Bar Chart', facecolor='lightgray')
mp.title('Bar Chart', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Volumn', fontsize=12)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
mp.bar(x-0.2, apples, 0.4, color='limegreen',
	label='Apples')
mp.bar(x+0.2, oranges, 0.4, color='orangered',
	label='Oranges')
# 修改x刻度文本
mp.xticks(x, ['Jan', 'Feb', 'Mar', 'Apr',
	'May', 'Jun', 'Jul', 'Aug', 'Sep',
	'Oct', 'Nov', 'Dec'])

mp.legend()
mp.show()

