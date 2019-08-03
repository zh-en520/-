"""
demo03_pie.py  饼状图
"""
import matplotlib.pyplot as mp

labels=['Python', 'Javascript', 
		'C++', 'Java', 'PHP']
values=[26, 17, 21, 29, 11]
spaces=[0.05, 0.01, 0.01, 0.01, 0.01]
colors=['dodgerblue', 'orangered', 
	'limegreen', 'violet', 'gold']

mp.figure('Pie Chart', facecolor='lightgray')
mp.title('Pie Chart', fontsize=14)
# 设置等轴比例显示饼状图
mp.axis('equal')
mp.pie(values, spaces, labels, colors,
	'%.2f%%', shadow=True, startangle=45)
mp.legend()
mp.show()



