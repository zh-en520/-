import matplotlib.pyplot as mp
import numpy as np

#整理数据
labels = ['Python',"Js",'C++','Java','PHP']
values = [26,17,21,29,5]
spaces = [0.05,0.01,0.01,0.01,0.01]
colors = ['dodgerblue', 'orangered',
	'limegreen', 'violet', 'gold']

#画图
mp.figure('Pie Chart',facecolor='lightgray')
mp.title('Pie Chart',fontsize=18)
mp.grid(linestyle=':')
mp.axis('equal')#等轴比例
mp.pie(values,spaces,labels,colors,'%.2f%%',shadow=True,startangle=90)

mp.legend()
mp.show()