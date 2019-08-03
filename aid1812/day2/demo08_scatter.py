"""
demo08_scatter.py  散点图
"""
import numpy as np
import matplotlib.pyplot as mp

# 随机生成身高与体重
n = 300
x = np.random.normal(173, 10, n)
y = np.random.normal(65, 5, n)

mp.figure('Persons', facecolor='lightgray')
mp.title('Person Points', fontsize=16)
mp.xlabel('Height', fontsize=12)
mp.ylabel('Weight', fontsize=12)
mp.grid(linestyle=':')

d = (x-173)**2 + (y-65)**2
mp.scatter(x, y, c=d, cmap='jet')
mp.show()