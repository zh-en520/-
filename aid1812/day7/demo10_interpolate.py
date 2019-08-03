"""
demo10_interpolate.py 插值器
"""
import scipy.interpolate as si
import matplotlib.pyplot as mp
import numpy as np

# 搞一组散点
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 15)
dis_y = np.sinc(dis_x)
mp.scatter(dis_x, dis_y, s=60, marker='o',
	label='Points', c='red')

#通过散点设计出符合线性规律的插值器函数
#返回的linear是一个函数  可以：linear(x)
linear=si.interp1d(dis_x, dis_y, 'linear')
x = np.linspace(min_x, max_x, 1000)
y = linear(x)
mp.plot(x, y, c='dodgerblue', label='linear')


# 三次样条插值器   cubic
cubic=si.interp1d(dis_x, dis_y, 'cubic')
x = np.linspace(min_x, max_x, 1000)
y = cubic(x)
mp.plot(x, y, c='orangered', label='cubic')

mp.legend()
mp.show()






