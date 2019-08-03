"""
demo05_tp.py 时间数据处理
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2wdays(dmy):
	# 把二进制字符串转为普通字符串
	dmy = str(dmy, encoding='utf-8')
	t = dt.datetime.strptime(dmy, '%d-%m-%Y')
	wday = t.date().weekday()
	return wday

wdays, opening_prices, highest_prices, \
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2wdays})

ave_closing_prices = np.zeros(5)
for wday in range(ave_closing_prices.size):
	ave_closing_prices[wday] = \
	    closing_prices[wdays==wday].mean()

for wday, ave_closing_price in zip(
	['MON', 'TUE', 'WED', 'THU', 'FRI'],
	ave_closing_prices):
	print(wday, ave_closing_price)









