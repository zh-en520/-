"""
demo06_apply.py 数据的轴向汇总
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

data = [opening_prices, highest_prices,
		lowest_prices, closing_prices]
data = np.array(data)
print(data.shape)

# 行方向的数据汇总
def rowfunc(row):
	return np.mean(row), np.std(row)

r = np.apply_along_axis(rowfunc, 1, data)
print(np.round(r,2))

# 列方向的数据汇总
def colfunc(col):
	return np.mean(col)

r = np.apply_along_axis(colfunc, 0, data)
print(np.round(r,2))



