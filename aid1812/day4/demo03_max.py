"""
demo03_max.py 最值
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2ymd(dmy):
	# 把二进制字符串转为普通字符串
	dmy = str(dmy, encoding='utf-8')
	t = dt.datetime.strptime(dmy, '%d-%m-%Y')
	s = t.date().strftime('%Y-%m-%d')
	return s

dates, opening_prices, highest_prices, \
	lowest_prices, closing_prices, \
	volumns = np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6,7),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

# 评估AAPL股价的波动性
max_price = np.max(highest_prices)
min_price = np.min(lowest_prices)
print(max_price, '~', min_price)

max_date = dates[np.argmax(highest_prices)]
min_date = dates[np.argmin(lowest_prices)]
print(max_date)
print(min_date)

a = np.arange(1, 10).reshape(3, 3)
b = a.ravel()[::-1].reshape(3, 3)
print(a)
print(b)
print(np.maximum(a, b))
print(np.minimum(a, b))
