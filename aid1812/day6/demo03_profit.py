"""
demo03_profit.py  
定义一种买入卖出策略 验证是否有效
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
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

print(dates, dates.dtype)
#绘制收盘价的折线图

mp.figure('Profits', facecolor='lightgray')
mp.title('Profits', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Profit', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
# 设置刻度定位器
ax = mp.gca()
#设置主刻度定位器-每周一一个主刻度
major_loc=md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
# 设置次刻度定位器为日定位器
minor_loc=md.DayLocator()
ax.xaxis.set_minor_locator(minor_loc)

def profit(opening_price, highest_price,
		lowest_price, closing_price):
	'''
	定义一种投资策略 
	开盘价*0.99倍买入， 收盘价卖出
	'''
	buying_price = opening_price*0.99
	if lowest_price<=buying_price<=highest_price:
		return (closing_price-buying_price)\
				/ buying_price

	return np.nan  # 无效值

# 矢量化profit函数，求得每天的收益率
profits=np.vectorize(profit)(opening_prices,
	highest_prices, lowest_prices,
	closing_prices)
print(profits)
# 判断profits中每个元素是否是nan
nan = np.isnan(profits)
dates, profits = dates[~nan], profits[~nan]

dates = dates.astype(md.datetime.datetime)
mp.plot(dates, profits, 'o-', c='dodgerblue',
	linewidth=2, label='Profits')
m = np.mean(profits)
mp.hlines(m, dates[0], dates[-1], 
	color='orangered', label='Mean(Profits)')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
