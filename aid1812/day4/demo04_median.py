"""
demo02_mean.py 均值
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

print(dates, dates.dtype)
#绘制收盘价的折线图

mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Price', fontsize=12)
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

mp.plot(dates, closing_prices, 
	c='dodgerblue', linestyle='--',
	linewidth=2, label='AAPL', alpha=0.5)

# 计算均值
mean = np.mean(closing_prices)
mean = closing_prices.mean()
print(mean)
mp.hlines(mean, dates[0], dates[-1], 
	color='orangered', linewidth=2, 
	label='Mean(closing_prices)')
# 计算加权平均价格
times = np.arange(1, 31)
wmean = np.average(closing_prices, 
		weights=times)
mp.hlines(wmean, dates[0], dates[-1], 
	color='green', linewidth=2, label='TWAP')
# 计算交易量加权平均价格
vwap = np.average(closing_prices, 
		weights=volumns)
mp.hlines(vwap, dates[0], dates[-1], 
	color='violet', linewidth=2, label='VWAP')

median = np.median(closing_prices)
mp.hlines(median, dates[0], dates[-1], 
	color='gold', linewidth=2, label='median')

std = np.std(closing_prices, ddof=1)
print(std)


mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
