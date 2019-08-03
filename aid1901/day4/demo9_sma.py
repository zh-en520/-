"""
demo07_sma.py 移动平均线
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
	np.loadtxt('../../datascience_ziliao/素材/da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8',
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

# 绘制5日均线
sma5 = np.zeros(closing_prices.size-4)
for i in range(sma5.size):
	sma5[i] = closing_prices[i:i+5].mean()
mp.plot(dates[4:], sma5, c='orangered',
	label='SMA(5)')

# 基于卷积运算实现5日均线
kernel = np.ones(5) / 5
sma52 = np.convolve(
	closing_prices, kernel, 'valid')
mp.plot(dates[4:], sma52, c='orange',
	label='SMA(5-2)', linewidth=7, 
	alpha=0.5)

# 基于卷积运算实现10日均线
kernel = np.ones(10) / 10
sma10 = np.convolve(
	closing_prices, kernel, 'valid')
mp.plot(dates[9:], sma10, c='red',
	label='SMA(10)')

# 5日加权卷积均线
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
print(weights)
sma53 = np.convolve(closing_prices, 
	weights[::-1], 'valid')
mp.plot(dates[4:], sma53, c='violet',
	label='SMA(5-3)', linewidth=2)

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
