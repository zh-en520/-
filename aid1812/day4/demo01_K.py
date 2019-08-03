"""
demo01_K.py 加载文件
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

# 绘制蜡烛图
# 控制颜色
rise = closing_prices >= opening_prices
color = np.array(
		[('white' if x else 'green') \
		for x in rise])
edgecolor = np.array(
		[('red' if x else 'green') \
		for x in rise])
# 绘制实体
mp.bar(dates, closing_prices-opening_prices,
	0.8, opening_prices, color=color,
	edgecolor=edgecolor, zorder=3)
# 绘制影线
mp.vlines(dates, lowest_prices, 
	highest_prices, color=edgecolor)

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
