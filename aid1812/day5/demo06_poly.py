"""
demo06_poly.py  拟合差价函数
1. 计算两只股票的差价
2. 利用多项式拟合得到多项式函数的系数数组
3. 把拟合到的多项式函数绘制出来
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

dates, vale_closing_prices = \
	np.loadtxt('../da_data/vale.csv',
		usecols=(1, 6),
		unpack=True,
		dtype='M8[D], f8',
		delimiter=',',
		converters={1:dmy2ymd})

bhp_closing_prices = \
	np.loadtxt('../da_data/bhp.csv',
		usecols=(6,),
		unpack=True,
		delimiter=',')

#绘制收盘价的折线图
mp.figure('PolyFit', facecolor='lightgray')
mp.title('PolyFit', fontsize=14)
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
dates = dates.astype(md.datetime.datetime)

# 计算差价
diff_prices = bhp_closing_prices-vale_closing_prices
mp.plot(dates, diff_prices, alpha=0.5)
# 多项式拟合
days = dates.astype('M8[D]').astype('int32')
P = np.polyfit(days, diff_prices, 10)
# 绘制多项式函数
y = np.polyval(P, days)
mp.plot(dates, y, linestyle='-', 
	linewidth=2, c='orangered', 
	label='polyfit line')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
