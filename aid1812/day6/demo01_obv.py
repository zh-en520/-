"""
demo01_obv.py  OBV能量潮
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
	volumns = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6,7),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

print(dates, dates.dtype)
#绘制收盘价的折线图

mp.figure('OBV', facecolor='lightgray')
mp.title('OBV', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Volumn', fontsize=12)
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

# 得到从第二天开始每天的涨跌值
diff_prices = np.diff(closing_prices)
# sign_prices = np.sign(diff_prices)
sign_prices = np.piecewise(diff_prices, 
	[diff_prices>0, diff_prices<0, diff_prices==0],
	[1, -1, 0])

print(sign_prices)

# 绘制OBV图
dates = dates.astype(md.datetime.datetime)
dates = dates[1:]
volumns = volumns[1:]
mp.bar(dates[sign_prices==1], 
	   volumns[sign_prices==1], 
	   0.8, color='red')
mp.bar(dates[sign_prices==-1], 
	   volumns[sign_prices==-1], 
	   0.8, color='green')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
