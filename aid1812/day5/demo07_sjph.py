"""
demo07_sjph.py  数据平滑
1. 计算两只股票的收益率曲线 并绘制
2. 分析曲线形状，确定投资策略
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

# 计算收益率
bhp_returns = np.diff(bhp_closing_prices) \
		/ bhp_closing_prices[:-1]
vale_returns = np.diff(vale_closing_prices) \
		/ vale_closing_prices[:-1]
dates = dates[:-1]
mp.plot(dates, bhp_returns, c='dodgerblue',
	label='bhp_returns', alpha=0.3)
mp.plot(dates, vale_returns, c='orangered',
	label='vale_returns', alpha=0.3)

# 卷积降噪
kernel = np.hanning(8)
kernel /= kernel.sum()
print(kernel)
bhp=np.convolve(bhp_returns,kernel,'valid')
vale=np.convolve(vale_returns,kernel,'valid')
# mp.plot(dates[7:], vale, c='orangered',
# 	label='vale_convolved')
# mp.plot(dates[7:], bhp, c='dodgerblue',
# 	label='bhp_convolved')

# 针对vale与bhp分别做多项式拟合
days = dates[7:].astype('M8[D]').astype('int32')
P_vale = np.polyfit(days, vale, 3)
P_bhp = np.polyfit(days, bhp, 3)
y_vale = np.polyval(P_vale, days)
y_bhp =  np.polyval(P_bhp, days)
mp.plot(dates[7:], y_vale, c='orangered',
	label='vale_convolved')
mp.plot(dates[7:], y_bhp, c='dodgerblue',
	label='bhp_convolved')

# 求两个多项式的交点位置
P = np.polysub(P_vale, P_bhp)
xs = np.roots(P)
dates = np.floor(xs).astype('M8[D]')
print(dates)
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
