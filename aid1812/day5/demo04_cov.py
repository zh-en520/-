"""
demo04_cov.py 协方差
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
mp.figure('COV', facecolor='lightgray')
mp.title('COV', fontsize=14)
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
mp.plot(dates, vale_closing_prices, 
	c='dodgerblue', linestyle='-',
	linewidth=2, label='VALE')

mp.plot(dates, bhp_closing_prices, 
	c='orangered', linestyle='-',
	linewidth=2, label='BHP')

# 计算两只股票收盘价的协方差
vale_mean = np.mean(vale_closing_prices)
bhp_mean = np.mean(bhp_closing_prices)
vale_dev = vale_closing_prices-vale_mean
bhp_dev = bhp_closing_prices-bhp_mean
cov = (vale_dev * bhp_dev).sum() / (vale_dev.size-1)
print(cov)

# 相关性系数
k = cov / (np.std(vale_closing_prices) * \
	np.std(bhp_closing_prices))
print(k)

# 相关性矩阵
m = np.corrcoef(vale_closing_prices, 
		bhp_closing_prices)
print(m)

# 协方差矩阵
m = np.cov(vale_closing_prices, 
		bhp_closing_prices)
print(m)

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()


