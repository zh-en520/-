# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_profit.py  数据平滑
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt

def dmy2ymd(dmy):
    # 把dmy格式的字符串转成ymd格式字符串
    dmy = str(dmy, encoding='utf-8')
    d = dt.datetime.strptime(dmy, '%d-%m-%Y')
    d = d.date()
    ymd = d.strftime('%Y-%m-%d')
    return ymd

dates, bhp_closing_prices = \
    np.loadtxt('../da_data/bhp.csv',
    delimiter=',', usecols=(1, 6),
    unpack=True, dtype='M8[D], f8',
    converters={1:dmy2ymd})

vale_closing_prices = \
    np.loadtxt('../da_data/vale.csv',
    delimiter=',', usecols=(6,),
    unpack=True)


# 绘制收盘价折线图
mp.figure('Profits', facecolor='lightgray')
mp.title('Profits', fontsize=18)
mp.xlabel('date', fontsize=14)
mp.ylabel('Profits', fontsize=14)
mp.tick_params(labelsize=8)
mp.grid(linestyle=':')
# 设置x轴的刻度定位器，使之更适合显示日期数据
ax = mp.gca()
import matplotlib.dates as md
# 主刻度  每周一为主刻度
ma_loc = md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(ma_loc)
ax.xaxis.set_major_formatter(
    md.DateFormatter('%Y-%m-%d'))
# 次刻度  每天一个刻度
ax.xaxis.set_minor_locator(md.DayLocator())
# 日期数据类型转换  更适合绘图
dates = dates.astype(md.datetime.datetime)

# 计算两只股票的收益率
print(bhp_closing_prices.shape)
print(np.diff(bhp_closing_prices))
print(np.diff(bhp_closing_prices).shape)
bhp_returns = np.diff(bhp_closing_prices) \
    / bhp_closing_prices[:-1]
vale_returns = np.diff(vale_closing_prices) \
    / vale_closing_prices[:-1]
# 绘制两支股票的收益率曲线
mp.plot(dates[:-1], bhp_returns, 
    color='dodgerblue', label='bhp returns',
    alpha=0.2)
mp.plot(dates[:-1], vale_returns, 
    color='orangered', label='vale returns',
    alpha=0.2)
# 使用卷积 对两组数据执行降噪
kernel = np.hanning(8)
kernel /= kernel.sum()
bhp_returns_convolved = np.convolve(
    bhp_returns, kernel, 'valid')
vale_returns_convolved = np.convolve(
    vale_returns, kernel, 'valid')
print(bhp_returns_convolved.shape)

mp.plot(dates[7:-1], bhp_returns_convolved,
    color='dodgerblue', label='bhp_convolved1')
mp.plot(dates[7:-1], vale_returns_convolved,
    color='orangered', label='vale_convolved1')

# 对两条曲线做多项式拟合
days=dates[7:-1].astype('M8[D]').astype('i4')
p1=np.polyfit(days,bhp_returns_convolved,3)
p2=np.polyfit(days,vale_returns_convolved,3)
# 绘制曲线
y1=np.polyval(p1, days)
y2=np.polyval(p2, days)
mp.plot(dates[7:-1], y1,
    color='dodgerblue', label='bhp_convolved')
mp.plot(dates[7:-1], y2,
    color='orangered', label='vale_convolved')
# 求曲线的交点坐标
p = np.polysub(p1, p2) # 求差函数
# 求根
xs = np.roots(p)
result = xs.astype('i4').astype('M8[D]')
print(result)


mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



