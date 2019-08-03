# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_bd.py  布林带
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

dates, opening_prices, highest_prices, \
    lowest_prices, closing_prices = \
    np.loadtxt('../da_data/aapl.csv',
    delimiter=',', usecols=(1, 3, 4, 5, 6),
    unpack=True, 
    dtype='M8[D], f8, f8, f8, f8',
    converters={1:dmy2ymd})

# 绘制收盘价折线图
mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=18)
mp.xlabel('date', fontsize=14)
mp.ylabel('Closing Price', fontsize=14)
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
yy = md.DayLocator()
ax.xaxis.set_minor_locator(md.DayLocator())
# 日期数据类型转换  更适合绘图
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices,
    linewidth=2, linestyle='--',
    color='dodgerblue', label='AAPL',
    alpha=0.5)

# 5日加权均线
weights = np.exp(np.linspace(-1, 0, 5))
weights = weights[::-1]
weights /= weights.sum()
sma53 = np.convolve(
    closing_prices, weights, 'valid')
mp.plot(dates[4:], sma53, color='orangered',
    label='SMA53')

# 绘制布林带的上轨与下轨
stds = np.zeros(sma53.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i+5].std()
lower = sma53 - 2 * stds
upper = sma53 + 2 * stds
mp.plot(dates[4:], upper, color='orangered',
    label='Upper')
mp.plot(dates[4:], lower, color='orangered',
    label='Lower')
# 填充
mp.fill_between(dates[4:], lower, upper,
    upper > lower, color='orangered', 
    alpha=0.3)


mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



