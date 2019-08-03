# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_profit.py 计算无脑策略的收益
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
mp.figure('Profit', facecolor='lightgray')
mp.title('Profit', fontsize=18)
mp.xlabel('date', fontsize=14)
mp.ylabel('Profit', fontsize=14)
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


def profit(opening_price, highest_price,
    lowest_price, closing_price):
    # 定义一种买入卖出策略
    buying_price = opening_price * 0.99
    if lowest_price <= buying_price <= \
       highest_price:
        return (closing_price - buying_price)\
        / buying_price * 100
    return np.nan

# 把profit函数矢量化，求得30天的交易数据
profits=np.vectorize(profit)(opening_prices,
    highest_prices, lowest_prices, 
    closing_prices)

# 使用掩码 获取所有交易数据
nan = np.isnan(profits)
dates, profits = dates[~nan], profits[~nan]
mp.plot(dates, profits, 'o-', label='Profits')
mean = np.mean(profits)
mp.hlines(mean, dates[0], dates[-1],
    color='orangered')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



