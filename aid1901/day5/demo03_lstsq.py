# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_lstsq.py 线性拟合
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
ax.xaxis.set_minor_locator(md.DayLocator())
# 日期数据类型转换  更适合绘图
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices,
    linewidth=2, linestyle='--',
    color='dodgerblue', label='AAPL',
    alpha=0.3)

# 计算每天的趋势价格
trend_prices = (highest_prices + \
    lowest_prices + closing_prices)/3
# 绘制每天的趋势点
mp.scatter(dates, trend_prices, s=80, 
    c='orangered', marker='o', 
    label='Trend Points')
# 线性拟合，绘制趋势线 参数：A, B
days = dates.astype('M8[D]').astype('int32')
print(days)
# 把一组x坐标与一组1并在一起，构建A矩阵
A = np.column_stack((days, np.ones_like(days)))
# print(A)
B = trend_prices
print(B)
kb = np.linalg.lstsq(A, B)[0]
print(kb)
# 绘制趋势线
y = kb[0] * days + kb[1]
mp.plot(dates, y, c='orangered',
    linewidth=3, label='Trend Line')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



