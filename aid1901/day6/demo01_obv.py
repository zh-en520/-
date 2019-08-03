# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_obv.py 净额成交量
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
    lowest_prices, closing_prices, \
    volumns = \
    np.loadtxt('../da_data/aapl.csv',
    delimiter=',', usecols=(1, 3, 4, 5, 6, 7),
    unpack=True, 
    dtype='M8[D], f8, f8, f8, f8, f8',
    converters={1:dmy2ymd})

# 绘制收盘价折线图
mp.figure('OBV', facecolor='lightgray')
mp.title('OBV', fontsize=18)
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

# 绘制OBV能量潮
# 若相比上一天的收盘价上涨，则为正成交量
# 若相比上一天的收盘价下跌，则为负成交量
diff_prices = np.diff(closing_prices)
sign_prices = np.sign(diff_prices)
obvs = volumns[1:]
color = [('red' if x==1 else 'green') \
         for x in sign_prices]
mp.bar(dates[1:], obvs, 0.8, color=color,
    label='OBV')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



