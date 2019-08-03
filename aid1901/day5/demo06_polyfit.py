# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_polyfit.py  多项式拟合
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
mp.figure('Polyfit', facecolor='lightgray')
mp.title('Polyfit', fontsize=18)
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

# 得到两支股票的差价数据
diff_prices = bhp_closing_prices - vale_closing_prices
mp.plot(dates, diff_prices,
    color='orangered', label='diff')
# 多项式拟合差价数据
days = dates.astype('M8[D]').astype('i4')
P = np.polyfit(days, diff_prices, 4)
pred_prices = np.polyval(P, days)
mp.plot(dates, pred_prices,
    color='dodgerblue', 
    linewidth=5, label='Polyfit Line')


mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



