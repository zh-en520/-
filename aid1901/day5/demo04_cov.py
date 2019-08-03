# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_cov.py 协方差
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
mp.figure('COV', facecolor='lightgray')
mp.title('COV', fontsize=18)
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

# 绘制两只股票的价格曲线
mp.plot(dates, bhp_closing_prices, 
    color='dodgerblue', label='BHP')
mp.plot(dates, vale_closing_prices, 
    color='orangered', label='VALE')

# 通过计算协方差，输出两支股票的相似情况
bhp_mean = np.mean(bhp_closing_prices)
vale_mean = np.mean(vale_closing_prices)
bhp_dev = bhp_closing_prices - bhp_mean
vale_dev = vale_closing_prices - vale_mean
#cov_ab = np.mean(bhp_dev * vale_dev)
cov_ab = (bhp_dev * vale_dev).sum() / (bhp_dev.size-1)
print(cov_ab)
# 输出两组样本的相关系数
k=cov_ab / (np.std(bhp_closing_prices, ddof=1) * \
    np.std(vale_closing_prices, ddof=1))
print(k)
# 输出相关矩阵
m = np.corrcoef(bhp_closing_prices,
                vale_closing_prices)
print(m)

covm = np.cov(bhp_closing_prices,
              vale_closing_prices)
print(covm)

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



