# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_lp.py 线性预测
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

# 实现线性预测
N = 3
# 计算预测股价数组
pred_prices = np.zeros(
    closing_prices.size - 2*N + 1)
for i in range(pred_prices.size):
    # 整理A与B，通过lstsq方法求得模型参数
    A = np.zeros((N, N))
    for j in range(N):
        A[j, ] = closing_prices[i+j:i+j+N]
    # print(A)
    B = closing_prices[i+N:i+N*2]
    print(B)
    # 计算模型参数
    x = np.linalg.lstsq(A, B)[0]
    y = np.linalg.solve(A, B)
    print('y:',y)
    # print('x:',x)
    pred = B.dot(x)  # 点积  对应位置相乘再相加
    pred_prices[i] = pred
    # print(pred_prices[i])
# 绘制预测股价的折线图
print(dates.shape)
print(pred_prices[-1])
mp.plot(dates[2*N:], pred_prices[:-1], 
    'o-', c='orangered', label='Predicts')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()



