import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2weekday(dmy):
    #把dmy格式的字符串专程ymd格式字符串
    dmy = str(dmy,encoding='utf-8')
    d = dt.datetime.strptime(dmy,'%d-%m-%Y')#strparsetime
    d = d.date()
    wday = d.weekday()
    return wday

wdays,opening_prices,highest_prices,lowest_prices,closing_prices,volumns = \
    np.loadtxt('../../datascience_ziliao/素材/da_data/aapl.csv',delimiter=',',usecols=(1,3,4,5,6,7),
           unpack=True,dtype='f8,f8,f8,f8,f8,f8',converters={1:dmy2weekday})
print(wdays)
ave_prices = np.zeros(5)#存储最终结果
for wday in range(5):
    ave_prices[wday] = closing_prices[wdays==wday].mean()
print(ave_prices)

#测试轴向统计的API
a = np.arange(1,13).reshape(3,4)
print(a)

def func(data):
    return np.sum(data)
r = np.apply_along_axis(func,0,a)
print(r)