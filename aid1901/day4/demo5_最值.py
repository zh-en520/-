import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2ymd(dmy):
    #把dmy格式的字符串专程ymd格式字符串
    dmy = str(dmy,encoding='utf-8')
    d = dt.datetime.strptime(dmy,'%d-%m-%Y')#strparsetime
    d = d.date()
    ymd = d.strftime('%Y-%m-%d')#strformattime(格式化)
    return ymd

dates,opening_prices,highest_prices,lowest_prices,closing_prices,volumns = \
    np.loadtxt('../../datascience_ziliao/素材/da_data/aapl.csv',delimiter=',',usecols=(1,3,4,5,6,7),
           unpack=True,dtype='M8[D],f8,f8,f8,f8,f8',converters={1:dmy2ymd})

#评估AAPL股票波动性
max_val = np.max(highest_prices)
min_val = np.min(lowest_prices)

#查看最高价最低价时的日期
print('max:',dates[np.argmax(highest_prices)])
print('min:',dates[np.argmin(lowest_prices)])

#查看maximum和minimum
a = np.arange(1,10).reshape(3,3)
b = np.arange(1,10)[::-1].reshape(3,3)
print(np.maximum(a,b))
print(np.minimum(a,b))

