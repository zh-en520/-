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

mean = np.mean(closing_prices)
D = closing_prices - mean#离差
D2 = D**2#离差方
V = np.mean(D2)#总体方差
S = np.sqrt(V)#总体标准差
print(S)

print(closing_prices)
s = np.std(closing_prices)
print(s)
s = np.std(closing_prices,ddof=1)
print(s)