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

dates,opening_prices,highest_prices,lowest_prices,closing_prices = \
    np.loadtxt('../../datascience_ziliao/素材/da_data/aapl.csv',delimiter=',',usecols=(1,3,4,5,6),
           unpack=True,dtype='M8[D],f8,f8,f8,f8',converters={1:dmy2ymd})

mp.figure('AAPL',facecolor='lightgray')
mp.title('AAPL',fontsize=18)
mp.xlabel('date',fontsize=14)
mp.ylabel('Closing prices',fontsize=14)
mp.tick_params(labelsize=8)
mp.grid(linestyle=':')

#设置x轴的刻度定位器，使之更适合显示日期数据
ax = mp.gca()
#主刻度，每周一
ma_loc = md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(ma_loc)
ax.xaxis.set_major_formatter(
    md.DateFormatter('%Y-%m-%d')
)
#次刻度，每天
mi_loc = md.DayLocator()
ax.xaxis.set_minor_locator(mi_loc)

#日期数据类型转换，更适合绘图
dates = dates.astype(md.datetime.datetime)
mp.plot(dates,closing_prices,linewidth=2,linestyle='--',color='dodgerblue',label='AAPL')

# 绘制蜡烛图
# 控制颜色
rise = closing_prices >= opening_prices
color = np.array(
		[('white' if x else 'green') \
		for x in rise])
edgecolor = np.array(
		[('red' if x else 'green') \
		for x in rise])
# 绘制实体
mp.bar(dates, closing_prices-opening_prices,
	0.8, opening_prices, color=color,
	edgecolor=edgecolor, zorder=3)
# 绘制影线
mp.vlines(dates, lowest_prices,
	highest_prices, color=edgecolor)


mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()#自动格式化x日期
mp.show()