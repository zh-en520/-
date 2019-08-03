# 数据分析DAY04

#### 绘制K线图

```python
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
```

#### 算数平均值

```
s = [s1, s2, s3 ... sn]
mean = (s1 + s2 + ... + sn) / n
```

均值可以体现对真值的无偏估计。

```python
mean = np.mean(array)
mean = array.mean()
```

案例：计算收盘价的均值。

```python
# 计算均值
mean = np.mean(closing_prices)
mean = closing_prices.mean()
print(mean)
mp.hlines(mean, dates[0], dates[-1], 
	color='orangered', linewidth=2, 
	label='Mean(closing_prices)')
```

#### 加权平均值

```
样本： S = [s1, s2, ..  , sn]
权重： W = [w1, w2, ..  , wn]
加权平均值：
a = (s1w1 + s2w2 + .. snwn)/(w1+..+wn)
```

np提供了求加权平均值的API：

```python
# array 原始数组
# ws  权重数组
np.average(array, weights=ws)
```

案例：TWAP - 时间加权平均价格

以时间因素作为每天股价的权重，使得离得较近的样本权重提高，较远的样本权重降低，有利于提高加权均值对真值估计的准确度。

```python
# 计算加权平均价格
times = np.arange(1, 31)
wmean = np.average(closing_prices, 
		weights=times)
mp.hlines(wmean, dates[0], dates[-1], 
	color='green', linewidth=2, label='TWAP')
```

VWAP - 交易量加权平均价格

```python
# 计算交易量加权平均价格
vwap = np.average(closing_prices, 
		weights=volumns)
mp.hlines(vwap, dates[0], dates[-1], 
	color='violet', linewidth=2, label='VWAP')
```

#### 最值

```python
np.max(a)  # 最大值
np.min(a)  # 最小值
np.ptp(a)  # a数组的极差
```

```python
np.argmax(a) # 返回最大值的索引
np.argmin(a) # 返回最小值的索引
```

```python
# 将a与b数组中对应元素最大值留下，组成新数组
np.maximum(a, b) 
# 将a与b数组中对应元素最小值留下，组成新数组
np.minimum(a, b)
```

案例：

```python
"""
demo03_max.py 最值
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2ymd(dmy):
	# 把二进制字符串转为普通字符串
	dmy = str(dmy, encoding='utf-8')
	t = dt.datetime.strptime(dmy, '%d-%m-%Y')
	s = t.date().strftime('%Y-%m-%d')
	return s

dates, opening_prices, highest_prices, \
	lowest_prices, closing_prices, \
	volumns = np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6,7),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

# 评估AAPL股价的波动性
max_price = np.max(highest_prices)
min_price = np.min(lowest_prices)
print(max_price, '~', min_price)

max_date = dates[np.argmax(highest_prices)]
min_date = dates[np.argmin(lowest_prices)]
print(max_date)
print(min_date)

a = np.arange(1, 10).reshape(3, 3)
b = a.ravel()[::-1].reshape(3, 3)
print(a)
print(b)
print(np.maximum(a, b))
print(np.minimum(a, b))
```

#### 中位数

对一个有序数组取中间位置的元素。

```
s数组是一个有序数组
s = [s1, s2, s3, s4, s5, s6]
其中位数为：s3与s4的均值
s = [s1, s2, s3, s4, s5]
其中位数为：s3
```

```python
median = np.median(ary)
```

案例

```python
median = np.median(closing_prices)
mp.hlines(median, dates[0], dates[-1], 
	color='gold', linewidth=2, label='median')
```

当样本数据中有异常值时，通常使用中位数而不使用平均数。尽量减少异常值对均值的影响。

#### 标准差

```
样本： S = [s1, s2, s3 ... sn]
均值： m = np.mean(s)
离差： D = [d1, d2 ... dn], di=si-m
离差方：Q = [q1, q2 ... qn], qi=di^2
总体方差：v = (q1+q2+..+qn)/n
总体标准差：s = np.sqrt(v)
样本方差：v2 = (q1+q2+..+qn)/(n-1)
样本标准差：s2 = np.sqrt(v2)
```

标准差的指标指示了一组数据的距离均值的离散程度。标准差越大震荡越剧烈，反之震荡越平缓。

```python
# 总体标准差
s = np.std(array)
# 样本标准差
s2 = np.std(array, ddof=1)
```

### 案例应用

#### 时间数据处理

案例：统计每个周一、周二...周五的收盘价的均值，并放入一个数组。

```python
"""
demo05_tp.py 时间数据处理
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2wdays(dmy):
	# 把二进制字符串转为普通字符串
	dmy = str(dmy, encoding='utf-8')
	t = dt.datetime.strptime(dmy, '%d-%m-%Y')
	wday = t.date().weekday()
	return wday

wdays, opening_prices, highest_prices, \
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2wdays})

ave_closing_prices = np.zeros(5)
for wday in range(ave_closing_prices.size):
	ave_closing_prices[wday] = \
	    closing_prices[wdays==wday].mean()

for wday, ave_closing_price in zip(
	['MON', 'TUE', 'WED', 'THU', 'FRI'],
	ave_closing_prices):
	print(wday, ave_closing_price)
```

#### 二维数组的轴向汇总

```python
def func(ary):
    pass
# 按照axis的轴向，把array数据交给func函数
# 进行汇总
np.apply_along_axis(func, axis, array)
```

案例：

```python
"""
demo06_apply.py 数据的轴向汇总
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2wdays(dmy):
	# 把二进制字符串转为普通字符串
	dmy = str(dmy, encoding='utf-8')
	t = dt.datetime.strptime(dmy, '%d-%m-%Y')
	wday = t.date().weekday()
	return wday

wdays, opening_prices, highest_prices, \
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='f8, f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2wdays})

data = [opening_prices, highest_prices,
		lowest_prices, closing_prices]
data = np.array(data)
print(data.shape)

# 行方向的数据汇总
def rowfunc(row):
	return np.mean(row), np.std(row)

r = np.apply_along_axis(rowfunc, 1, data)
print(np.round(r,2))

# 列方向的数据汇总
def colfunc(col):
	return np.mean(col)

r = np.apply_along_axis(colfunc, 0, data)
print(np.round(r,2))
```

#### 移动平均线

收盘价5日移动均线：从第五天开始，每天计算最近5天的收盘价的均值从而构成的一条线。

案例：

```python
# 绘制5日均线
sma5 = np.zeros(closing_prices.size-4)
for i in range(sma5.size):
	sma5[i] = closing_prices[i:i+5].mean()

mp.plot(dates[4:], sma5, c='orangered',
	label='SMA(5)')
```

#### 卷积

卷积运算的运算规则：

```
a = [1 2 3 4 5]   原始数组
b = [6 7 8]       卷积核数组
使用b作为卷积核对a数组执行卷积运算的过程如下
           40 61 82      有效卷积 valid
        19 40 61 82 67    同维卷积 same
      6 19 40 61 82 67 40 完全卷积 full
0  0  1  2  3  4  5  0  0
8  7  6
   8  7  6
      8  7  6
         8  7  6
            8  7  6
               8  7  6
                  8  7  6
```

使用卷积运算实现5日移动平均线：

```
a = [a, b, c, d, e, f, g, h, i, j...]
b = [1/5, 1/5, 1/5, 1/5, 1/5]
```

卷积相关API：

```python
# array 原始数组
# kernel 卷积核
# valid 卷积类型：same  full  valid
r=np.convolve(array, kernel, 'valid')
```

```python

# 基于卷积运算实现5日均线
kernel = np.ones(5) / 5
sma52 = np.convolve(
	closing_prices, kernel, 'valid')
mp.plot(dates[4:], sma52, c='orange',
	label='SMA(5-2)', linewidth=7, 
	alpha=0.5)

# 基于卷积运算实现10日均线
kernel = np.ones(10) / 10
sma10 = np.convolve(
	closing_prices, kernel, 'valid')
mp.plot(dates[9:], sma10, c='red',
	label='SMA(10)')
```

当遇到某些业务需要边移动边计算，并且计算当前位置的数值时需要周围其他位置的数据作为参数。比较适合使用卷积。

**加权卷积**

```
s = [a, b, c, d, e, f, g, h ...]
k = [9/25, 7/25, 5/25, 3/25, 1/25]
```

案例：

```python
# 5日加权卷积均线
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
print(weights)
sma53 = np.convolve(closing_prices, 
	weights[::-1], 'valid')
mp.plot(dates[4:], sma53, c='violet',
	label='SMA(5-3)', linewidth=2)
```

#### 布林带

布林带由3条线组成：

中轨： 移动均线

上轨： 中轨 + 2*5日标准差

下轨： 中轨 - 2*5日标准差

案例：绘制5日均线的布林带：

```python
# 绘制布林带
stds = np.zeros(sma53.size)
for i in range(stds.size):
	stds[i] = closing_prices[i:i+5].std()
lowers = sma53 - 2*stds
uppers = sma53 + 2*stds
mp.plot(dates[4:], lowers, c='orangered',
	label='lowers')
mp.plot(dates[4:], uppers, c='orangered',
	label='uppers')
mp.fill_between(dates[4:], lowers, 
	uppers, lowers<uppers, color='orangered',
	alpha=0.2)
```













