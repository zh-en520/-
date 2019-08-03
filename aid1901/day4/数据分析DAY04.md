# 数据分析DAY04

#### 绘制K线图

绘制每一天的蜡烛图:

```python
# 绘制蜡烛图
# 整理颜色
rise = closing_prices > opening_prices
color = np.array(
    [('white' if x else 'green') for x in rise])
ecolor = np.array(
    [('red' if x else 'green') for x in rise])
# 绘制实体
mp.bar(dates, closing_prices-opening_prices,
	0.8, opening_prices, color=color,
	edgecolor=ecolor, zorder=3)
# 绘制影线
mp.vlines(dates, lowest_prices, highest_prices,
	color=ecolor)
```

#### 算数平均值

```python
s = [s1, s2, s3 .. sn]
mean = (s1 + s2 + .. sn)/n
```

均值可以体现对真值的无偏估计。

```python
mean = np.mean(s)   # 第一种	
mean = s.mean()		# 第二种
```

案例：

```python
# 计算均值，绘制图像
mean = np.mean(closing_prices)
mp.hlines(mean, dates[0], dates[-1],
    color='orangered', label='Mean(CP)')
```

#### 加权平均值

```python
s = [s1, s2, s3 .. sn]    原数组
w = [w1, w2, w3 .. wn]    权重数组
# 加权均值
a = (s1w1 + s2*w2 + s3*w3 .. sn*wn) / (w1+..+wn)
```

相关API：

```python
a = np.average(array, weights=w)
```

案例：VWAP（交易量加权平均价格）

交易量体现了市场对当前交易价格的认可度。交易量越高代表市场对当前交易价格越认可，该价格越接近股票价值的真值。

```python
# 计算VWAP 交易量加权均值
a = np.average(closing_prices, weights=volumns)
mp.hlines(a, dates[0], dates[-1],
    color='violet', label='Average(CP)')
```

案例：TWAP（时间加权平均价格）

越靠近当前时间的收盘价对均值的影响程度（权重）越高。

```python
# 计算TWAP 时间加权均值
w = np.linspace(1, 7, 30)
a = np.average(closing_prices, weights=w)
mp.hlines(a, dates[0], dates[-1],
    color='gold', label='TWAP')
```

#### 最值

```python
np.max(a)  # 最大值
np.min(a)  # 最小值
np.ptp(a)  # 极差  max-min

np.argmax(a)   # 最大值索引
np.argmin(a)   # 最小值索引

np.maximum(a, b)  # a与b对应位置元素相比，留下较大的
np.minimum(a, b)  # a与b对应位置元素相比，留下较小的
```

案例：

```python
# 评估AAPL股票波动性
max_val = np.max(highest_prices)
min_val = np.min(lowest_prices)
print(max_val, '~', min_val)

# 查看最高价最低价时的日期
print('max:', dates[np.argmax(highest_prices)])
print('min:', dates[np.argmin(lowest_prices)])

# 
a = np.arange(1, 10).reshape(3, 3)
b = np.arange(1, 10)[::-1].reshape(3, 3)
print(a)
print(b)
print(np.maximum(a, b))
print(np.minimum(a, b))
```

#### 中位数

中位数也可以估计一组数据的真值。

将多个样本按照大小排序，取中间位置的元素。如果一组数据有些异常值对最终结果影响较大。

```python
median = np.median(array)
```

案例：

```python
# 计算中位数
median = np.median(closing_prices)
mp.hlines(median, dates[0], dates[-1],
    color='blue', label='Median(CP)')
print(median)
# 自己算
sorted_prices = np.msort(closing_prices)
size = closing_prices.size
median = ((sorted_prices[int((size-1)/2)]) + \
    (sorted_prices[int(size/2)]))/2
print(median)
```

#### 标准差

```python
样本：S = [s1, s2, s3 .. sn]   
均值：m = np.mean(S)
离差：D = [d1, d2, d3 .. dn], di=si-m
离差方：Q = [q1, q2, .. qn], qi=di^2
总体方差：v = np.mean(Q) # 表示这组数据的离散程度
总体标准差：s = sqrt(v)

样本方差：v' = (q1+q2+..+qn)/(n-1)
样本标准差：s' = sqrt(v')
```

标准差相关API：

```python
# 数组的总体标准差
s = np.std(array)
# 数组的样本标准差
s = np.std(array, ddof=1)
```

案例：

```python
# 手动
mean = np.mean(closing_prices)
D = closing_prices-mean   # 离差
D2 = D**2 # 离差方
V = np.mean(D2)  # 总体方差
S = np.sqrt(V) # 总体标准差
print(S)

# API
s = np.std(closing_prices)
print(s)
s = np.std(closing_prices, ddof=1)
print(s)
```

#### 数据轴向统计

案例：统计每个周一、周二、.. 周五的收盘价的均值，并输出。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_wd.py  时间数据处理
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt

def dmy2wday(dmy):
    # 把dmy格式的字符串转成ymd格式字符串
    dmy = str(dmy, encoding='utf-8')
    d = dt.datetime.strptime(dmy, '%d-%m-%Y')
    d = d.date()
    wday = d.weekday() # 返回周几
    return wday

wdays, opening_prices, highest_prices, \
    lowest_prices, closing_prices, \
    volumns = \
    np.loadtxt('../da_data/aapl.csv',
    delimiter=',', usecols=(1, 3, 4, 5, 6, 7),
    unpack=True, 
    dtype='f8, f8, f8, f8, f8, f8',
    converters={1:dmy2wday})

print(wdays)

ave_prices = np.zeros(5) # 存储最终结果
for wday in range(5):
    ave_prices[wday] = \
        closing_prices[wdays==wday].mean()
print(ave_prices)
```

统计相关API：

```python
def func(data):
    return r
# axis： 轴向 0垂直方向   1水平方向
# array是一个二维数组，沿着axis定义的轴向执行数据汇总
# 把每一行（列）的数据交给func函数进行处理并获取汇总结果
np.apply_along_axis(func, axis, array)
```

案例：

```python
# 测试轴向统计的API
a = np.arange(1, 13).reshape(3, 4)
print(a)

def func(data):
    return (np.sum(data), np.max(data), np.std(data))

# 把a的每一列都交给func函数进行处理
r = np.apply_along_axis(func, 0, a)
print(r)
```

#### 移动均线

收盘价的5日均线：从第五天开始，每天计算最近5天的收盘价的均值而构成的一条线。

```
(a+b+c+d+e) / 5
(b+c+d+e+f) / 5
(c+d+e+f+g) / 5
....
```

案例：实现5日均线。

```python
# 绘制5日均线
sma = np.zeros(closing_prices.size - 4)
for i in range(sma.size):
    sma[i] = closing_prices[i:i+5].mean()
mp.plot(dates[4:], sma, color='orangered',
    label='SMA5')
```

#### 卷积

卷积运算的运算规则：

```
a = [1 2 3 4 5]    原数组
b = [8 7 6]        卷积核数组
使用b作为卷积核对a数组执行卷积运算的过程如下：

           44 65 86        - 有效卷积（valid）
        23 44 65 86 59     - 同维卷积（same）
      8 23 44 65 86 59 30  - 完全卷积（full）
0  0  1  2  3  4  5  0  0 
6  7  8
   6  7  8
      6  7  8
         6  7  8
            6  7  8
               6  7  8
                  6  7  8
```

基于卷积运算解决5日均线问题：

```python
a = [a, b, c, d, e, f, g, h]   原数组
b = [1/5, 1/5, 1/5, 1/5, 1/5]  卷积核
```

卷积相关API：

```python
# array：原数组
# kernel：卷积核
# 卷积类型：'same' 'full' 'valid'
r = np.convolve(array, kernel, 'valid')
```

案例：

```python
# 基于卷积实现5日均线
kernel = np.ones(5) / 5
sma52 = np.convolve(
    closing_prices, kernel, 'valid')
mp.plot(dates[4:], sma52, color='orangered',
    linewidth=7, alpha=0.3,
    label='SMA52')

# 基于卷积实现10日均线
kernel = np.ones(10) / 10
sma10 = np.convolve(
    closing_prices, kernel, 'valid')
mp.plot(dates[9:], sma10, color='limegreen',
    label='SMA10')
```

**加权卷积**

```
原数组：[a     b     c     d     e]
卷积核：[1/25, 3/25, 5/25, 7/25, 9/25]
```

案例：

```python
# 5日加权均线
weights = np.exp(np.linspace(-1, 0, 5))
weights = weights[::-1]
weights /= weights.sum()
sma53 = np.convolve(
    closing_prices, weights, 'valid')
mp.plot(dates[4:], sma53, color='gold',
    label='SMA53')
```

卷积适合解决的问题：

1. 边移动变运算，对应位置相乘再相加。
2. 数据平滑处理，数据降噪。















