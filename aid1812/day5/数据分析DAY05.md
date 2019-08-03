# 数据分析DAY05

#### 线性预测

什么是线性关系？

```
1  2  3  4  5
60 65 70 75 ?
```

假设一组数据符合一种线性规律， 那么就可以预测未来将会出现的数据。

```
a  b  c  d  e  f  ?
假设符合线性关系：
ax + by + cz = d
bx + cy + dz = e
cx + dy + ez = f
可以解三元一次方程组求得 x, y, z  -> 
k = ax + by + cz
```

$$
\left[ 
\begin{array}{c}
a & b & c \\
b & c & d \\
c & d & e \\
\end{array}
\right]
\times
\left[ 
\begin{array}{c}
x \\
y \\
z \\
\end{array}
\right]
=
\left[ 
\begin{array}{c}
d \\
e \\
f \\
\end{array}
\right]
$$

numpy提供了API求取xyz向量：

```python
x = np.linalg.lstsq(A, B)[0]
```

案例：使用线性预测，预测下一天的收盘价。

```python
"""
demo01_lp.py 线性预测
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
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

#绘制收盘价的折线图

mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Price', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
# 设置刻度定位器
ax = mp.gca()
#设置主刻度定位器-每周一一个主刻度
major_loc=md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
# 设置次刻度定位器为日定位器
minor_loc=md.DayLocator()
ax.xaxis.set_minor_locator(minor_loc)

mp.plot(dates, closing_prices, 
	c='dodgerblue', linestyle='--',
	linewidth=2, label='AAPL', alpha=0.5)

# 线性预测  
N = 5

pred_prices = np.zeros(
	closing_prices.size - 2*N +1)
for i in range(pred_prices.size):
	# 通过前6个元素，组成3元一次方程组，解之
	# 整理出矩阵A, 列向量B，调用API求出xyz
	A = np.zeros((N, N))
	for j in range(N):
		A[j,] = closing_prices[i+j : i+j+N]
	B = closing_prices[i+N : i+N*2]
	x = np.linalg.lstsq(A, B)[0]
	# print(x)
	pred_price = B.dot(x)  # 点乘
	# print(pred_price, closing_prices[6])
	pred_prices[i] = pred_price

# 绘制图像
mp.plot(dates[2*N:], pred_prices[:-1],
	'o-', c='orangered', label='pred_price')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

#### 线性拟合

线性拟合可以寻求与一组散点走向趋势规律相匹配的线性表达式方程。

有一组散点坐标：

```
[x1,y1], [x2,y2], [x3,y3]...
[xn,yn]
```

根据 y=kx+b 方程：

```
kx1 + b = y1
kx2 + b = y2
kx3 + b = y3
..
kxn + b = yn
```

写成矩阵相乘的形式：
$$
\left[ 
\begin{array}{c}
x_1 & 1 \\
x_2 & 1 \\
x_3 & 1 \\
x_n & 1 \\
\end{array}
\right]
\times
\left[ 
\begin{array}{c}
k \\
b \\
\end{array}
\right]
=
\left[ 
\begin{array}{c}
y_1\\
y_2\\
y_3\\
y_n\\
\end{array}
\right]
$$
np.linalg.lstsq(A, B) 可以求出最优的k与b使得与所有点的误差最小。

案例：绘制股价趋势线。趋势可以表示为（最高价、最低价、收盘价的均值）

```python
"""
demo02_lstsq.py 线性拟合
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
	lowest_prices, closing_prices = \
	np.loadtxt('../da_data/aapl.csv',
		usecols=(1,3,4,5,6),
		unpack=True,
		dtype='M8[D], f8, f8, f8, f8',
		delimiter=',',
		converters={1:dmy2ymd})

#绘制收盘价的折线图

mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Price', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
# 设置刻度定位器
ax = mp.gca()
#设置主刻度定位器-每周一一个主刻度
major_loc=md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
# 设置次刻度定位器为日定位器
minor_loc=md.DayLocator()
ax.xaxis.set_minor_locator(minor_loc)

dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, 
	c='dodgerblue', linestyle='--',
	linewidth=2, label='AAPL', alpha=0.2)

# 绘制所有的趋势点 
trend_points = (opening_prices + \
	highest_prices + closing_prices)/3
mp.scatter(dates, trend_points, 
	s=60, c='orangered', label='TrendPoints')

# 线性拟合，整理A与B
days = dates.astype('M8[D]').astype('int32')
A=np.column_stack((days, np.ones_like(days)))
B=trend_points
x = np.linalg.lstsq(A, B)[0]
# x -> [k, b]
trend_line = x[0]*days + x[1]
mp.plot(dates, trend_line, c='orangered',
	label='TrendLine')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

#### 数组的裁剪与压缩

```python
# 数组裁剪
ndarray.clip(min=下限, max=上限)
# 数组的压缩
ndarray.compress(条件)
```

```python
"""
demo03_clip.py  数组处理函数
"""
import numpy as np

a = np.arange(1, 11)
print(a)
print(a.clip(min=5, max=8))
print(np.clip(a, 5, 8))

# print(a.compress((a>3) and (a<8)))
print(a.compress(np.all([a>3, a<8], axis=0)))
```

#### 协方差、相关矩阵、相关系数

通过两组统计数据计算而得的协方差可以评估者两组统计数据的相关程度。

样本：

```
A = [a1, a2, a3 ... an]
B = [b1, b2, b3 ... bn]
```

平均值：

```
ave_a = np.mean(A)
ave_b = np.mean(B)
```

离差：

```
dev_a = A - ave_a
dev_b = B - ave_b
```

**协方差：**

```
cov_ab = np.mean(dev_a * dev_b)
cov_ba = np.mean(dev_b * dev_a)
```

协方差可以简单反应两组统计样本的相关性，值为正则为正相关；值为负，则为负相关。绝对值越大相关性越强。

案例：vale.csv    bhp.csv

```python

# 计算两只股票收盘价的协方差
vale_mean = np.mean(vale_closing_prices)
bhp_mean = np.mean(bhp_closing_prices)
vale_dev = vale_closing_prices-vale_mean
bhp_dev = bhp_closing_prices-bhp_mean
cov = np.mean(vale_dev * bhp_dev)
print(cov)
```

**相关系数**

协方差除以两组统计样本标准差之积是一个[-1,1]之间数，该结果成为两组统计样本的相关性系数。

通过相关性系数分析两组样本的相关性：

```
若相关系数越接近于1，两组样本越正相关
若相关系数越接近于-1，两组样本越负相关
若相关系数越接近于0，两组样本越不相关
```

案例：

```python
# 相关性系数
k = cov / (np.std(vale_closing_prices) * np.std(bhp_closing_prices))
print(k)
```

**相关矩阵**

```python
m = np.corrcoef(a, b)
```

$$
\left[ 
\begin{array}{c}
a与a的相关系数 & a与b的相关系数 \\
b与a的相关系数 & b与b的相关系数 \\
\end{array}
\right]
$$

**协方差矩阵**

```python
m = np.cov(a, b)
```

$$
\left[ 
\begin{array}{c}
a与a的协方差 & a与b的协方差 \\
b与a的协方差 & b与b的协方差 \\
\end{array}
\right]
$$



#### 多项式拟合

多项式的一般形式：
$$
y = p_0x^n + p_1x^{n-1} + p_2x^{n-2} + ...p_n
$$
多项式拟合的目的为了找到一组p<sub>0</sub> ~ p<sub>n</sub> , 使得拟合方程尽可能的与实际样本数据向符合。

假设拟合得到的多项式如下：
$$
f(x) =p_0x^n + p_1x^{n-1} + p_2x^{n-2} + ...p_n
$$
则拟合函数与真实结果的误差方如下：
$$
loss = (y_1-f(x_1))^2 + (y_2-f(x_2))^2 + ... (y_n-f(x_n))^2
$$
numpy提供了一些有关多项式操作的API：

```python
# 多项式拟合 
# X，Y为样本点的坐标
# 返回的P为多项式的系数数组
P = np.polyfit(X, Y, 最高次幂)
# P代表多项式， x代表一组自变量
# 该方法可以将x带入多项式求出相对应的函数值
y = np.polyval(P, X)
# 求多项式的导函数  Q为导函数的系数
Q = np.polyder(P)
# 求多项式函数的根
xs = np.roots(P)
# 求P1与P2两个多项式函数的差函数
Q = np.polysub(P1, P2)
```

案例：求多项式 y = 4x<sup>3</sup> + 3x<sup>2</sup> -1000x +1 曲线拐点位置。

```python
"""
demo05_poly.py 多项式演示
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-20, 20, 1000)
y = 3*x**3 + 3*x**2 - 1000*x + 1
# 
P = [3, 3, -1000, 1]
Q = np.polyder(P)
xs = np.roots(Q)
print(xs)
ys = np.polyval(P, xs)
mp.plot(x, y)
mp.scatter(xs, ys, s=60, c='red', marker='o')
mp.show()
```

案例：使用多项式函数拟合两只股票bhp、vale的差价函数。

```python

# 计算差价
diff_prices = bhp_closing_prices-vale_closing_prices
mp.plot(dates, diff_prices, alpha=0.5)
# 多项式拟合
days = dates.astype('M8[D]').astype('int32')
P = np.polyfit(days, diff_prices, 10)
# 绘制多项式函数
y = np.polyval(P, days)
mp.plot(dates, y, linestyle='-', 
	linewidth=2, c='orangered', 
	label='polyfit line')
```

#### 数据平滑

数据平滑处理通常包含降噪、拟合等操作。降噪的功能在于去除额外的影响因素， 拟合的目的在于数学模型化，可以通过更多的数学方法识别曲线的特征。

案例：绘制两只股票收益率曲线。

收益率=(当天的收盘价-前一天的收盘价)/前一天的收盘价

```python
"""
demo07_sjph.py  数据平滑
1. 计算两只股票的收益率曲线 并绘制
2. 分析曲线形状，确定投资策略
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

dates, vale_closing_prices = \
	np.loadtxt('../da_data/vale.csv',
		usecols=(1, 6),
		unpack=True,
		dtype='M8[D], f8',
		delimiter=',',
		converters={1:dmy2ymd})

bhp_closing_prices = \
	np.loadtxt('../da_data/bhp.csv',
		usecols=(6,),
		unpack=True,
		delimiter=',')

#绘制收盘价的折线图
mp.figure('PolyFit', facecolor='lightgray')
mp.title('PolyFit', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Price', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
# 设置刻度定位器
ax = mp.gca()
#设置主刻度定位器-每周一一个主刻度
major_loc=md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
# 设置次刻度定位器为日定位器
minor_loc=md.DayLocator()
ax.xaxis.set_minor_locator(minor_loc)
dates = dates.astype(md.datetime.datetime)

# 计算收益率
bhp_returns = np.diff(bhp_closing_prices) \
		/ bhp_closing_prices[:-1]
vale_returns = np.diff(vale_closing_prices) \
		/ vale_closing_prices[:-1]
dates = dates[:-1]
mp.plot(dates, bhp_returns, c='dodgerblue',
	label='bhp_returns', alpha=0.3)
mp.plot(dates, vale_returns, c='orangered',
	label='vale_returns', alpha=0.3)

# 卷积降噪
kernel = np.hanning(8)
kernel /= kernel.sum()
print(kernel)
bhp=np.convolve(bhp_returns,kernel,'valid')
vale=np.convolve(vale_returns,kernel,'valid')
# mp.plot(dates[7:], vale, c='orangered',
# 	label='vale_convolved')
# mp.plot(dates[7:], bhp, c='dodgerblue',
# 	label='bhp_convolved')

# 针对vale与bhp分别做多项式拟合
days = dates[7:].astype('M8[D]').astype('int32')
P_vale = np.polyfit(days, vale, 3)
P_bhp = np.polyfit(days, bhp, 3)
y_vale = np.polyval(P_vale, days)
y_bhp =  np.polyval(P_bhp, days)
mp.plot(dates[7:], y_vale, c='orangered',
	label='vale_convolved')
mp.plot(dates[7:], y_bhp, c='dodgerblue',
	label='bhp_convolved')

# 求两个多项式的交点位置
P = np.polysub(P_vale, P_bhp)
xs = np.roots(P)
dates = np.floor(xs).astype('M8[D]')
print(dates)
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```







