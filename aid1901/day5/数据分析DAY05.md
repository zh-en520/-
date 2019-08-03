# 数据分析DAY05

#### 布林带

布林带由三条线组成：

上轨：中轨 + 2*5日收盘价标准差

中轨：移动平均线

下轨：中轨 - 2*5日收盘价标准差

案例：绘制5日均线的布林带。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_bd.py  布林带
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
    alpha=0.5)

# 5日加权均线
weights = np.exp(np.linspace(-1, 0, 5))
weights = weights[::-1]
weights /= weights.sum()
sma53 = np.convolve(
    closing_prices, weights, 'valid')
mp.plot(dates[4:], sma53, color='orangered',
    label='SMA53')

# 绘制布林带的上轨与下轨
stds = np.zeros(sma53.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i+5].std()
lower = sma53 - 2 * stds
upper = sma53 + 2 * stds
mp.plot(dates[4:], upper, color='orangered',
    label='Upper')
mp.plot(dates[4:], lower, color='orangered',
    label='Lower')
# 填充
mp.fill_between(dates[4:], lower, upper,
    upper > lower, color='orangered', 
    alpha=0.3)

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

带宽收窄

当汇价经过短时间的大幅拉升后带宽收窄，则应及时卖出。前期拉升的幅度越大、上下轨之间的距离越大，则未来下跌幅度越大。该形态的确立是以汇价的上轨线开始掉头向下、汇价向下跌破短期均线为准。对于收口型喇叭口形态的出现，投资者如能则能保住收益、减少较大的下跌损失。

当汇价经过长时间的下跌后带宽收窄，应观望等待，也可以少量建仓。这是一种显示汇价将长期小幅盘整筑底的形态。它是形成于汇价经过长期大幅下跌后。面临着长期[调整](https://baike.baidu.com/item/%E8%B0%83%E6%95%B4)的一种走势。布林线的上下轨线的逐步小幅靠拢，预示着多空双方的力量逐步处于平衡，汇价将处于长期横盘整理的行情中。

##### 布林线指标

⑴股价由下向上穿越下轨线（Down）时，可视为买进信号。

⑵股价由下向上穿越中轨时，股价将加速上扬，是加仓买进的信号。

⑶股价在中轨与上轨(UPER）之间波动运行时为多头市场，可持股观望。

⑷股价长时间在中轨与上轨（UPER）间运行后，由上向下跌破中轨为卖出信号。

⑸股价在中轨与下轨（Down）之间向下波动运行时为空头市场，此时投资者应持币观望。

⑹布林中轨经长期大幅下跌后转平，出现向上的拐点，且股价在2～3日内均在中轨之上。此时，若股价回调，其回档低点往往是适量低吸的中短线切入点。

⑺对于在布林中轨与上轨之间运作的强势股，不妨以回抽中轨作为低吸买点，并以中轨作为其重要的止盈、止损线。

⑻飚升股往往股价会短期冲出布林线上轨运行，一旦冲出上轨过多，而成交量又无法持续放出，注意短线高抛了结，如果由上轨外回落跌破上轨，此时也是一个卖点。



#### 线性预测

```python
输入: 1  2  3  4  5
输出: 60 65 70 75 ?
```

通过一组已知的输入与输出可以构建出一个简单的线性方程。这样可以把预测输入带入线性方程从而求得预测输出，达到数据预测的目的。

假设股价符合一种线性规律，那么就可以预测未来的股价。

```
a  b  c  d  e  f  ?

ax + by + cz = d
bx + cy + dz = e
cx + dy + ez = f
通过上述公式求出x、y、z，把def带入公式预测结果：
? = dx + ey + fz
```

基于矩阵的方式解方程组：
$$
\left[ \begin{array}{ccc}
a & b & c \\
b & c & d \\
c & d & e \\
\end{array} \right]
\times
\left[ \begin{array}{ccc}
x \\
y \\
z \\
\end{array} \right]
=
\left[ \begin{array}{ccc}
d \\
e \\
f \\
\end{array} \right] \\ \\
A  \quad\quad\quad\quad\quad\quad\quad\quad\quad B
$$
基于numpy提供的API，通过A与B求得xyz：

```python
r = np.linalg.lstsq(A, B)[0]
```

案例：基于线性预测，预测收盘价格。

```python
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
    B = closing_prices[i+N:i+N*2]
    # 计算模型参数
    x = np.linalg.lstsq(A, B)[0]
    pred = B.dot(x)  # 点积  对应位置相乘再相加
    pred_prices[i] = pred
# 绘制预测股价的折线图
mp.plot(dates[2*N:], pred_prices[:-1], 
    'o-', c='orangered', label='Predicts')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

#### 线性拟合

线性拟合可以寻求与一组散点走向趋势相适应的线性表达式方程。

```
[x1, y1],[x2, y2],[x3, y3][xn, yn]
```

根据线性方程得：

```
kx1 + b = y1
kx2 + b = y2
kx3 + b = y3
...
kxn + b = yn
```

写成矩阵相乘的方式：
$$
\left[ \begin{array}{ccc}
x_1 & 1 \\
x_2 & 1 \\
x_3 & 1 \\
x_n & 1 \\
\end{array} \right]
\times
\left[ \begin{array}{ccc}
k \\
b \\
\end{array} \right]
=
\left[ \begin{array}{ccc}
y_1\\
y_2\\
y_3\\
y_n\\
\end{array} \right]
$$
通过np.linalg.lstsq(A, B) 求得k与b，使得所有样本点到直线的误差最小。这样找到的直线即为线性拟合得到的结果。

案例：利用线性拟合画出股价的趋势线。

趋势价（每天的最高价、最低价、收盘价的均值）

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_lstsq.py 线性拟合
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

# 计算每天的趋势价格
trend_prices = (highest_prices + \
    lowest_prices + closing_prices)/3
# 绘制每天的趋势点
mp.scatter(dates, trend_prices, s=80, 
    c='orangered', marker='o', 
    label='Trend Points')
# 线性拟合，绘制趋势线 参数：A, B
days = dates.astype('M8[D]').astype('int32')
# 把一组x坐标与一组1并在一起，构建A矩阵
A = np.column_stack((days, np.ones_like(days)))
B = trend_prices
kb = np.linalg.lstsq(A, B)[0]
# 绘制趋势线
y = kb[0] * days + kb[1]
mp.plot(dates, y, c='orangered',
    linewidth=3, label='Trend Line')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

#### 协方差、相关矩阵、相关系数

通过两组统计数据计算而得的协方差可以评估这两组统计数据的相似程度。

样本：

```
A = [a1, a2, ..., an]
B = [b1, b2, ..., bn]
```

平均值：

```
ave_A = np.mean(A)
ave_B = np.mean(B)
```

离差：

```
dev_A = [a1, a2, ..., an] - ave_A
dev_B = [b1, b2, ..., bn] - ave_b
```

协方差：

```
cov_ab = np.mean(dev_A * dev_B)
cov_ba = np.mean(dev_B * dev_A)
```

协方差可以简单的反应出两组统计样本的相关性，值为正，则正相关；值为负，则负相关。绝对值越大则相关性越强。

案例：bhp.csv,   vale.csv，评估两支股票的相关性。

```python
# 通过计算协方差，输出两支股票的相似情况
bhp_mean = np.mean(bhp_closing_prices)
vale_mean = np.mean(vale_closing_prices)
bhp_dev = bhp_closing_prices - bhp_mean
vale_dev = vale_closing_prices - vale_mean
cov_ab = np.mean(bhp_dev * vale_dev)
print(cov_ab)
```

**相关系数**

协方差除以两组统计样本标准差之积是一个[-1, 1]之间的数。该结果称为两组统计样本的相关系数。

```
若相关系数越接近于1， 表示两组样本正相关性越强。
若相关系数越接近于-1， 表示两组样本负相关性越强。
若相关系数越接近于0， 表示两组样本越不相关。
```

案例：

```python
# 输出两组样本的相关系数
k=cov_ab / (np.std(bhp_closing_prices) * \
    np.std(vale_closing_prices))
print(k)
```

**相关矩阵**

numpy提供了API可以方便的获取两组数据的相关系数。如下：

```python
m = np.corrcoef(A, B)
k = m[0,1]
```

corrcoef方法可以计算两组样本的相关系数，但是返回的m的结构为相关矩阵（2 x 2）。
$$
\left[ \begin{array}{ccc}
a与a的相关系数 & a与b的相关系数\\
b与a的相关系数 & b与b的相关系数\\
\end{array} \right]
$$
**协方差矩阵**

```python
covm = np.cov(A, B)
```

$$
\left[ \begin{array}{ccc}
a与a的协方差 & a与b的协方差\\
b与a的协方差 & b与b的协方差\\
\end{array} \right]
$$



#### 多项式拟合

多项式的一般形式：
$$
y=p_0x^n + p_1x^{n-1} +p_2x^{n-2} + ...+p_n
$$
在numpy中可以使用一组系数p<sub>0</sub> ~ p<sub>n</sub>  表示一个多项式方程。

多项式拟合的目的就是为了找到一组p<sub>0</sub> ~ p<sub>n</sub>  , 使得拟合方程尽可能与实际数据相符合。

假设拟合得到的多项式如下：
$$
f(x) = p_0x^n + p_1x^{n-1} +p_2x^{n-2} + ...+p_n
$$
则拟合的多项式函数与真实结果的误差如下表示：
$$
loss = (y_1-f(x_1))^2 + (y_2-f(x_2))^2 + ... (y_n-f(x_n))^2  
$$
那么多项式拟合的本质即是求取一组p<sub>0</sub> ~ p<sub>n</sub> 使得loss函数的值最小。

numpy提供的对多项式的操作：

```python
P = [4, 5, 2 -1]  # 可以描述一个多项式
# 把X带入多项式P，得到相应的函数值
Y = np.polyval(P, X)
# 求多项式函数的导函数  返回导函数的系数数组
Q = np.polyder(P)
# 求多项式的根
xs = np.roots(P)
# 多项式拟合  传入一组X与一组Y
P = np.polyfit(X, Y, 最高次幂)
```

案例：求多项式y=4x<sup>3</sup> + 3x<sup>2</sup> -1000x +1 曲线驻点坐标。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_poly.py 多项式操作
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-20, 20, 1000)
y = 4*x**3 + 3*x**2 - 1000*x + 1

# 求多项式函数的导函数
P = [4, 3, -1000, 1]
Q = np.polyder(P)
xs = np.roots(Q)
# 把x坐标带入原函数，求得函数值
ys = np.polyval(P, xs)
# 绘制
mp.plot(x, y)
mp.scatter(xs, ys, marker='D',
    color='red', s=60, zorder=3)
mp.show()
```

案例：基于多项式函数拟合两支股票bhp、vale的差价数组：

```python
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
```

#### 数据平滑

数据的平滑处理通常包含有降噪、拟合等操作。降噪的功能在于去除额外的影响因素，拟合的目的在于数学模型化，可以通过更多的数学方法识别曲线的特征。

案例： 绘制两支股票收益率曲线。

收益率  = （明天的收盘价-今天的收盘价）/ 今天的收盘价

```python

```













