# 数据分析DAY06

#### 符号数组

np.sign()函数可以把样本数组变成对应的符号数组，所有正数变为1，负数变为-1， 0还是0。

```python
np.sign(array)
```

**OBV能量潮（净额成交量）**

```python
# 绘制OBV能量潮
# 若相比上一天的收盘价上涨，则为正成交量
# 若相比上一天的收盘价下跌，则为负成交量
diff_prices = np.diff(closing_prices)
sign_prices = np.sign(diff_prices)
obvs = volumns[1:]
color = [('red' if x==1 else 'green') \
         for x in sign_prices]
mp.bar(dates[1:], obvs, 0.8, color=color,
    label='OBV')
```

**数组处理函数**  

```python
ary = np.piecewise(原数组，条件序列，取值序列)
```

案例：

```python
a = np.array([50, 65, 78, 82, 99])
# 数组处理，分类分数级别
r = np.piecewise(a, 
    [a>=85, (a>=60) & (a<85), a<60], 
    [1, 2, 3])
print(r)
```

#### 矢量化

一般函数只能处理标量参数，对该函数执行函数矢量化后则会具备处理数组数据的能力。

numpy提供了vectorize函数，可以把处理标量的函数矢量化，返回的函数可以直接接受矢量参数。

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_vectorize.py  矢量化
"""
import math as m
import numpy as np

def f(a, b):
    return m.sqrt(a**2 + b**2)
print(f(3, 4))

a = np.array([3, 4, 5, 6])
b = np.array([4, 5, 6, 7])
# 函数矢量化 改造f函数，使之可以处理矢量数据
# print(f(a, b))
f_vec = np.vectorize(f)  # 返回矢量函数
print(f_vec(a, b))
print(np.vectorize(f)(a, 5))
```

numpy还提供了frompyfunc函数，也可以完成矢量化：

```python
# 使用frompyfunc函数实现函数矢量化
# 2：f函数接收2个参数
# 1：f函数有1个返回值
f_fpf = np.frompyfunc(f, 2, 1)
print(f_fpf(a, b))
```

案例：定义一种买入卖出策略，通过历史数据判断这种策略是否值得实施。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_profit.py 计算无脑策略的收益
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
mp.figure('Profit', facecolor='lightgray')
mp.title('Profit', fontsize=18)
mp.xlabel('date', fontsize=14)
mp.ylabel('Profit', fontsize=14)
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


def profit(opening_price, highest_price,
    lowest_price, closing_price):
    # 定义一种买入卖出策略
    buying_price = opening_price * 0.99
    if lowest_price <= buying_price <= \
       highest_price:
        return (closing_price - buying_price)\
        / buying_price * 100
    return np.nan

# 把profit函数矢量化，求得30天的交易数据
profits=np.vectorize(profit)(opening_prices,
    highest_prices, lowest_prices, 
    closing_prices)

# 使用掩码 获取所有交易数据
nan = np.isnan(profits)
dates, profits = dates[~nan], profits[~nan]
mp.plot(dates, profits, 'o-', label='Profits')
mean = np.mean(profits)
mp.hlines(mean, dates[0], dates[-1],
    color='orangered')

mp.tight_layout()
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

### 矩阵

矩阵是numpy.matrix类型的对象。该类继承自ndarray。所以几乎所有针对ndarray数组的操作对矩阵对象同样有效。作为子类，矩阵又集合了自身特点做了必要的扩充。比如：乘法运算、求逆等。

#### 矩阵对象的创建

```python
# ary：可以被解释为矩阵的二维数据
# copy：是否复制一份新数据
m = np.matrix(ary, copy=True)
# 等价于：np.matrix(ary, copy=False)
np.mat(ary)
# 字符串拼块规则: '1 2 3; 4 5 6; 7 8 9'
np.mat(字符串拼块规则)
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_matrix.py  矩阵
"""
import numpy as np

ary = np.arange(1, 10).reshape(3, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=True)
print(m, type(m))
ary[0,0] = 999
print(m, type(m))

m = np.mat('1 2 3; 5 6 4')
print(m)
```

#### 矩阵的乘法运算

```python
# 测试矩阵的乘法
print('-'*45)
print(m)
print(m * m)
print(m.dot(m))
```

#### 矩阵的逆矩阵

若两个矩阵A、B满足：A*B=E  （E为单位矩阵），则称B为A的逆矩阵。

```python
m = np.mat('...')
print(m.I)	# 输出m的逆
print(np.linalg.inv(m))  # 输出m的逆  针对方阵
```

案例：

```python
# 测试矩阵的逆矩阵
b = np.mat('4 6 7; 2 3 7; 8 4 2')
print(b)
print(np.linalg.inv(b))
print(b * b.I)

c = np.mat('4 6 7; 2 3 7')
print(c)
print(c.I)
print(c * c.I)
```

案例：假设学校旅游：去程小孩票价3元，家长票价3.2， 共花了118.4； 回来时小孩票价3.5， 家长票价3.6，共花了135.2。 求小孩和家长的人数。
$$
\left[ \begin{array}{ccc}
3 & 3.2 \\
3.5 & 3.6 \\
\end{array} \right]
\times
\left[ \begin{array}{ccc}
x\\
y\\
\end{array} \right]
=
\left[ \begin{array}{ccc}
118.4\\
135.2\\
\end{array} \right]
$$

```python
# 测试矩阵的逆矩阵
b = np.mat('4 6 7; 2 3 7; 8 4 2')
print(b)
print(np.linalg.inv(b))
print(b * b.I)

c = np.mat('4 6 7; 2 3 7')
print(c)
print(c.I)
print(c * c.I)
```



案例：斐波那契数列

```
1 1 2 3 5 8 13 ...

	 1 1   1 1   1 1
	 1 0   1 0   1 0
-------------------------------------
1 1  2 1   3 2   5 3
1 0  1 1   2 1   3 2    ....

m^1  m^2   m^3   m^4
```

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_matrix.py  矩阵
"""
import numpy as np

ary = np.arange(1, 10).reshape(3, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=True)
print(m, type(m))
ary[0,0] = 999
print(m, type(m))

m2 = np.mat('1 2 3; 5 6 4')
print(m2)

# 测试矩阵的乘法
print('-'*45)
print(m)
print(m * m)

# 测试矩阵的逆矩阵
b = np.mat('4 6 7; 2 3 7; 8 4 2')
print(b)
print(np.linalg.inv(b))
print(b * b.I)

c = np.mat('4 6 7; 2 3 7')
print(c)
print(c.I)
print(c * c.I)

# 解应用题

prices = np.mat('3 3.2; 3.5 3.6')
totals = np.mat('118.4; 135.2')
x = np.linalg.lstsq(prices, totals)[0]
print(x)

persons = prices.I * totals
print(persons)

# 测试
m = np.mat('1 1; 1 0')
print(m**20)
```

### 通用函数

#### 裁剪与压缩

```python
ndarray.clip(min=下限, max=上限)
ndarray.compress(条件)
```

案例：

```python
import numpy as np

a = np.arange(1, 11)
print(a.clip(min=4, max=8))
print(a.compress(np.all([(a<8), (a>3)], axis=0)))
```

#### 加法与乘法相关通用函数

```python
np.add(a, b)    # a + b
np.add.reduce(a)  # 求a数组所有元素的累加和
np.add.accumulate(a) # 求a数组所有元素累加和的过程
np.add.outer([10,20,30], a)  # 外和
np.prod(a)   # 求a数组所有元素的累乘
np.cumprod(a)   # 求a数组所有元素累乘的过程
np.outer([10,20,30], a) # 外积
```

#### 除法与取整通用函数

```python
np.divide(a, b)   # a/b
np.floor_divide(a, b)  # 地板除
np.ceil(a)  # 天花板取整
np.round(a)  # 四舍五入
np.trunc(a)   # 截断取整    truncate table xxxxx
```

#### 位运算通用函数

**位异或**

```
c = a ^ b
c = np.bitwise_xor(a, b)
```

案例：判断两个数据是否同号。

```
-8  1000
-7  1001
-6  1010
-5  1011
-4  1100
-3  1101
-2  1110
-1  1111
0   0000
1   0001
2   0010
3   0011
4   0100
5   0101
6   0110
7   0111
```

```python
a = np.array([0, -1, 2, 3, 4, -5])
b = np.array([1, 1, 2, 3, 4, 5])
c = a ^ b
print(c)
print(np.where(c < 0))
```

**位与、位或、取反、移位**

```python
np.bitwise_and(a, b)
np.bitwise_or(a, b)
np.bitwise_not(a)
np.left_shift(a, 2)
np.right_shift(a, 2)
```

利用位与计算某个数字是否是2的幂。

```
1 2 4 8 16 32 64 ...

1  2^0  00001      0  00000
2  2^1  00010      1  00001
4  2^2  00100      3  00011
8  2&3  01000      7  00111
16 2^4  10000     15  01111

if a & (a-1) == 0:
   True
```

#### 三角函数通用函数

```
np.sin()  np.cos() 
```

**傅里叶定理**

任何一个曲线，无论多么跳跃或不规则，都可以被解析成一组光滑正弦函数的叠加。

**合成方波**
$$
y = \frac{4\pi}{2n-1} \times sin((2n-1)x)
$$
案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_sin.py   合成方波
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y1 = 4*np.pi * np.sin(x)
y2 = 4*np.pi/3 * np.sin(3*x)

n = 1000
y = np.zeros(1000)
for i in range(1, n+1):
    y += 4*np.pi/(2*i-1) * np.sin((2*i-1)*x)

mp.plot(x, y1, label='y1', alpha=0.3)
mp.plot(x, y2, label='y2', alpha=0.3)
mp.plot(x, y, label='y')
mp.legend()
mp.show()
```

























