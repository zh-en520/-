# 数据分析DAY06

#### 符号数组

np.sign()函数可以把样本数组变为对应的符号数组，正数为1， 负数为-1，0则为0.

```
ary = np.sign(原数组)
```

净额成交量（OBV能量潮）

```python
# 得到从第二天开始每天的涨跌值
diff_prices = np.diff(closing_prices)
sign_prices = np.sign(diff_prices)
print(sign_prices)

# 绘制OBV图
dates = dates.astype(md.datetime.datetime)
dates = dates[1:]
volumns = volumns[1:]
mp.bar(dates[sign_prices==1], 
	   volumns[sign_prices==1], 
	   0.8, color='red')
mp.bar(dates[sign_prices==-1], 
	   volumns[sign_prices==-1], 
	   0.8, color='green')
```

**数组处理函数**

```python
a=np.piecewise(原数组，条件序列，取值序列)
```

针对原数组中每一个元素，检测其是否符合条件序列中的某个条件，符合哪一个条件就使用取值序列中与之对应的值表示该元素，放到结果数组返回。

```python
sign_prices = np.piecewise(
    diff_prices, 
	[diff_prices>0, 
     diff_prices<0, 
     diff_prices==0],
	[1, -1, 0])
```

#### 函数矢量化

矢量化指的是用数组代替标量来操作数组里的每个元素。

numpy提供了vectorize函数，可以把处理标量的普通函数矢量化，返回一个矢量函数，该矢量函数可以直接处理数组并返回数组。

```python
def f(a, b):
    pass
f2 = np.vectorize(f)
```

案例：

```python
"""
demo02_vec.py  函数矢量化
"""
import numpy as np
import math as m

def f (a, b):
	return m.sqrt(a**2 + b**2)

print(f(3, 4))
a = np.array([3,4,5,6])
b = np.array([4,5,6,7])
# print(f(a, b))   报错 
# 矢量化过后的函数  f_vec
f_vec = np.vectorize(f)
print(f_vec(a, b))
print(f_vec(a, 5))

# np.frompyfunc() 也可以实现函数矢量化
f_fpf = np.frompyfunc(f, 2, 1)
print(f_fpf(a, b))
```

frompyfunc函数也可以实现函数矢量化：

```python
# np.frompyfunc() 也可以实现函数矢量化
# 2： f函数接收2个参数
# 1： f函数返回1个返回值
f_fpf = np.frompyfunc(f, 2, 1)
print(f_fpf(a, b))
```

案例：自定义一种买入卖出策略，通过历史数据判断这种策略是否值得实施。

```python
"""
demo03_profit.py  
定义一种买入卖出策略 验证是否有效
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

print(dates, dates.dtype)
#绘制收盘价的折线图

mp.figure('Profits', facecolor='lightgray')
mp.title('Profits', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Profit', fontsize=12)
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

def profit(opening_price, highest_price,
		lowest_price, closing_price):
	'''
	定义一种投资策略 
	开盘价*0.99倍买入， 收盘价卖出
	'''
	buying_price = opening_price*0.99
	if lowest_price<=buying_price<=highest_price:
		return (closing_price-buying_price)\
				/ buying_price

	return np.nan  # 无效值

# 矢量化profit函数，求得每天的收益率
profits=np.vectorize(profit)(opening_prices,
	highest_prices, lowest_prices,
	closing_prices)
print(profits)
# 判断profits中每个元素是否是nan
nan = np.isnan(profits)
dates, profits = dates[~nan], profits[~nan]

dates = dates.astype(md.datetime.datetime)
mp.plot(dates, profits, 'o-', c='dodgerblue',
	linewidth=2, label='Profits')
m = np.mean(profits)
mp.hlines(m, dates[0], dates[-1], 
	color='orangered', label='Mean(Profits)')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

#### 矩阵

矩阵是numpy.matrix类型的对象。该类继承自ndarray,  所以任何针对多维数组的操作对矩阵对象同样有效。但是作为子类，矩阵又结合其自身的特点，做了必要的扩充，比如：乘法计算、求逆等。

**矩阵对象的创建**

```python
# ary: 任何可以被解释为矩阵的二维容器
# copy: 创建的矩阵对象是否复制一份新数据
np.matrix(ary, copy=True)
# ary: 任何可以被解释为矩阵的二维容器
# 默认copy=False
np.mat(ary)
# 字符串拼块规则构建矩阵对象
np.mat('1 2 3; 4 5 6; 6 7 8')
```

案例：

```python
"""
demo04_matrix.py  矩阵示例
"""
import numpy as np

ary = np.arange(1, 7).reshape(2, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=True)
print(m, type(m))
ary[0,0] = 999
print(ary, type(ary))
print(m, type(m))


# 字符串拼块规则 
ary = np.mat('1 2 3; 4 5 6; 7 8 9') 
print(ary, type(ary), ary.dtype)
```

**矩阵的乘法运算**

若A x B： 结果矩阵的第i行第j列的元素等于：

A矩阵的第i行与B矩阵的第j列的点积。

```python
# 字符串拼块规则 
ary = np.mat('1 2 3; 4 5 6; 7 8 9') 
print(ary, type(ary), ary.dtype)
b = np.arange(1, 10).reshape(3, 3)
print(b.dot(b))
```

**矩阵的逆矩阵**

若两个矩阵A、B满足 A x B=E （E为单位矩阵），则称B为A矩阵的逆矩阵。

```python
matrix=np.mat(...)
# 1. 
matrix.I
# 2.
np.linalg.inv(matrix)
```

案例：

```python
# 测试矩阵的逆矩阵
a = np.mat('1 2 3; 4 4 3; 2 3 6')
#a = np.mat('1 2 3; 4 5 6; 7 8 9')
print('-' * 45)
print(a)
print(a.I)
print(a * a.I)
print(a.I * a)
```

一般人旅游，去程坐大巴，小孩3元/位，大人3.2/位，花了118.4； 回来的时候坐火车，小孩3.5/位， 大人3.6/位， 花了135.2； 求出大人小孩的人数。
$$
x: 小孩  y：大人 \\
\left[ 
\begin{array}{c}
3 & 3.2 \\
3.5 & 3.6 \\
\end{array}
\right]
\times
\left[ 
\begin{array}{c}
x \\
y \\
\end{array}
\right]
=
\left[ 
\begin{array}{c}
118.4 \\
135.2 \\
\end{array}
\right]
$$
**斐波那契数列**

1  1  2   3   5   8   13  ....

```
     1 1   1 1   1 1
     1 0   1 0   1 0
-----------------------------------
1 1  2 1   3 2   5 3
1 0  1 1   2 1   3 2   ....

m^1  m^2   m^3   m^4   ....
```

案例：

```python
"""
demo04_matrix.py  矩阵示例
"""
import numpy as np

ary = np.arange(1, 7).reshape(2, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=True)
print(m, type(m))
ary[0,0] = 999
print(ary, type(ary))
print(m, type(m))

# 字符串拼块规则 
ary = np.mat('1 2 3; 4 5 6; 7 8 9') 
print(ary, type(ary), ary.dtype)
b = np.arange(1, 10).reshape(3, 3)
print(b.dot(b))

# 测试矩阵的逆矩阵
a = np.mat('1 2 3; 4 4 3; 2 3 6')
#a = np.mat('1 2 3; 4 5 6; 7 8 9')
print('-' * 45)
print(a)
print(a.I)
print(a * a.I)
print(a.I * a)


print('-' * 45)
A = np.mat('3 3.2; 3.5 3.6')
B = np.mat('118.4; 135.2')
x = np.linalg.lstsq(A, B)[0]
print(x)
persons = A.I * B
print(persons)

n = 32
m = np.mat('1 1; 1 0')
print((m**n)[0,0])
```

### numpy通用函数

#### 加法与乘法的通用函数

```python
np.add(a, a)  # 加法  a + a
np.add.reduce(a)  # 返回a的累加和 
np.add.accumulate(a) # 返回累加和的过程
np.add.outer([10,20,30], a) # 外和

np.prod(a)  # 累乘
np.cumprod(a)  # 返回累乘的过程
np.outer([10,20,30], a) # 外积
```

案例：

```python
"""
demo05_add.py  通用函数
"""
import numpy as np
a = np.arange(1, 7)
print(a)
print(np.add(a, a))
print(np.add.reduce(a))
print(np.add.accumulate(a))
print(np.add.outer([10,20,30], a))

print(np.prod(a))
print(np.cumprod(a))
print(np.outer([10,20,30], a))
```

#### 除法及取整通用函数

```python
np.divide(a, b)  
np.true_divide(a, b)
np.floor_divide(a, b)

np.ceil(a / b)
np.floor(a / b)
np.round(a / b)
np.trunc(a / b)
```

```python
"""
demo06_divide.py  通用函数
"""
import numpy as np

a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])

print(np.divide(a, b))
print(np.floor_divide(a, b))
print(np.ceil(a / b))
print(np.floor(a / b))
print(np.round(a / b))
print(np.trunc(a / b))  # 截断除
```

#### 位运算通用函数

**位异或：**

```
c = a ^ b
c = np.bitwise_xor(a, b)
```

按位异或操作可以方便的判断两个数据是否同号：

```
-8	1000
-7	1001
-6	1010
-5	1011
-4	1100
-3	1101
-2	1110
-1	1111
0	0000
1	0001
2	0010
3	0011		
4	0100
5	0101
6	0110
7	0111
```

**位与：**

```python
e = a & b
e = np.bitwise_and(a, b)
```

利用位与运算计算某个数是否是2的幂：

```python
'''
1	2^0	00001		0	00000
2	2^1 00010		1	00001
4	2^2 00100		3	00011
8	2^3 01000		7	00111
16	2^4 10000		15	01111
...

'''
# 1000以内2的幂
a = np.arange(1, 1000)
mask = np.bitwise_and(a, a-1)==0
print(a[mask])
```

**其他位运算通用函数**

```python
np.bitwise_or(a, b)  # 或运算
np.bitwise_not(a)	# 取反
np.left_shift(a, 1)   # a左移1位
np.right_shift(a, 1)	# a右移1位
```

#### 三角函数通用函数

```python
np.sin()
```

**傅里叶定理**

法国科学家傅里叶说过：任何一个周期曲线，无论多么跳跃或不规则，都可以被看做一组不同振幅、频率、相位的正弦函数叠加而成。











