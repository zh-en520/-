# 数据分析DAY01

**徐铭   xuming@tedu.cn   15201603213**



## 什么是数据分析

数据分析是指用适当的统计分析方法对收集来的大量数据进行业务分析，从而提取有用的数据形成结论，加以详细研究和概括总结的过程。

### 数据分析所使用的常用库

1. numpy   基础的数值运算
2. scipy       专注科学计算
3. matplotlib    数据可视化
4. pandas   提供了高级序列函数

## Numpy概述

1. 补充了Python语言所欠缺的数值运算能力。
2. numpy是其它机器学习库的底层库。
3. numpy完全标准C语言实现，运行效率充分优化。
4. 开源免费。

### numpy的核心

**多维数组 numpy.ndarray**

```python
# ndarray数组
ary = np.array([1, 2, 3, 4, 5])
print(ary, type(ary))

ary2 = np.array([2, 2, 4, 4, 5])
print(ary2, type(ary2))

ary3 = ary * 3
print(ary3, type(ary3))
```

#### 内存中的ndarray对象

**元数据（metadata）**

存储对目标数组的描述信息，如：dim count, dtype, data等。

**实际数据**

完整的数组数据。

将元数据与实际数据分开存放，一方面提高了内存空间的使用效率，另一方面减少对实际数据的访问频率，提高访问性能。

#### ndarray数组对象的创建

```python
#p: 任何可被解释为多维数组的逻辑结构
np.array(p)
# 与range()的参数一致  (0, 10, 1)
np.arange(起始值，终止值，步长)
# 构建一个全为0的数组
np.zeros(size)
# 构建一个全为1的数组
np.ones(size)
# 构建一个全为0的数组, 维度与ary保持一致
np.zeros_like(ary)
np.ones_like(ary)
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
demo02_ndarray.py  测试数组创建
'''
import numpy as np

a = np.arange(0, 5, 1)
b = np.arange(0, 10, 2)
print(a, b)

z = np.zeros(10)
print(z)
o = np.ones(10)
print(o)

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)
print(np.zeros_like(a))

print(np.ones(5) / 5)
```

#### ndarray对象属性的基本操作

**数组的维度：ndarray.shape**

```python
# 测试shape属性： 维度
ary = np.array([1, 2, 3, 4, 5])
print(type(ary), ary.shape)
ary = np.array([[1, 2, 3], [4, 5, 6]])
print(ary, ary.shape)

ary.shape = (3, 2)
print(ary, ary.shape)
```

**元素的类型：ndarray.dtype**

```python
# 测试dtype属性：元素类型
print('---------------------------')
print(ary, ary.dtype)
#ary.dtype = 'float32'  # 该类型不是这么改的
#print(ary, ary.dtype)  
ary2 = ary.astype('float64')
print(ary2, ary2.dtype)
```

**数组元素个数：ndarray.size    len(ndarray)**

```python
# 测试size属性： 元素个数
print('---------------------------')
print(ary, ary.size, len(ary), ary.shape[1])
```

**数组的索引：ary[0]**

```python
# 测试索引属性
print('---------------------------')
a = np.arange(1, 9)
a.shape = (2, 2, 2)
print(a)
print(a[1])
print(a[1][0])
print(a[1][0][0], a[1, 0, 0])

# 尝试使用for循环，遍历所有元素 并输出
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            print(a[i, j, k], end=' ')
```

#### ndarray对象属性操作详解

**Numpy的内部基本数据类型**

| 类型名     | 类型表示符                           |
| ---------- | ------------------------------------ |
| 布尔型     | bool_                                |
| 有符号整型 | int8/int16/int32/int64               |
| 无符号整型 | uint8/uint16/uint32/uint64           |
| 浮点型     | float16/float32/float64              |
| 复数型     | complex64/complex128                 |
| 字串型     | str_,  每个字符用32位Unicode编码表示 |

```python
np.array([1,2,3,0], dtype='bool_')
```

**自定义复合类型**

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_types.py 自定义复合类型
"""
import numpy as np

data = [
    ('zs', [60, 61, 65], 15),
    ('ls', [62, 66, 69], 16),
    ('ww', [64, 68, 63], 17)]

# 第四种设置dtype的方式
d = np.array(data, dtype={
    'name': ('U3', 0),
    'scores': ('3int32', 16),
    'age': ('int32', 28)})
print(d[0]['age'])


# 第三种设置dtype的方式
c = np.array(data, dtype={
    'names': ['name', 'scores', 'age'],
    'formats': ['U2', '3int32', 'int32']})
print(c, c[1]['name'])


# 第二种设置dtype的方式
b = np.array(data, dtype=[
    ('name', 'str', 2),
    ('scores', 'int32', 3),
    ('age', 'int32', 1) ])
print(b, b[1]['scores'])


# 第一种设置dtype的方式
a = np.array(data, dtype='U2, 3i4, i4')
print(a, a[2]['f0'])

print('-'*45)
# 测试日期类型数组  
a = np.array(['2011', '2012-01-01',
    '2011-01-01 10:10:10', '2011-02-01'])
print(a.dtype)
b = a.astype('M8[Y]')
print(b, b.dtype)
# 日期的减法
print(b[1] - b[0])
```

**类型字符码**

| 类型           | 字符码          |
| -------------- | --------------- |
| bool_          | ?               |
| int8/16/32/64  | i1/i2/i4/i8     |
| uint8/16/32/64 | u1/u2/u4/u8     |
| complex64/128  | c8/c16          |
| str_           | U<字符数>       |
| datetime64     | M8[Y/M/D/h/m/s] |



**ndarray数组的维度操作**

视图变维: a.reshape()  a.ravel()    **数据共享**

复制变维: a.flatten()    **数据独立**

就地变维: a.resize()   a.shape    **改变自己**

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_shape.py  维度处理
"""
import numpy as np

a = np.arange(1, 9)
print(a)
b = a.reshape(2, 4)  # 变维 2 * 4
print(b)
c = a.reshape(2, 2, 2)  # 变维 2 * 2 * 2
print(c)

a[0] = 999
print(a)
print(b)
print(c.ravel())

# 复制变维
d = a.flatten()  
print(a)
print(d)
d[0] = 888
print(a)
print(d)

# 就地变维
print('-'*45)
print(c)
c.resize(2, 4)
print(c)
```



**ndarray数组的切片操作**

```python
a = np.arange(1, 10)
print(a)
print(a[:3])
print(a[3:6])
print(a[6:])
print(a[::-1])
print(a[:-4:-1])
print(a[::])
print(a[::3])
print(a[1::3])
print(a[2::3])

# 多维数组的切片
b = a.reshape(3, 3)
print(b)
# 前两行的前两列
print(b[:2, :2])
```



**ndarray数组的掩码操作**

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_mask.py 掩码操作
"""
import numpy as np

a = np.arange(1, 7)
mask = a % 2 == 0 
b = a[mask]
print('a:', a)
print('mask:', mask)
print('b:', b)

# 输出100以内 3与7的公倍数
a = np.arange(1, 100)
mask = (a%3==0) & (a%7==0)
print(a[mask])

# 基于索引的掩码操作
a = np.array(['A', 'B', 'C', 'D', 'E'])
mask = [0, 3, 4, 3, 4, 2, 4, 1, 2, 3, 1]
print(a[mask])
```



**ndarray多维数组的组合与拆分**

垂直方向的操作：

```python
# 垂直方向，把a与b摞在一起
c = np.vstack((a, b))
a, b = np.vsplit(c, 2)
```

水平方向的操作：

```python
# 水平方向，把a与b并在一起
c = np.hstack((a, b))
a, b = np.hsplit(c, 2)
```

深度方向的操作：

```python
# 深度方向，把a与b叠在一起
c = np.dstack((a, b))
a, b = np.dsplit(c, 2)
```

多维数组的组合与拆分系相关函数：

```python
# 数组的组合   根据axis轴向对a与b数组进行合并
# axis=0: 垂直
# axis=1: 水平
# axis=2: 深度  （a与b都为三维数组才可以深度方向合并）
c = np.concatenate((a, b), axis=0)
# 数组的拆分
a, b = np.split(c, 2, axis=0)
```



简单的**一维数组**的组合方案：

```python
# 把两个一维数组a与b，摞在一起成为两行
np.row_stack((a, b))
# 把两个一维数组a与b，并在一起成为两列
np.column_stack((a, b))
```



#### ndarray的其他属性

1. ndim           维数
2. itemsize     元素的字节数
3. nbytes        对象占用的总字节数
4. real             复数的实部
5. imag           复数的虚部
6. T                  二维数组的转置视图
7. flat              返回数组的扁平迭代器













