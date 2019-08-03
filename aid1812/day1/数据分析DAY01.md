# 数据分析DAY01

## 徐铭     xuming@tedu.cn

## 15201603213

**什么是数据分析？**

数据分析是指用适当的统计分析方法对收集来的大量数据进行分析，提取有用的信息并对其加以详细研究最后得出分析结论的过程。

**使用python做数据分析的常用库**

1. numpy  基础数值算法
2. scipy      科学计算
3. matplotlib   数据可视化
4. pandas   提供更多的序列高级函数

## numpy概述

1. Numerical Python. 补充了python所欠缺的数值计算能力。
2. Numpy是其他数据分析及机器学习库的底层库。
3. Numpy完全标准C语言实现，运行效率充分优化。
4. Numpy开源免费。

## numpy基础

### ndarray数组

```python
import numpy as np

ary = np.array([1, 2, 3, 4, 5])
print(ary, type(ary))

print(ary + ary)
print(ary * ary)
print(ary > 3)
```

**内存中的ndarray对象**

**元数据（metadata）**

存储对目标数组的描述信息，如：dim count、 dtype、data、shape等。

**实际数据**

完整的数组数据

将实际数据与元数据分开存放，一方面提高了内存空间的使用效率，另一方面减少对实际数据的访问频率，提高性能。

#### ndarray数组对象的创建

```python
np.array([[],[],[]])
# 从0到10， 步长为1
np.arange(0, 10, 1)
np.zeros(10)
np.ones(10)
np.zeros_like(ary)
```

```python
"""
demo02_ndarray.py  ndarray示例
"""
import numpy as np

a = np.array([1,2,3,4,5,6])
b = np.arange(7, 13, 1)
print(a)
print(b)
c = np.zeros(6)
d = np.ones(6)
print(c)
print(d / 5)
e = np.array([[1, 2, 3], [4, 5, 6]])
print(e)
print(np.ones_like(e))
```

#### ndarray对象属性的基本操作

**数组的维度：** ary.shape

**元素的类型：**  ary.dtype

**数组元素的个数：**  ary.size

**数组元素的索引：**  ary[0]

```python
"""
demo03_attr.py 属性操作
"""
import numpy as np

# 数组维度处理
ary = np.array([1, 2, 3, 4, 5, 6])
print(ary, ary.shape)
print(ary[4])
ary.shape = (2, 3)
print(ary, ary.shape)
print(ary[1][1])

# 数组元素类型
print('-'*45)
print(ary, ary.dtype)
# ary.dtype = np.int64
# print(ary, ary.dtype)
b = ary.astype('float32')
print(b, b.dtype)

#数组元素的个数
print('-'*45)
print(ary.size)
print(len(ary))

# 数组元素的索引
print('-'*45)
ary = np.arange(1, 9)
ary.shape=(2, 2, 2)
print(ary, ary.shape)
print(ary[0])   # 0页数据
print(ary[0][0])   # 0页0行数据
print(ary[0][0][0])   # 0页0行0列数据
print(ary[0, 0, 0])   # 0页0行0列数据

# 使用for循环，把ary数组中的元素都遍历出来。
for i in range(ary.shape[0]):
    for j in range(ary.shape[1]):
        for k in range(ary.shape[2]):
            print(ary[i, j, k], end=' ')
```

#### ndarray对象属性操作详解

**Numpy的内部基本数据类型**

| 类型名     | 类型表示符                |
| ---------- | ------------------------- |
| 布尔类型   | bool_                     |
| 有符号整型 | int8/16/32/64             |
| 无符号整型 | uint8/16/32/64            |
| 浮点型     | float16/32/64             |
| 复数型     | complex64/128             |
| 字符串型   | str_，每个字符32位Unicode |

**自定义复合类型**

```python
"""
demo04_dtype.py
"""
import numpy as np

data = [('zs', [90, 80, 70], 15),
        ('ls', [99, 89, 79], 16),
        ('ww', [91, 81, 71], 17)]

# 2个Unicode字符，3个int32，1个int32组成的元组
ary = np.array(data, dtype='U2, 3int32, int32')
print(ary, ary.dtype)
print(ary[1][2], ary[1]['f2'])

# 第二种设置dtype的方式  为每个字段起别名
ary = np.array(data, dtype=[('name', 'str_', 2),
                            ('scores', 'int32', 3),
                            ('age', 'int32', 1)])
print(ary[2]['age'])

# 第三种设置dtype的方式
ary = np.array(data, dtype={
        'names' : ['name', 'scores', 'age'],
        'formats' : ['U2', '3int32', 'int32']})
print(ary[2]['scores'])

# 第四种设置dtype的方式 手动指定每个字段的存储偏移字节数
# name从0字节开始输出，输出3个Unicode
# scores从16字节开始输出，输出3个int32
ary = np.array(data, dtype={
        'name' : ('U3', 0),
        'scores' : ('3int32', 16),
        'age' : ('int32', 28)})

print(ary[0]['name'])

# ndarray数组中存储日期类型数据
dates = np.array(['2011', '2012-01-01',
        '2013-01-01 11:11:11', '2011-02-01'])
print(dates, dates.dtype)
dates = dates.astype('M8[D]') # datetime64精确到Day
print(dates, dates.dtype)
print(dates[-1] - dates[0])

dates = dates.astype('int32')
print(dates, dates.dtype)
```

**类型字符码**

| 类型           | 字符码          |
| -------------- | --------------- |
| bool_          | ?               |
| int8/16/32/64  | i1/2/4/8        |
| uint8/16/32/64 | u1/2/4/8        |
| float16/32/64  | f2/4/8          |
| complex64/128  | c8/16           |
| str_           | U               |
| datetime64     | M8[Y/M/D/h/m/s] |

#### ndarray对象的维度操作详解

视图变维：reshape()    ravel()

```python
# 测试视图变维 reshape()  ravel()
b = a.reshape(2, 3)
print(b, b.shape)
a[0] = 999
print(b, b.shape)
c = b.ravel()
print(c, c.shape)
```



复制变维：flatten()

```python
# 复制变维  flatten()   copy()
d = b.flatten()  # 扁平化
print(b, b.shape)
print(d, d.shape)
d[0] = 1
print(b, b.shape)
print(d, d.shape)
```



就地变维：shape    resize()

```python
# 就地变维
print(b, b.shape)
b.resize(3, 2)
print(b, b.shape)

```

**ndarray对象索引操作详解**

**ndarray数组的切片**

```python
a = np.arange(1, 10)
print(a[:3])
print(a[3:6])
print(a[6:])
print(a[::-1])
print(a[:-4:-1])
print(a[::])
print(a[::3])
print(a[1::3])
print(a[2::3])


# 针对高维数组的切片
a = a.reshape(3, 3)
print(a, a.shape)
print(a[:2, :2])
print(a[:2, 0])
```

**ndarray数组的掩码操作**

```python
"""
demo07_mask.py  掩码操作
"""
import numpy as np

a = np.arange(1, 8)
mask = a > 5
print(a)
print(mask)
print(a[mask])

# 输出100以内3与7的公倍数。
a = np.arange(1, 100)
mask = (a%3==0) & (a%7==0)
print(a[mask])

# 利用掩码运算对数组进行排序
p = np.array(['Mi', 'Apple', 'Huawei', 'Oppo'])
r = [1, 3, 2, 0]
print(p[r])
```

**多维数组的组合与拆分**

垂直方向操作：

```python
# 垂直方向合并
c = np.vstack((a, b))
# 把c拆成2份， a与b
a, b = np.vsplit(c, 2)
```

水平方向操作：

```python
# 水平方向合并
c = np.hstack((a, b))
# 把c拆成2份， a与b
a, b = np.hsplit(c, 2)
```

深度方向操作：

```python
# 深度方向合并
c = np.dstack((a, b))
# 把c拆成2份， a与b
a, b = np.dsplit(c, 2)
```

多维数组组合与拆分相关函数：

```python
# 把a与b按照axis的轴向进行组合
# axis 数组组合的轴向
#   0：垂直   1：水平   2：深度
# 注意：若axis=2，则要求a与b都是3维数组
c = np.concatenate((a, b), axis=0)
# 把c按照axis的轴向拆成2部分
a, b = np.split(c, 2, axis=0)
```

**简单的一维数组的组合方案**

```python
# 把两个一维数组摞在一起成两行
c = np.row_stack((a, b))
# 把两个一维数组并在一起成两列
c = np.column_stack((a, b))
```

#### ndarray的其他属性

1. shape, dtype, size ....
2. ndim   维数    n维数组的那个n
3. itemsize   每个元素的字节数
4. nbytes     数组占用内存的总字节数
5. real    复数数组的数据的实部
6. imag   复数数组的数据的虚部
7. T    返回数组的转置视图
8. flat   返回数组的扁平迭代器

























