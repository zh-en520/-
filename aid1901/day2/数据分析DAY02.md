# 数据分析DAY02

## matplotlib概述

matplotlib是python的一个绘图库，方便的绘制出版质量级别的图像。

## matplotlib基本功能

1. 基本绘图
   1. 绘制简单图形，线型、线宽、颜色。
   2. 处理坐标轴。
   3. 图例。
   4. 特殊点。
   5. 备注。
2. 高级绘图
   1. 操作图形窗口。子图。
   2. 刻度定位器。刻度网格线。
   3. 半对数坐标。
   4. 散点图、条形图、填充图、饼状图
   5. 等高线图、热成像图。
   6. 极坐标系。
   7. 绘制三维图形。
   8. 简单动画。



## matplotlib基本功能详解

### 基本绘图

**绘图核心API：**

```python
import matplotlib.pyplot as mp
mp.plot(xarray, yarray)
mp.show()
```

绘制水平线与垂直线：

```python
# 垂直线  val：x坐标值   xmin，xmax：x轴的区间
mp.vlines(val, xmin, xmax)
# 水平线  val：y坐标值   ymin，ymax：y轴的区间
mp.hlines(val, ymin, ymax)
```

绘制正弦曲线：y=sin(x)

```python
# 绘制曲线，设置线型、线宽、颜色
mp.plot(x, y, linestyle='--',
        linewidth=3, color='dodgerblue',
        alpha=0.7)
mp.show()
```

#### 线型 线宽 颜色

```python
# 绘制曲线，设置线型、线宽、颜色
# linestyle: 线型：'-'  '--'  ':' 
# color: 颜色的英文单词 或 常见单词首字母 或
#        #abcdab 或 (1, 1, 0.9) 或 (1, 1, 1, 1)
# alpha: 透明度
mp.plot(x, y, linestyle='--',
        linewidth=3, color='dodgerblue',
        alpha=0.7)
mp.show()
```



#### 设置坐标轴范围

```python
# 把x轴限制在[x_min, x_max]
mp.xlim(x_min, x_max)
# 把y轴限制在[y_min, y_max]
mp.ylim(y_min, y_max)
```



#### 设置坐标刻度

```python
# 设置水平轴刻度值与文本
# val_list: 刻度值的列表
# test_list: 刻度值的文本列表
mp.xticks(val_list, text_list)
# 设置垂直轴刻度值与文本
mp.yticks(val_list, text_list)
```

**刻度文本的特殊语法：laTeX语法**
$$
x^2 + y^2 = z^2 \qquad \sqrt{2}
$$


#### 设置坐标轴

```python
# 获取当前坐标系
ax = mp.gca()
# 获取坐标系中的某一个轴
axis = ax.spines['top']   # bottom  left  right
# 设置坐标轴的颜色
axis.set_color('none')
# 设置坐标轴的位置
# data: 设置坐标轴位置时的参照系  0:坐标值
axis.set_position(('data', 0))
```

```python
# 设置坐标轴
ax = mp.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
```



#### 图例

```python
# 设置当前曲线的标签文本
mp.plot(......., label='y=sin(x)')
# 自动显示图例
'''
 |              ===============   =============
 |              Location String   Location Code
 |              ===============   =============
 |              'best'            0
 |              'upper right'     1
 |              'upper left'      2
 |              'lower left'      3
 |              'lower right'     4
 |              'right'           5
 |              'center left'     6
 |              'center right'    7
 |              'lower center'    8
 |              'upper center'    9
 |              'center'          10
 |              ===============   =============
'''
mp.legend()
```



#### 特殊点

```python
mp.scatter(xarray, yarray,
          marker='',    # 点型
          s=60,			# 大小
          edgecolor='', # 边缘色
          facecolor='', # 填充色
          color='',     # 颜色
          zorder=3		# 图层编号
)
```



```python
# 绘制特殊点
mp.scatter([np.pi/2, np.pi/2], [0, 1],
    marker='o', s=80, facecolor='steelblue',
    edgecolor='red', zorder=3)
```



#### 备注

```python
mp.annotate(
	r'.....', 			# 备注文本
    xycoords='data',    # 目标点坐标
    xy=(1, 2),
    textcoords='offset points',  # 文本坐标
    xytext=(1, 2),
    fontsize=10,
    arrowprops=dict(
    	arrowstyle='->',			# 箭头样式
        connectionstyle='angle3'    # 连接线样式
    )
)
```



### 高级绘图

#### 窗口操作

mp.figure方法的绘制原则如下：

```python
mp.figure('titleA', facecolor='lightgray')
mp.plot() # 针对titleA窗口进行绘制
mp.figure('titleB', facecolor='lightgray')
mp.plot() # 针对titleB窗口进行绘制
mp.figure('titleA') # 把titleA置为当前窗口 
mp.plot() # 针对titleA窗口进行绘制
mp.show()
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_figure.py 测试窗口
"""
import matplotlib.pyplot as mp
mp.figure('FigureA', facecolor='lightgray')

mp.figure('FigureB', facecolor='gray')
mp.title('Figure B', fontsize=18) # 图表标题
mp.xlabel('X label', fontsize=14) # x标签文本
mp.ylabel('Y label', fontsize=14) # y标签文本
mp.tick_params(labelsize=10) # 刻度参数  字体大小
mp.grid(linestyle='-') # 网格线
mp.tight_layout() # 紧凑布局
mp.show()
```

#### 子图

matplotlib的子图支持在一个窗口中绘制多张图表。

**矩阵式布局**

```python
mp.figure('')
# 绘制子图(行，列，编号)
mp.subplot(row, col, num)
mp.subplot(3, 3, 1)
mp.subplot(3, 3, 2)
...
mp.show()
```

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_subplot.py  矩阵式子图
"""
import matplotlib.pyplot as mp

mp.figure('Subplot', facecolor='lightgray')

for i in range(9):
    mp.subplot(3, 3, i+1)
    mp.text(0.5, 0.5, i+1, size=36,
        alpha=0.8, ha='center', va='center')
    mp.xticks([])
    mp.yticks([])

mp.tight_layout()
mp.show()
```

**网格式布局**

网格式布局支持单元格的合并。

```python
import matplotlib.gridspec as mg
mp.figure('')
gs = mg.GridSpec(3, 3) # 3行3列的网格布局对象
mp.subplot(gs[0, :2]) 
mp.plot()
mp.show()
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_grid.py   网格布局
"""
import matplotlib.pyplot as mp
import matplotlib.gridspec as mg
mp.figure('Grid Spec', facecolor='lightgray')
gs = mg.GridSpec(3, 3)

mp.subplot(gs[0, :2])
mp.text(0.5, 0.5, 1, ha='center', va='center',
    size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[:2, 2])
mp.text(0.5, 0.5, 2, ha='center', va='center',
    size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[1, 1])
mp.text(0.5, 0.5, 3, ha='center', va='center',
    size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[1:, 0])
mp.text(0.5, 0.5, 4, ha='center', va='center',
    size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[2, 1:])
mp.text(0.5, 0.5, 5, ha='center', va='center',
    size=36)
mp.xticks([])
mp.yticks([])

mp.tight_layout()
mp.show()
```

**自由布局**

```python
mp.figure('')
# 0.03, 0.03：图表做下顶点的位置
# 0.94, 0.94：图表宽度与高度都为0.94
# 这些数值指的是与窗口绘图区域高度、宽度的比例值
mp.axes([0.03, 0.03, 0.94, 0.94])
mp.show()
```

#### 刻度定位器

matplotlib提供了多种刻度定位器可以很方便的设置坐标轴的刻度数据。

```python
ax = mp.gca()
loc = mp.MultipleLocator(1)
# 设置主刻度定位器对象
ax.xaxis.set_major_locator(loc)
# 设置次刻度定位器对象
ax.xaxis.set_minor_locator(loc)
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
deo07_locator.py  刻度定位器
"""
import matplotlib.pyplot as mp

locators = ['mp.MultipleLocator(1)',
            'mp.NullLocator()',
            'mp.MaxNLocator(nbins=4)',
            'mp.AutoLocator()']

mp.figure('Locators', facecolor='lightgray')

for i, locator in enumerate(locators):
    mp.subplot(len(locators), 1, i+1)
    ax = mp.gca()
    # 设置坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    mp.ylim(-1, 1)
    mp.xlim(0, 10)
    ax.spines['bottom'].set_position(('data', 0))
    mp.xticks([])
    mp.yticks([])
    # 设置刻度定位器  多点定位器
    ma_loc = eval(locator)
    ax.xaxis.set_major_locator(ma_loc)
    mi_loc = mp.MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(mi_loc)

mp.tight_layout()
mp.show()
```

#### 刻度网格线

```python
ax = mp.gca()
ax.grid(
	which='',  # major  minor  both
    axis='',   # x  y  both
    linewidth=2,
    linestyle=':',
    color=''
    alpha=0.3
)
```

案例： [1,10,100,1000,100,10,1]

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_grid.py  刻度网格线
"""
import matplotlib.pyplot as mp

y = [1, 10, 100, 1000, 100, 10, 1]

mp.figure('GridLine', facecolor='lightgray')
mp.title('GridLine', fontsize=14)
# 设置水平与垂直方向的刻度定位器
ax = mp.gca()
ax.xaxis.set_major_locator(
    mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(
    mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(
    mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(
    mp.MultipleLocator(50))
# 设置刻度网格线
ax.grid(which='major', axis='both',
    linestyle='-', linewidth=0.75,
    color='orange')
ax.grid(which='minor', axis='both',
    linestyle='-', linewidth=0.25,
    color='orange')

mp.plot(y)
mp.show()
```

#### 半对数坐标

y轴将会以指数方式递增。这样可以更好的显示底部数据的细节。

```python
mp.semilogy(x, y)
```

#### 散点图

在图表中以一组散点，描述一组样本。每个样本有不同的特征属性，可以用散点的不同属性描述这些特征。

| 身高 | 体重 | 性别 | 年龄段 | 种族 |
| ---- | ---- | ---- | ------ | ---- |
| 175  | 65   | M    | 青年   | 亚洲 |

相关API:

```python
mp.scatter(xarray, yarray,
          marker='',    # 点型
          s=60,			# 大小
          zorder=3,		# 图层编号
          c=d,	# 设置颜色
          cmap='jet' # 设置颜色映射
)
```

np提供了相关API获取一组符合正态分布的随机数：

```python
# 随机生成n个符合正态分布的数
# 175: 期望
# 10:  标准差
x = np.random.normal(175, 10, n)
```

案例：绘制散点图：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo10_scatter.py  散点图
"""
import numpy as np
import matplotlib.pyplot as mp

# 随机生成一组数
n = 300
x = np.random.normal(175, 5, n)
y = np.random.normal(65, 10, n)

# 画图
mp.figure('Persons', facecolor='lightgray')
mp.title('Persons', fontsize=16)
mp.xlabel('Height', fontsize=12)
mp.ylabel('Weight', fontsize=12)
d = (x-175)**2 + (y-65)**2
mp.scatter(x, y, s=60, alpha=0.8,
    c=d, cmap='jet_r', marker='o', label='Persons')
mp.legend()
mp.show()
```





















