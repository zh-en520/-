# 数据分析DAY02

## matplotlib概述

matplotlib是python的一个绘图库，可以方便的绘制出版质量级别的图形。

## matplotlib基本功能

1. 基本绘图
   1. 设置线型、线宽、颜色
   2. 设置坐标轴范围、坐标刻度、坐标轴
   3. 图例
   4. 绘制特殊点与备注
2. 高级绘图
   1. 图像窗口与子图
   2. 刻度定位器与刻度网格线
   3. 半对数坐标
   4. 散点图
   5. 填充
   6. 条形图、饼状图
   7. 等高线图、热成像图
   8. 3D图像绘制
   9. 简单动画

## matplotlib基本功能详解

### 基本绘图

**绘图核心API**

```python
import matplotlib.pyplot as mp
# x: x坐标数组
# y: y坐标数组
mp.plot(x, y)
mp.show()
```

绘制水平线与垂直线：

```python
# 垂直线
# 在ymin~ymax的区间之内绘制垂直线
mp.vlines(val, ymin, ymax)
# 水平线
# 在xmin~xmax的区间之内绘制水平线
mp.hlines(val, xmin, xmax)
```

#### 线型、线宽、颜色

```python
# linestyle: 线型 '-'  '--'  ':'
# linewidth: 线宽
# color: 颜色
#    常见颜色英文单词 或 首字母
#    #4433ab  (0.8,0.8,0.8)
#    (0.8,0.8,0.8,0.5)  0.5为透明度
# alpha: 透明度
mp.plot(x, y, linestyle='',
       linewidth=3,color='',alpha=0.5)
```

#### 设置坐标范围

```python
# x轴的可视区域
mp.xlim(x_min, x_max)
# y轴的可视区域
mp.ylim(y_min, y_max)
```

#### 设置坐标刻度

```python
# 设置x轴的刻度
# val_list: 刻度值的列表
# text_list: 文本列表
mp.xticks(val_list, text_list)
# 设置y轴的刻度  参数列表同上
mp.yticks(val_list, text_list)
```

***latex排版语法字符串***
$$
-\frac{\pi}{2} \quad \quad 
a^2 + b^2 = c^2
$$

#### 设置坐标轴

主要用于设置坐标轴的颜色与位置。

```python
# 获取当前坐标轴对象
ax = mp.gca()
# 'left' 'right' 'bottom' 'top'
axis = ax.spines['坐标轴名']
# 设置位置
axis.set_position(('data', 0))
# 设置颜色
axis.set_color(color) 
```

#### 设置图例

```python
# 绘制曲线时，添加label，表示图例中的文本
mp.plot(x, y, label='')
mp.legend(loc=0)
```

#### 绘制特殊点

```python
mp.scatter(
    x, y,    # 所有点的坐标列表
    s=60,    # 大小
    marker='',   # 点型 
	edgecolor='',  # 边缘色
	facecolor='',  # 填充色
	zorder=3  # 绘制图层的编号 (编号越大，图层越靠上，覆盖在编号小的图层之上)
)
```

#### 备注

```python
mp.annotate(
	'text',
    xycoords='data',  # 目标点的坐标系
    xy=(1,1),   # 目标点的坐标
    textcoords='offset points',
    xytext=(10,20),
    fontsize=12,
    arrowprops=dict(
    	arrowstyle='->',
        connectionstyle='angle3'
    )
)
```

### 高级绘图

#### 窗口与子图

显示多个窗口：

```python
mp.figure('Figure A')
mp.plot()  # 针对A窗口绘制
mp.figure('Figure B')
mp.plot()  # 针对B窗口绘制
mp.figure('Figure A') 
#将不会重新创建新窗口，而是把A置为当前窗口
mp.plot() # 针对A窗口
mp.show()
```

```python
"""
demo03_figure.py  窗口操作
"""
import matplotlib.pyplot as mp

mp.figure('Figure A', facecolor='gray')
mp.plot([1,2,3,2,3,1,7])
mp.figure('Figure B', facecolor='lightgray')
mp.plot([10,20,30,20,30,10,70])
mp.figure('Figure A')
mp.plot([6,3,6,4,8,3,1])
# 窗口常见的属性参数设置方法
mp.title('Figure A', fontsize=16)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

#mp.tight_layout()  # 使用紧凑布局
mp.show()
```

**子图 （在同一个窗口中显示多个图表）**

**矩阵式布局**

用于绘制规则的子图布局：

```python
mp.figure()
mp.subplot(rows, cols, num)
mp.plot()
mp.subplot(2, 2, 2) # 2行2列布局下的2图
mp.plot()
mp.show()
```

网格式布局

网格式布局支持单元格合并。

```python
import matplotlib.gridspec as mg
# 构建GridSpec对象
gs = mg.GridSpec(3, 3)
mp.subplot(gs[0, :2])
mp.show()
```

自由布局

```python
# 0.03, 0.03 左下角定位点的位置 
# 0.94, 0.94 子图的宽度与高度
mp.axes([0.03, 0.03, 0.94, 0.94])
mp.text()
mp.plot()
```

#### 刻度定位器

```python
# 当前坐标系
ax = mp.gca()
locator = mp.MultipleLocator(10)
# 设置x轴的主刻度定位器
ax.xaxis.set_major_locator(locator)
# 设置x轴的次刻度定位器
ax.xaxis.set_minor_locator(locator)
```

案例：绘制数轴

```python
"""
demo06_locator.py 刻度定位器
"""
import matplotlib.pyplot as mp

locators = ['mp.NullLocator()',
			'mp.MultipleLocator(1)', 
			'mp.MaxNLocator(nbins=4)',
			'mp.AutoLocator()']

mp.figure('Locator', facecolor='lightgray')

for i, locator in enumerate(locators):
	mp.subplot(4, 1, i+1)
	ax = mp.gca()
	# 干掉上左右轴
	ax.spines['top'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	mp.xlim(0, 10)
	mp.ylim(-1, 1)
	ax.spines['bottom'].set_position(('data',0))
	mp.yticks([])
	# 设置主刻度定位器  每隔1就显示一个主刻度
	major_locator = eval(locator)
	ax.xaxis.set_major_locator(major_locator)
	minor_locator = mp.MultipleLocator(0.1)
	ax.xaxis.set_minor_locator(minor_locator)

mp.show()
```

#### 刻度网格线

```python
ax = mp.gca()
ax.grid(
	which='',   'major|minor|both'
    axis='',	'x|y|both'
    linewidth=2,
    linestyle=':',
    color=''
)
```

案例：

```python
"""
demo07_grid.py  刻度网格线
"""
import matplotlib.pyplot as mp
import numpy as np

y = np.array([1, 10, 100, 1000, 100, 10, 1])

mp.figure('GridLine', facecolor='lightgray')
mp.title('GridLine', fontsize=14)
mp.ylabel('Y', fontsize=12)
# 设置刻度定位器与网格线
ax = mp.gca()
ax.xaxis.set_major_locator(
			mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(
			mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(
			mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(
			mp.MultipleLocator(50))
ax.grid(which='major', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.75)
ax.grid(which='minor', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.25)


mp.plot(y, 'o-', color='dodgerblue')
mp.show()
```

#### 半对数坐标轴

y轴将会以指数方式递增。可以更好的显示底部数据的细节。

```python
mp.plot() 换为 mp.semilogy()
```

#### 散点图

基于散点图可以方便看到数据的分布状态。

```python
mp.scatter(
    x, y,    # 所有点的坐标列表
    s=60,    # 大小
    marker='',   # 点型 
	edgecolor='',  # 边缘色
	facecolor='',  # 填充色
	zorder=3  # 绘制图层的编号 (编号越大，图层越靠上，覆盖在编号小的图层之上)
)
```

案例：随机生成100个人（身高随机，体重随机）

numpy提供了normal函数用于产生符合正态分布的随机数。

```python
# 基于正态分布随机生成n个数
# 172： 期望
# 20：  标准差
x = np.random.normal(172, 20, n)
```







