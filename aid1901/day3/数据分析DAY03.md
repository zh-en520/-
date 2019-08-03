# 数据分析DAY03

#### 填充图

以某种颜色填充两条曲线的闭合区域。

```python
mp.fill_between(
	x, 		# 水平坐标数组
    sinx, 	# 函数曲线1
    cosx, 	# 函数曲线2
    sinx < cosx, # 填充条件
    color='',
    alpha=0.5
)
```

案例：sinx=sin(x)  cosx=cos(x/2)/2    [0, 8π]

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_fill.py  填充
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
x = np.linspace(0, 8*np.pi, n)
sinx = np.sin(x)
cosx = np.cos(x/2) / 2

mp.figure('Fill', facecolor='lightgray')
mp.title('Fill', fontsize=18)
mp.grid(linestyle=':')
mp.plot(x, sinx, color='dodgerblue',
    label='sinx') 
mp.plot(x, cosx, color='orangered',
    label='cosx')
# 填充
mp.fill_between(x, sinx, cosx, sinx<cosx, 
    alpha=0.5, color='dodgerblue')
mp.fill_between(x, sinx, cosx, sinx>cosx, 
    alpha=0.5, color='orangered')

mp.legend()
mp.show()
```

#### 条形图(柱状图)

```python
mp.bar(
	x,	# 一个数组
    y,  # 每个柱子的高度
    width, # 柱子的宽度 0~1
    bottom,  # 绘制柱子的底部起始位置
    color='',
    ...label... alpha ... 
)
```

案例：绘制条形图表示苹果的12个月的销量。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_bar.py  柱状图
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.arange(12)
apples=[87,12,58,12,41,82,54,12,98,36,40,45]
oranges=[80,16,34,12,98,34,61,20,56,21,34,32]

mp.figure('Bar', facecolor='lightgray')
mp.title('Bar', fontsize=18)
mp.grid(linestyle=':')
mp.bar(x-0.2, apples, 0.4, color='limegreen', 
    label='Apple', align='center')
mp.bar(x+0.2, oranges, 0.4, color='orange', 
    label='Orange', align='center')
mp.xticks(x, ['Jan', 'Feb', 'Mar', 'Apr',
    'May', 'Jun', 'Jul', 'Aug', 'Sep',
    'Oct', 'Nov', 'Dec'])

mp.legend()
mp.show()
```

#### 饼状图

```python
mp.axis('equal')  # 设置等轴比例
mp.pie(
	values,	 # 一组值
    spaces,  # 一组扇形间距
    labels,  # 一组标签文本
    colors,	 # 一组颜色
    '%.2f%%', # 标签所占比例的字符串格式
    shadow=True,  # 是否有阴影
    starangle=0,  # 起始旋转角度
    radius=1      # 半径
)
```

案例：显示编程语言的占有率。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_pie.py  饼状图
"""
import numpy as np
import matplotlib.pyplot as mp
# 整理数据
labels=['Python', 'JS', 'C++', 'Java', 'PHP']
values=[26, 17, 21, 29, 5]
spaces=[0.05, 0.01, 0.01, 0.01, 0.01]
colors=['dodgerblue', 'orangered',
    'limegreen', 'violet', 'gold']
# 画图
mp.figure('Pie Chart', facecolor='lightgray')
mp.title('Pie Chart', fontsize=18)
mp.grid(linestyle=':')
mp.axis('equal') # 等轴比例
mp.pie(values, spaces, labels, colors, 
    '%.2f%%', shadow=True, 
    startangle=45)
mp.legend()
mp.show()
```

#### 等高线图

组成等高线图需要网格点坐标矩阵，也需要获取每个坐标点的高度值。所以等高线属于3D数学模型。

```python
# 绘制等高线
cntr = mp.contour(
	x, y,	# x与y都是2维数组，组成网格坐标矩阵
    z,      # z是2维数组，表示相应坐标点的高度值
    8,      # 高度分8份
    colors='',  # 等高线的颜色
    linewidths=0.5  # 线宽
)
# 为等高线图添加高度标签
mp.clabel(cntr, fmt='%.1f', fontsize=8,
    inline_spacing=1)
# 填充等高线图
mp.contourf(x, y, z, 8, cmap='jet')
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_contour.py  等高线图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 根据一个公式，通过x, y计算高度值。
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制等高线
mp.figure('Contour', facecolor='lightgray')
mp.title('Contour', fontsize=18)
mp.grid(linestyle=':')
cntr = mp.contour(x, y, z, 8, colors='black',
    linewidths=0.5)
# 为等高线图添加高度标签
mp.clabel(cntr, fmt='%.1f', fontsize=8,
    inline_spacing=1)
# 填充等高线图
mp.contourf(x, y, z, 8, cmap='jet')
mp.show()
```

#### 热成像图 

用图形的方式显示矩阵的内容。

```python
mp.imshow(z, cmap='jet')
mp.colorbar()
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_imshow.py  热成像图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
x, y = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
# 根据一个公式，通过x, y计算高度值。
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制
mp.figure('Imshow', facecolor='lightgray')
mp.title('Imshow', fontsize=18)
mp.grid(linestyle=':')
mp.imshow(z, cmap='jet', origin='lower')
mp.colorbar()
mp.show()
```















