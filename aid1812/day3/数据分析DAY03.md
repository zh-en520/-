# 数据分析DAY03

#### 填充

```python
mp.fill_between(
	x,       # x轴水平坐标数组
    sinx, 
    cosx, 
    sinx < cosx,
    color='',
    alpha=0.5
)
```

案例：

```python
"""
demo01_fill.py 填充
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
x = np.linspace(0, 8*np.pi, n)
sinx = np.sin(x)
cosx = np.cos(x/2) / 2 

mp.figure('Fill', facecolor='lightgray')
mp.title('Fill', fontsize=14)
mp.xlabel('X', fontsize=12)
mp.ylabel('Y', fontsize=12)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
mp.plot(x, sinx, c='dodgerblue', 
	label=r'$y=sin(x)$')
mp.plot(x, cosx, c='orangered', 
	label=r'$y=\frac{1}{2}cos(\frac{x}{2})$')

mp.fill_between(x, sinx, cosx, sinx>cosx,
	color='dodgerblue', alpha=0.5)
mp.fill_between(x, sinx, cosx, sinx<cosx,
	color='orangered', alpha=0.5)
mp.legend()
mp.show()
```

#### 条形图（柱状图）

```python
mp.bar(
	x,
    y,     # 每个柱子的高度数组
    width, # 每个柱子的宽度
    color='',
    label='',
    alpha=0.2
)
```

案例：

```python
"""
demo02_bar.py 柱状图
"""
import numpy as np
import matplotlib.pyplot as mp

# 整理苹果12个月销量
apples=[92,34,75,32,96,52,36,10,23,41,22,35]
oranges=[23,43,46,58,23,74,56,72,38,95,63,9]
x = np.arange(len(apples))
mp.figure('Bar Chart', facecolor='lightgray')
mp.title('Bar Chart', fontsize=14)
mp.xlabel('Date', fontsize=12)
mp.ylabel('Volumn', fontsize=12)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
mp.bar(x-0.2, apples, 0.4, color='limegreen',
	label='Apples')
mp.bar(x+0.2, oranges, 0.4, color='orangered',
	label='Oranges')
# 修改x刻度文本
mp.xticks(x, ['Jan', 'Feb', 'Mar', 'Apr',
	'May', 'Jun', 'Jul', 'Aug', 'Sep',
	'Oct', 'Nov', 'Dec'])

mp.legend()
mp.show()
```

#### 饼状图

```python
mp.axis('equal')
mp.pie(
	values,		# 值列表
    spaces,  	# 扇形间的间隙列表
    labels, 	# 每个扇形的标签列表
    colors,		# 每个扇形的颜色列表
    '%.2f%%',	# 每个标签占比的输出格式
    shadow=True, # 显示阴影
    startangle=45, # 起始角度
    radius=1	# 饼状图的半径
)
```

案例：

```python
"""
demo03_pie.py  饼状图
"""
import matplotlib.pyplot as mp

labels=['Python', 'Javascript', 
		'C++', 'Java', 'PHP']
values=[26, 17, 21, 29, 11]
spaces=[0.05, 0.01, 0.01, 0.01, 0.01]
colors=['dodgerblue', 'orangered', 
	'limegreen', 'violet', 'gold']

mp.figure('Pie Chart', facecolor='lightgray')
mp.title('Pie Chart', fontsize=14)
# 设置等轴比例显示饼状图
mp.axis('equal')
mp.pie(values, spaces, labels, colors,
	'%.2f%%', shadow=True, startangle=45)
mp.legend()
mp.show()
```

#### 等高线图

绘制等高线需要网格点坐标矩阵，也需要每个点的高度。所以等高线属于3D数学模型。

```python
cntr = contour(
	x, y,  # x与y可以组成网格点坐标矩阵
    z,     # 每个坐标点的高度
    8,	   # 把总8份
    colors='',  # 等高线的颜色
    linewidths=0.5
)
# 为每个等高线绘制高度标签
mp.clabel(cntr, inline_spacing=1, 
	fmt='%.1f', fontsize=10)
```

案例：

```python
"""
demo04_contour.py 等高线图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
# 构建网格点坐标矩阵
x, y = np.meshgrid(np.linspace(-3,3,n),
				   np.linspace(-3,3,n))
# 根据每个坐标点的x与y计算高度值z
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制等高线
mp.figure('Coutour', facecolor='lightgray')
mp.title('Coutour', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
cntr=mp.contour(x, y, z, 8, colors='black',
	linewidths=0.5)
# 为每个等高线绘制高度标签
mp.clabel(cntr, inline_spacing=1, 
	fmt='%.1f', fontsize=10)
# 为等高线区间填充颜色
mp.contourf(x, y, z, 8, cmap='jet')
mp.show()
```

#### 热成像图

```python
# 使用cmap的颜色映射，以图像方式显示z矩阵
# origin: y轴方向  'lower' 'upper'
mp.imshow(z, cmap='', origin='lower')
mp.colorbar()
```

案例：

```python
"""
demo05_imshow.py 热成像图
"""
import numpy as np
import matplotlib.pyplot as mp

n = 1000
# 构建网格点坐标矩阵
x, y = np.meshgrid(np.linspace(-3,3,n),
				   np.linspace(-3,3,n))
# 根据每个坐标点的x与y计算高度值z
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制热成像图
mp.figure('Imshow', facecolor='lightgray')
mp.title('Imshow', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.imshow(z, cmap='jet', origin='lower')
mp.show()
```

#### 3D图像绘制

```python
from mpl_toolkits.mplot3d import axes3d
# 获取3d坐标系
ax3d = mp.gca(projection='3d')
```

**3D散点图** 

```python
ax3d.scatter(
	x, y, z, 
    marker='',
    s=60, color='', c=d, cmap=''
)
```

案例：

```python
"""
demo06_3dscatter.py   三维散点图
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 500
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
z = np.random.normal(0, 1, n)

mp.figure('3D Scatter')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('x', fontsize=14)
ax3d.set_ylabel('y', fontsize=14)
ax3d.set_zlabel('z', fontsize=14)
mp.tick_params(labelsize=10)
d = x**2 + y**2 + z**2
ax3d.scatter(x, y, z, s=30, c=d,
	alpha=0.7, cmap='jet_r')
mp.show()
```

**3D线框图**

```python
ax3d.plot_wireframe(
	x, y, z,  #xy网格点坐标矩阵z为高度
    rstride=30,  # 行跨距
    cstride=30,  # 列跨距
	linewidth=3, color=''
)
```

**3D曲面图**

```python
ax3d.plot_surface(
    x, y, z,  #xy网格点坐标矩阵z为高度
    rstride=30,  # 行跨距
    cstride=30,  # 列跨距
    cmap='jet'
)
```

```python
"""
demo07_surface.py 3d平面图
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 1000
# 构建网格点坐标矩阵
x, y = np.meshgrid(np.linspace(-3,3,n),
				   np.linspace(-3,3,n))
# 根据每个坐标点的x与y计算高度值z
z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)
# 绘制等高线
mp.figure('3D Surface', facecolor='lightgray')
mp.tick_params(labelsize=10)
ax3d = mp.gca(projection='3d')
ax3d.plot_surface(x, y, z, cmap='jet',
	rstride=30, cstride=30)
mp.show()
```

#### 极坐标系

```python
# polar将会把当前坐标系改为极坐标系
mp.gca(projection='polar')
```

#### 简单动画

动画即是在一段时间内快速连续的重新绘制图像的过程。

```python
import matplotlib.animation as ma
# update中编写更新界面的代码
def update(number):
    pass
# 每10毫秒执行一次update函数
ma.FuncAnimation(
    mp.gcf(), update, interval=10)
mp.show()
```

案例：随机生成100个泡泡，使他们不断增大。

```python
"""
demo10_anim.py 动画
"""
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma

# 随机生成100个泡泡
n = 100
balls = np.zeros(n, dtype=[
		('position', float, 2),
		('size', float, 1),
		('growth', float, 1),
		('color', float, 4)])

# 初始化泡泡的属性
balls['position']=np.random.uniform(0,1,(n,2))
balls['size']=np.random.uniform(30,70,n)
balls['growth']=np.random.uniform(10,20,n)
balls['color']=np.random.uniform(0,1,(n,4))

mp.figure('Animation', facecolor='lightgray')
mp.title('Animation', fontsize=14)
mp.xticks([])
mp.yticks([])
sc = mp.scatter(balls['position'][:,0], 
		balls['position'][:,1],
		c=balls['color'],
		s=balls['size'])

# 没30毫秒 更新泡泡大小
def update(number):
	balls['size']+=balls['growth']
	ind = number % n
	balls[ind]['size'] = \
			np.random.uniform(30, 70, 1)
	balls[ind]['position'] = \
			np.random.uniform(0, 1, (1,2))
	# 修改属性后重新绘制界面
	sc.set_sizes(balls['size'])
	sc.set_offsets(balls['position'])

anim=ma.FuncAnimation(mp.gcf(), update, interval=30)

mp.show()
```

基于生成器提供数据，实现动画绘制。

```python
# 生成器 提供数据
def generator():
    yield data
# 更新界面
def update(data):
    pass
# 每30毫秒，调用generator获取数据，把数据
# 传递给update函数更新界面
anim=ma.FuncAnimation(
    mp.gcf(), update, generator, 
    interval=30)
```

案例：

```python
"""
demo11_anim.py 动画
"""
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.animation as ma


mp.figure('Animation', facecolor='lightgray')
mp.title('Animation', fontsize=14)
mp.xlim(0, 10)
mp.ylim(-3, 3)

pl = mp.plot([],[])[0]

# 每30毫秒 更新
def update(data):
	t, v = data
	# 把新坐标加入曲线
	x, y = pl.get_data()
	x = np.append(x, t)
	y = np.append(y, v)
	pl.set_data(x, y)
	if x[-1]>10:
		mp.xlim(x[-1]-10, x[-1])

x = 0
def generator():
	global x
	y = np.sin(2*np.pi*x) * \
		np.exp(np.sin(0.2*np.pi*x))
	yield (x,y)
	x+=0.05

anim=ma.FuncAnimation(mp.gcf(), update, 
	generator, interval=30)

mp.show()
```

### 文件加载

numpy提供了函数用于加载文件：

```python
dates, closing_prices = np.loadtxt(
	'../data/xx.csv', # 文件路径
    delimiter=',',  # 分隔符
    usecols=(1, 6),
    unpack=True,  # 是否拆包
    dtype='U10, f8',
    converters={1:func}
)
```

案例：









