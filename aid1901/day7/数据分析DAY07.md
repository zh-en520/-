# 数据分析DAY07

### 特征值与特征向量

对于n阶方阵A，如果存在数a和非零n维列向量x， 使得Ax=ax，则称a是矩阵A的一个特征值，x是矩阵A属于特征值a的特征向量。

```python
# 提取特征值与特征向量
# A：原方阵    
# eigvals：一组特征值
# eigvecs：与特征值对应的一组特征向量
eigvals, eigvecs = np.linalg.eig(A)
# 已知特征值与特征向量，求方阵
S = eigvecs * np.diag(eigvals) * eigvecs.I
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_eig.py  特征值提取
"""
import numpy as np

A = np.mat('4 8 9; 3 5 8; 1 9 3')
print(A)
# 提取特征信息
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
# 逆向推方阵
A2 = eigvecs * np.diag(eigvals) * eigvecs.I
print(A2)

# 如果只保留一部分特征值 则：
eigvals[3:] = 0
A3 = eigvecs * np.diag(eigvals) * eigvecs.I
print(A3)
```

案例：提取图像的特征值，保留部分特征，生成新图片。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_eig.py 提取图像特征值
"""
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as mp

image = sm.imread('../da_data/lily.jpg', True)
print(image, image.shape)
# 提取 image 的特征
eigvals, eigvecs = np.linalg.eig(image)
# 逆向生成图像
# 只保留50个特征值
eigvals[100:] = 0
image2 = np.mat(eigvecs)\
         * np.diag(eigvals)\
         * np.mat(eigvecs).I
# 绘制
mp.figure('EIG Image')
mp.subplot(121)
mp.imshow(image, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.subplot(122)
mp.imshow(image2.real, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()
mp.show()
```

### 奇异值分解

有一个矩阵M， 可以分解为3个矩阵U、S、V，使得U x S x V等于M。U与V都是正交矩阵（乘以自身的转置矩阵结果为单位矩阵）。那么S矩阵主对角线上的值称为M的奇异值，其他元素都为0。

```python
# U与V是正交矩阵
# sv: 分解所得的奇异值数组
U, sv, V = np.linalg.svd(M, full_matrices=False)
# 逆向推导M：
M2 = U * np.diag(sv) * V
```

### 快速傅里叶变换模块(fft)

**什么是傅里叶定理？**

法国科学家傅里叶说：任何一条周期性曲线，无论多么跳跃或不规则，都能表示为一组光滑的正弦函数的叠加。

**什么是傅里叶变换？**

傅里叶变换即是将一条周期曲线基于傅里叶定理进行拆解，得到一组光滑正弦曲线的变换过程。

傅里叶变换的目的是可将时间域的信号转为频域（频率域）信号，随着域的不同，对同一个事物的了解角度也就随之改变；因此在时域中某些不好处理的地方，在频域中可以较为简单的处理。例如：数据存储、数据降噪等。

**FFT相关函数：**

```python
import numpy.fft as fft
# 快速傅里叶变换
# 结果中的每个复数即可以描述一条正弦曲线
# 复数的模代表振幅A， 复数的辅角代表相位角φ
复数数组 = fft.fft(原函数y数组)
# 通过采样数、采样周期求得曲线的频率序列
freqs = fft.fftfreq(采样数量，采样周期)

# 逆向傅里叶变换
y2 = fft.ifft(复数数组)
```

案例：拆解方波。绘制图像。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_fft.py   快速傅里叶变换
"""
import numpy as np
import matplotlib.pyplot as mp
import numpy.fft as nf
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
# 合成方波
n = 1000
y = np.zeros(1000)
for i in range(1, n+1):
    y += 4*np.pi/(2*i-1) * np.sin((2*i-1)*x)
# FFT拆解方波
freqs = nf.fftfreq(y.size, y[1]-y[0])
complex_ary = nf.fft(y)
# 逆向傅里叶变换
y2 = nf.ifft(complex_ary)
#绘时域图
mp.figure('FFT')
mp.subplot(121)
mp.title('Time Domain')
mp.grid(linestyle=':')
mp.plot(x, y, label='y')
mp.plot(x, y2, linewidth=7, alpha=0.4,
    label='y2')
mp.legend()

mp.subplot(122)
mp.title('Frequency Domain')
mp.grid(linestyle=':')
power = np.abs(complex_ary)
mp.plot(freqs[freqs>0], power[freqs>0] )

mp.tight_layout()
mp.show()
```

##### 基于傅里叶变换的频域滤波

含噪信号由高能信号与低能噪声叠加而成，可以通过FFT的频域滤波实现降噪。通过FFT使含噪信号转为含噪频谱，留下高能频谱，去除低能噪声，经过IFFT生成高能信号。

案例：基于傅里叶变换的频域滤波

1. 读取音频文件，获取音频文件的基本信息：采样率，采样周期，每个采样点的声音位移值。绘制音频的时域的：时间/位移图像。

```python
# 1.  
# sample_rate: 采样率 每秒采样点的个数
# noised_sigs：每个采样点的位移
sample_rate, noised_sigs = \
    wf.read('../da_data/noised.wav')
print('sample_rate:', sample_rate)
print('noised_sigs:', noised_sigs.shape)
times = np.arange(len(noised_sigs))/sample_rate

mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.title('Time Domain')
mp.ylabel('Signal')
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178], 
    c='dodgerblue', label='Signals')
mp.legend()
mp.tight_layout()
mp.show()
```

2. 基于傅里叶变换，获取音频频域信息，绘制频域的：频率/能量图像。

```python
freqs = nf.fftfreq(times.size, 1/sample_rate)
complex_ary = nf.fft(noised_sigs)
powers = np.abs(complex_ary)
mp.subplot(222)
mp.title('Frequency Domain')
mp.ylabel('power')
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs>0], powers[freqs>0],
    c='orangered', label='Noised')
mp.legend()
```

3. 将低能噪声去除，绘制音频频域：频率/能量图像。

```python
# 3.
fund_freq = freqs[powers.argmax()]
noised_indices = np.where(freqs != fund_freq)
complex_filter = complex_ary.copy()
complex_filter[noised_indices] = 0  # 滤波
power_filter = np.abs(complex_filter)
mp.subplot(224)
mp.ylabel('power')
mp.grid(linestyle=':')
mp.plot(freqs[freqs>0], power_filter[freqs>0],
    c='orangered', label='Filter')
mp.legend()
```

4. 基于IFFT，生成新的音频信号，绘制图像。

```python
filter_sigs = nf.ifft(complex_filter)
mp.subplot(223)
mp.ylabel('Signal')
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178], 
    c='dodgerblue', label='Signals')
mp.legend()
```

5. 生成音频文件。

```python
wf.write('../da_data/out.wav', sample_rate,
   filter_sigs.astype('i2'))
```

### 随机数random模块

生成服从特定统计规律的随机数序列。

#### 二项分布(binomial)

二项分布就是重复n次的伯努利实验。每次实验只有两种可能的结果，而且两种结果发生与否相互对立相互独立。事件发生与否的概率在每一次实验中都保持不变。

```python
# 产生size个随机数，符合二项分布。
# 每个随机数来自n次尝试中成功的次数，其中每次尝试成功的
# 概率为p
r = np.random.binomial(n, p, size)
```

#### 正态分布(normal)

```python
# 随机生成一组服从标准正态分布的随机数 期望：0 标准差  1
np.random.normal(size)
# 随机生成一组服从正态分布的随机数 期望：175 标准差：10
np.random.normal(175, 10, size)
```

#### 平均分布（uniform）

```python
# 产生size个随机数，服从平均分布 [40~70]
np.random.uniform(40,70, size)
```

#### 超几何分布(hypergeometic)

```python
# 产生size个随机数，每个随机数为在总样本中随机抽取
# nsample个样本后好样本的个数。
# 所有样本由ngood个好样本与nbad个坏样本组成。
r = np.random.hypergeometic(
        ngood, nbad,nsample, size)
```

### 杂项功能

#### 排序

**联合间接排序**

```python
排序后的有序索引 = np.lexsort((次序列，主序列))
```

案例：

```python
import numpy as np

prods = np.array(['Apple', 'Huawei', 'Mi',
                  'Oppo', 'Vivo'])
prices = [8000, 4999, 2999, 3999, 3999]
volumns = np.array([40, 80, 50, 35, 40])
indices = np.lexsort((-volumns, prices))
print(indices)
```

**复数数组排序**

先按实部排序，若实部数值相同，再按虚部排序。

```python
np.sort_complex(复数数组)
```

**插入排序**

若已知有序数组，需要向该数组中插入元素，使数组依然有序：

```python
待插入索引 = np.seachsorted(有序数组，待插入数据)
# 把b元素按照indices的索引位置，插入a数组
d = np.insert(a, indices, b)
```

案例：

```python
# 插入排序
a = np.array([1, 2, 3, 6, 9])
b = np.array([5, 8])
indices = np.searchsorted(a, b)
print(indices)
# 把b元素按照indices的索引位置，插入a数组
d = np.insert(a, indices, b)
print(d)
```

#### 插值

scipy提供了常见的插值算法，可以通过一组离散数据生成符合一定规律的插值函数（连续函数）。这样就可以传入x，得到函数值。 插值是实现离散数据连续化的一种方式。

```python
import scipy.interpolate as si
func = si.interp1d(
	离散数据x坐标,
    离散数据y坐标,
    kind='linear'  （插值算法）
)
```

案例：搞出13个散点，基于scipy的插值得到一个连续函数，绘制这个连续函数的图像。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_inter.py  插值
"""
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as mp
# 造一些散点数据
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 15)
dis_y = np.sinc(dis_x)
# 绘图
mp.scatter(dis_x, dis_y, c='orangered',
    s=60, marker='o')

# 基于这些离散数据，使用插值获得连续函数
linear=si.interp1d(dis_x, dis_y, kind='linear')
# 绘制linear函数图像
x = np.linspace(min_x, max_x, 1000)
y = linear(x)
# mp.plot(x, y)

# 三次样条插值器
cubic=si.interp1d(dis_x, dis_y, kind='cubic')
# 绘制linear函数图像
y = cubic(x)
mp.plot(x, y)

mp.show()
```

插值与随机数都可用于数据预处理，异常值的修正、空白值的填充等。













