# 数据分析DAY07

#### 三角函数通用函数

```python
np.sin()
```

**傅里叶定理**

法国科学家傅里叶说过：任何一个周期曲线，无论多么跳跃或不规则，都可以被看做一组不同振幅、频率、相位的正弦函数叠加而成。
$$
y = \frac{4\pi}{2n-1} \times sin((2n-1)x)
$$
案例：

```python
"""
demo01_sin.py   测试傅里叶定理
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.zeros(1000)
n = 1000
for i in range(1, n+1):
	y += 4/((2*i-1)*np.pi)*np.sin((2*i-1)*x)

mp.plot(x, y, label='n=4')
mp.legend()
mp.show()
```

### 特征值与特征向量

对于n阶方阵A，如果存在数a和非零n维列向量x，使得Ax=ax，则称a是矩阵A的一个特征值，x是矩阵A属于特征值a的特征向量。

```python
# 已知n阶方阵，提取特征值
# 特征值，特征向量=xxxx(原矩阵)
eigvals, eigvecs = np.linalg.eig(A)

# 逆向推导原方阵
A = np.mat(eigvecs) * 
    np.mat(np.diag(eigvals)) * 
    np.mat(eigvecs).I
```

案例：

```python
"""
demo02_eig.py  特征值提取
"""
import numpy as np
A = np.mat('2 3 1; 4 7 4; 8 5 2')
print(A)
vals, vecs = np.linalg.eig(A)
print(vals)
print(vecs, type(vecs))
# 回推导原方阵
S = vecs * np.diag(vals) * vecs.I
print(S.real)
# 抹掉一部分特征值 推导原方阵
vals[2:] = 0
S = vecs * np.diag(vals) * vecs.I
print(S.real)
```

案例：读取图片的亮度矩阵，提取特征值与特征向量，保留部分特征值，重新生成新的亮度矩阵，绘制图片。

```python
"""
demo03_eig.py  提取图片特征值
读取图片的亮度矩阵，提取特征值与特征向量，
保留部分特征值，重新生成新的亮度矩阵，绘制。
"""
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as mp
# True：提取图片亮度矩阵  二维
# False:提取图片颜色矩阵  三维
img=sm.imread('../da_data/lily.jpg', True)
print(type(img), img.shape, img.dtype)
print(img[0,0])
#提取特征值
vals, vecs = np.linalg.eig(img)
vals[10	:] = 0
vecs = np.mat(vecs)
img2 = vecs * np.diag(vals) * vecs.I
img2 = img2.real

mp.figure('EIG', facecolor='lightgray')
mp.subplot(121)
mp.xticks([])
mp.yticks([])
mp.imshow(img, cmap='gray')
mp.subplot(122)
mp.xticks([])
mp.yticks([])
mp.imshow(img2, cmap='gray')
mp.tight_layout()
mp.show()
```

### 奇异值分解

有一个矩阵M，可以分解为3个矩阵U、S、V，使得U x S x V等于M。U与V都是正交矩阵（自身乘以自身的转置矩阵为单位矩阵）。那么S矩阵主对角线上的元素称为矩阵M的奇异值，其他元素都为0.

```python
# 奇异值分解    sv中保存了奇异值
U, sv, V = np.linalg.svd(M)
# 已知奇异值， 推导原矩阵
M2 = U * np.diag(sv) * V
```

案例：

```python
# 奇异值分解
U, sv, V = np.linalg.svd(img)
sv[50:] = 0
img3 = np.mat(U) * np.diag(sv) * np.mat(V)
img3 = img3.real
```

### 傅里叶变换(fft)

傅里叶定理：任何一个周期曲线，无论多么跳跃或不规则，都可以被看做一组不同振幅、频率、相位的正弦函数叠加而成。

傅里叶变换即是把一个跳跃或不规则的曲线拆解成一组不同振幅、频率、相位的正弦函数的过程。

傅里叶变换的意义在于可以将时域（时间域）上的信号转变为频域（频率域）上的信号，随着域的不同，对同一个事物的了解角度也随之发生改变，因此在时域中某些不好处理的地方，在频率就可以较为简单的处理。大量减少处理信号的存储量。

**傅里叶变换相关函数**

```python
import numpy.fft as nf
# 对原函数执行快速傅里叶变换
# 复数数组：保存fft处理后的一组正弦函数信息
#    复数的模：振幅
#    复数的辅角：相位角
复数数组 = nf.fft(原函数值序列)
# 逆向傅里叶变换
y2 = nf.ifft(复数数组)
# 根据采样数量，采样周期计算fft的频率数组
freqs = nf.fftfreq(采样数量，采样周期)
```

案例：对方波执行fft

```python
"""
demo06_fft.py   傅里叶变换 拆方波
"""
import numpy as np
import matplotlib.pyplot as mp
import numpy.fft as nf
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.zeros(1000)
n = 1000
for i in range(1, n+1):
	y += 4/((2*i-1)*np.pi)*np.sin((2*i-1)*x)

mp.subplot(121)
mp.plot(x, y, label='y')
# 针对y 做fft 
y_complex = nf.fft(y)
print(y_complex.dtype, y_complex.shape)
# 逆向傅里叶变换
# y_complex[990:] = 0
y2 = nf.ifft(y_complex).real
mp.plot(x, y2, label='y2', c='orangered',
		linewidth=7, alpha=0.4)
mp.legend()

# 绘制频域图像  频率/能量
freqs = nf.fftfreq(n, x[1] - x[0])
powers = np.abs(y_complex)
mp.subplot(122)
mp.plot(freqs[freqs>0], powers[freqs>0], 
	c='orangered', label='Freq Domain')
mp.legend()
mp.show()
```

**基于傅里叶变换的频域滤波**

含噪信号是高能信号与低能噪声叠加的信号，可以通过fft的频域滤波实现降噪。

通过fft分解含噪信号，生成含噪频谱。在频域中去除低能噪声，留下高能信号，通过ifft生成新的音频数据。

**案例：**

1. 读取音频文件，获取文件基本信息：采样个数，采样周期。与每个采样点的声音信号值。绘制时域：时间\位移图像。

```python
#1.读取音频文件，获取文件基本信息：
#  采样个数，采样周期。与每个采样点的
#  声音信号值。绘制时域：时间\位移图像。

sample_rates, noised_sigs = \
	wf.read('../da_data/noised.wav')
print(sample_rates)
print(noised_sigs.shape, noised_sigs[0])
# 整理一组x坐标
times = np.arange(len(noised_sigs)) \
				  / sample_rates
mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178], 
		c='dodgerblue', label='Noised')
mp.legend()
```

2. 基于傅里叶变换，获取音频频域信息，绘制：频率/能量图像。

```python
# 基于傅里叶变换，获取音频频域信息，
# 绘制：频率/能量图像。
freqs=nf.fftfreq(times.size, 1/sample_rates)
noised_complex = nf.fft(noised_sigs)
noised_pows = np.abs(noised_complex)
mp.subplot(222)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs>0], noised_pows[freqs>0], 
	c='orangered', label='Noised')
mp.legend()
```

3. 将低能噪声去除，绘制频域 : 频率/能量图像.

```python
fund_freqs = freqs[noised_pows.argmax()]
# 所有噪声的索引
noised_inds=np.where(freqs != fund_freqs)
filter_complex = noised_complex.copy()
filter_complex[noised_inds] = 0
filter_pows = np.abs(filter_complex)
mp.subplot(224)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs>0], filter_pows[freqs>0], 
	c='orangered', label='Filter')
mp.legend()
```

4. 基于逆向傅里叶变换，生成新的音频信号，绘制音频时域图。

```python
filter_sigs = nf.ifft(filter_complex).real
mp.subplot(223)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178], 
		c='dodgerblue', label='Filter')
mp.legend()

# 5. 生成文件
wf.write('../da_data/out.wav', sample_rates,
	filter_sigs.astype('int16'))
```

### 概率分布

#### 二项分布(binomial)

二项分布就是重复n次独立事件的伯努利实验。在每次实验中只有两种结果，而且这两种结果发生与否相互对立，并且相互独立。事件发生与否的概率在每次独立实验中都保持不变。

```python
# 产生size个随机数，每个随机数来自n次尝试
# 的成功次数， 其中每次尝试成功的概率为p
r = np.random.binomial(n, p, size)
np.random.binomial(10, 0.8, 1)
```

案例：

```python
import numpy as np
# 投篮球案例  命中率0.8
a = np.random.binomial(10, 0.8, 100000)
print((a==8).sum() / 100000)
```

#### 超几何分布

```python
# 产生size个随机数， 每个随机数为在总样本中
# 随机抽取nsample个样本后好样本的个数。
# 总样本由ngood个好样本与nbad个坏样本组成。
r = np.random.hypergeometric(
	ngood, nbad, nsample, size)
```

随机数random的使用场景：

在数据预处理阶段，处理异常值与空白值时可能会用到随机数。

### 杂项功能

#### 排序

```python
a = np.msort(a)
# 联合间接排序   返回排序后的索引数组
indices = np.lexsort((次序列,主序列))
```

````python
"""
demo09_sort.py  排序
"""
import numpy as np

pros =['Apple','Huawei','Mi','Oppo','Vivo']
prices=[8888, 4999, 2999, 3999, 3999]
v =np.array([100, 70, 60, 50, 40])

# 普通排序
print(np.msort(v))
# 联合间接排序
indices = np.lexsort((-v, prices))
pros = np.array(pros)
print(pros[indices])
# 复数数组排序  先实部  后虚部
c = [1+2j, 1-3j, 2+2j]
print(np.sort_complex(c))
#插入排序
a = np.array([1,2,3,5,7,9])
b = np.array([6,8])
# 把b中的元素都插入a数组中
indices = np.searchsorted(a, b)
print(indices)
d = np.insert(a, indices, b)
print(d)
````

#### 插值

scipy提供了常见的插值器。可以通过一组散列的点基于某种插值器生成连续的函数。这样的话就可以传入未知的自变量，计算函数值。

案例：

```python
"""
demo10_interpolate.py 插值器
"""
import scipy.interpolate as si
import matplotlib.pyplot as mp
import numpy as np

# 搞一组散点
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 15)
dis_y = np.sinc(dis_x)
mp.scatter(dis_x, dis_y, s=60, marker='o',
	label='Points', c='red')

#通过散点设计出符合线性规律的插值器函数
#返回的linear是一个函数  可以：linear(x)
linear=si.interp1d(dis_x, dis_y, 'linear')
x = np.linspace(min_x, max_x, 1000)
y = linear(x)
mp.plot(x, y, c='dodgerblue', label='linear')

# 三次样条插值器   cubic
cubic=si.interp1d(dis_x, dis_y, 'cubic')
x = np.linspace(min_x, max_x, 1000)
y = cubic(x)
mp.plot(x, y, c='orangered', label='cubic')

mp.legend()
mp.show()
```

#### 积分

```python
import scipy.integrate  as si
# a：下限   b：上限
def f(x):
    pass
val = si.quad(f, a, b)
```

案例：

```python
"""
demo11_integrate.py  积分
"""
import numpy as np
import scipy.integrate as si

def f(x):
	return 2 * x**2 + 3*x + 4

val = si.quad(f, -5, 5)
print(val)
# val[0]：积分值    val[1]：积分误差
```









