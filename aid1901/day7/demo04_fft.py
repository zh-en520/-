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
print(freqs.shape)
print(complex_ary.shape)
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
print(power.shape)
mp.plot(freqs, power )

mp.tight_layout()
mp.show()


