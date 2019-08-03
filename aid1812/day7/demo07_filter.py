"""
demo07_filter.py  滤波
"""
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

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

#3. 
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

#4 . 
filter_sigs = nf.ifft(filter_complex).real
mp.subplot(223)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178], 
		c='dodgerblue', label='Filter')
mp.legend()

5.
wf.write('../da_data/out.wav', sample_rates,
	filter_sigs.astype('int16'))

mp.tight_layout()
mp.show()