# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_filter.py  频域滤波
"""
import numpy as np
import matplotlib.pyplot as mp
import scipy.io.wavfile as wf
import numpy.fft as nf

# 1.
sample_rate, noised_sigs = \
    wf.read('../da_data/noised.wav')
print('sample_rate:', sample_rate)
print('noised_sigs:', noised_sigs.shape)
times = np.arange(len(noised_sigs))/sample_rate
print(times.shape)
mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.title('Time Domain')
mp.ylabel('Signal')
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178], 
    c='dodgerblue', label='Signals')
mp.legend()

#2. 
freqs = nf.fftfreq(times.size, 1/sample_rate)
complex_ary = nf.fft(noised_sigs)
powers = np.abs(complex_ary)
mp.subplot(222)
mp.title('Frequenct Domain')
mp.ylabel('power')
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs>0], powers[freqs>0],
    c='orangered', label='Noised')
mp.legend()

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

# 4. 
filter_sigs = nf.ifft(complex_filter).real
mp.subplot(223)
mp.ylabel('Signal')
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178], 
    c='dodgerblue', label='Signals')
mp.legend()

wf.write('../da_data/out.wav', sample_rate,
   filter_sigs.astype('i2'))

mp.tight_layout()
mp.show()





