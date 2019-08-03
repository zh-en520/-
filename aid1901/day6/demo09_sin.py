# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_sin.py   合成方波
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y1 = 4*np.pi * np.sin(x)
y2 = 4*np.pi/3 * np.sin(3*x)

n = 1000
y = np.zeros(1000)
for i in range(1, n+1):
    y += 4*np.pi/(2*i-1) * np.sin((2*i-1)*x)

mp.plot(x, y1, label='y1', alpha=0.3)
mp.plot(x, y2, label='y2', alpha=0.3)
mp.plot(x, y, label='y')
mp.legend()
mp.show()


