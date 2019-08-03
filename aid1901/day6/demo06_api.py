# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_api.py  通用函数
"""
import numpy as np

a = np.arange(1, 11)
print(a.clip(min=4, max=8))
print(a.compress(np.all([(a<8), (a>3)], axis=0)))








