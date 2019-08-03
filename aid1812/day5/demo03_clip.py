"""
demo03_clip.py  数组处理函数
"""
import numpy as np

a = np.arange(1, 11)
print(a)
print(a.clip(min=5, max=8))
print(np.clip(a, 5, 8))

# print(a.compress((a>3) and (a<8)))
print(a.compress(np.all([a>3, a<8], axis=0)))
