"""
demo07_mask.py  掩码操作
"""
import numpy as np

a = np.arange(1, 8)
mask = a > 5
print(a)
print(mask)
print(a[mask])

# 输出100以内3与7的公倍数。
a = np.arange(1, 100)
mask = (a%3==0) & (a%7==0)
print(a[mask])

# 利用掩码运算对数组进行排序
p = np.array(['Mi', 'Apple', 'Huawei', 'Oppo'])
r = [1, 3, 2, 0]
print(p[r])

