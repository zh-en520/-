#掩码操作
import numpy as np

a = np.arange(1,7)
mask = a % 2 == 0
b = a[mask]
print('a',a)
print('mask',mask)
print('b',b)

#输出100以内　３与７的公倍数
print('----------------------')
a = np.arange(1,101)
mask = (a % 3 == 0) & (a % 7 == 0)
b = a[mask]
print('b:',b)


#基于索引的掩码操作
print('----------------------')
a = np.array(['A','B','C','D','E'])
mask = [0,3,4,2,4,3,4,3,2,2]
print(a[mask])