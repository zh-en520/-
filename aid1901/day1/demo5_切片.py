#数组切片
import numpy as np

a = np.arange(1,10)
print(a)
print(a[:3])
print(a[3:6])
print(a[6:])
print(a[::-1])
print(a[:-4:-1])
print(a[::])
print(a[::3])
print(a[1::3])
print(a[2::3])

#多维数组切片
print('---------------')
b = a.reshape(3,3)
print(b)
#前两行的前两列
print(b[:2,:2])