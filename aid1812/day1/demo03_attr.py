"""
demo03_attr.py 属性操作
"""
import numpy as np

# 数组维度处理
ary = np.array([1, 2, 3, 4, 5, 6])
print(ary, ary.shape)
print(ary[4])
ary.shape = (2, 3)
print(ary, ary.shape)
print(ary[1][1])

# 数组元素类型
print('-'*45)
print(ary, ary.dtype)
# ary.dtype = np.int64
# print(ary, ary.dtype)
b = ary.astype('float32')
print(b, b.dtype)

#数组元素的个数
print('-'*45)
print(ary.size)
print(len(ary))

# 数组元素的索引
print('-'*45)
ary = np.arange(1, 9)
ary.shape=(2, 2, 2)
print(ary, ary.shape)
print(ary[0])   # 0页数据
print(ary[0][0])   # 0页0行数据
print(ary[0][0][0])   # 0页0行0列数据
print(ary[0, 0, 0])   # 0页0行0列数据

# 使用for循环，把ary数组中的元素都遍历出来。
for i in range(ary.shape[0]):
    for j in range(ary.shape[1]):
        for k in range(ary.shape[2]):
            print(ary[i, j, k], end=' ')
