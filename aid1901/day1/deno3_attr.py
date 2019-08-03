#数组属性基本操作
import numpy as np
ary = np.array([1,2,3,4,5])
print(type(ary),ary.shape)

ary2 = np.array([[1,2,3],[4,5,6]])
print(type(ary2),ary2.shape)

#变维
ary2.shape = (3,2)
print(ary2,ary2.shape)

#测试dtype属性：元素类型
print('---------------')
print(ary2,ary2.dtype)
# ary2.dtype = 'float64'#改类型不推荐这样改
# print(ary2,ary2.dtype,ary2.shape)
print('-----------')
ary3 = ary2.astype('float64')#astype方法需要赋值，因为它有返回值
print(ary3,ary3.dtype,ary3.shape)

#size属性：元素个数
print('--------------')
print(ary2,ary2.size,len(ary2),ary2.shape[1])

#测试index属性
print('-------------')
a = np.arange(1,9)
a.shape = (2,2,2)
print(a,a[1][0][0])
print(a[1,0,0])

#尝试使用for 循环遍历所有元素
print('------------------------')
ary4 = np.arange(1,28)
ary4.shape = (3,3,3)
print(ary4)
for i in range(ary4.shape[0]):
    for j in range(ary4.shape[1]):
        for x in range(ary4.shape[2]):
            print(ary4[i,j,x],end=' ')



