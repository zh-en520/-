"""
demo04_matrix.py  矩阵示例
"""
import numpy as np

ary = np.arange(1, 7).reshape(2, 3)
print(ary, type(ary))
m = np.matrix(ary, copy=True)
print(m, type(m))
ary[0,0] = 999
print(ary, type(ary))
print(m, type(m))

# 字符串拼块规则 
ary = np.mat('1 2 3; 4 5 6; 7 8 9') 
print(ary, type(ary), ary.dtype)
b = np.arange(1, 10).reshape(3, 3)
print(b.dot(b))

# 测试矩阵的逆矩阵
a = np.mat('1 2 3; 4 4 3; 2 3 6')
#a = np.mat('1 2 3; 4 5 6; 7 8 9')
print('-' * 45)
print(a)
print(a.I)
print(a * a.I)
print(a.I * a)


print('-' * 45)
A = np.mat('3 3.2; 3.5 3.6')
B = np.mat('118.4; 135.2')
x = np.linalg.lstsq(A, B)[0]
print(x)
persons = A.I * B
print(persons)

n = 32
m = np.mat('1 1; 1 0')
print((m**n)[0,0])