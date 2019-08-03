#维度处理
import numpy as np
a = np.arange(1,9)
print(a)

b = a.reshape(2,4)#变维2＊4
print(a,b)

c = a.reshape(2,2,2)#变维2＊2*2
print(c)

a[0] = 999
print(a)
print(b)
print(c)
print(c.ravel())
print('==========================')

#复制变维
d = b.flatten()
print(d)
d[0] = 888
print(d)
print(b)

print('-------------------------')
#就地变维
print(c)
c.resize(2,4)
print(c)