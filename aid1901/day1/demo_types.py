#自定义符合类型
import numpy as np
data = [
    ('zs',[60,61,65],15),
    ('ls',[60,61,65],15),
    ('ww',[60,61,65],15),
]

#第一种设置dtype的方式
a = np.array(data,dtype='U2,3int32,int32')
print(a[1]['f2'])#f2是字段的别名
print('-----------------------')

#第二种设置dtype的方式
b = np.array(data,dtype=[
    ('name','str',2),
    ('scores','int32',3),
    ('age','int32',1),
])
print(b)
print(b[1]['scores'])
print('-----------------')

#第三种设置dtype的方式
c = np.array(data,dtype={
    'names':['name','scores','age'],
    'formats':['U2','3int32','int32']
})
print(c[1]['name'])
print('----------------------')

#第四种设置dtype的方式(仅作了解)
d = np.array(data,dtype={
    'name':('U3',0),
    'scores':('3int32',16),
    'age':('int32',28)
})
print(d[1]['age'])
print('--------------------')

#测试日期类型数组
a = np.array([
    '2011','2012-01-01',
    '2011-01-01','2011-02-01'
])
print(a.dtype)#<U10小端字节序
b = a.astype('M8[D]')
print(b,b.dtype)
print(b[1]-b[0])