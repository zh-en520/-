import numpy as np

a = np.arange(1,7).reshape(3,2)
b = np.arange(7,13).reshape(3,2)
#垂直方向操作
c = np.vstack((a,b))
print(c)
a,b = np.hsplit(c,2)#把c拆分成２份赋值给a,b
print(a,b,sep='\n')



#垂直方向操作
# c = np.hstack((a,b))
# print(c)
# a,b = np.hsplit(c,2)#把c拆分成２份赋值给a,b
# print(a,b,sep='\n')
#
#
# #垂直方向操作
# c = np.dstack((a,b))
# print(c)
# a,b = np.dsplit(c,2)#把c拆分成２份赋值给a,b
# print(a,b,sep='\n')