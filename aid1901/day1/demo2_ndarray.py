import numpy as np

a = np.arange(0,5,1)
b = np.arange(0,10,2)
print(a,b)

z = np.zeros(10)
print(z)

o = np.ones(10)
print(o)

a = np.array([[1,2],[3,4],[5,6]])
print(a,a.shape)
print(np.zeros_like(a))
print(np.ones(5)/5)