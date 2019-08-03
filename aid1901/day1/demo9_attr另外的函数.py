import numpy as np

a = np.array([
    [1+1j,2+4j,3+7j],
    [4+2j,5+5j,6+8j],
    [7+3j,8+6j,9+9j],
])
print('shape:',a.shape)
print('dtype:',a.dtype)
print('ndim:',a.ndim)
print('itemsize:',a.itemsize)
print('nbytes:',a.nbytes)
print(a.real,a.imag,sep='\n')
print('T:',a.T)

print([elem for elem in a.flat])