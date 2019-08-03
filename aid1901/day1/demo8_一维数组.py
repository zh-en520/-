import numpy as np

a = np.arange(1,7)
b = np.arange(7,13)
a = a.ravel()
b = b.ravel()
c = np.row_stack((a,b))
d = np.column_stack((a,b))
print(c,d,sep='\n')