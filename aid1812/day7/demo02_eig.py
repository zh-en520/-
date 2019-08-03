"""
demo02_eig.py  特征值提取
"""
import numpy as np
A = np.mat('2 3 1; 4 7 4; 8 5 2')
print(A)
vals, vecs = np.linalg.eig(A)
print(vals)
print(vecs, type(vecs))
# 回推导原方阵
S = vecs * np.diag(vals) * vecs.I
print(S.real)
# 抹掉一部分特征值 推导原方阵
vals[2:] = 0
S = vecs * np.diag(vals) * vecs.I
print(S.real)
