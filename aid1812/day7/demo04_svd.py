"""
demo04_svd.py   奇异值分解
"""
import numpy as np

M = np.mat('2 4 8; 5 9 1')
print(M)
U, sv, V = np.linalg.svd(
		M, full_matrices=False)
print(U, type(U))
print(V, type(V))
print(sv, type(sv))
sv[1] = 0
# 推导原矩阵
M2 = U * np.diag(sv) * V
print(M2)

