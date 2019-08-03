# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_eig.py  特征值提取
"""
import numpy as np

A = np.mat('4 8 9; 3 5 8; 1 9 3')
print(A)
# 提取特征信息
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
# 逆向推方阵
A2 = eigvecs * np.diag(eigvals) * eigvecs.I
print(A2)

# 如果只保留一部分特征值 则：
eigvals[2:] = 0
A3 = eigvecs * np.diag(eigvals) * eigvecs.I
print(A3)


