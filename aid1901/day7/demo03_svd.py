# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_svd.py  奇异值分解
"""
import numpy as np

A = np.mat('4 8 9; 3 5 8')
print(A)
# 奇异值分解
U, sv, V = np.linalg.svd(A, full_matrices=True)
print(U.shape, V.shape)
print(sv)
print(U * np.diag(sv) * V)
