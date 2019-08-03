# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_eig.py 提取图像特征值
"""
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as mp

image = sm.imread('../da_data/lily.jpg', True)
print(image, image.shape)
# 提取 image 的特征
eigvals, eigvecs = np.linalg.eig(image)
print(eigvals.shape)
print(eigvecs.shape)
# 逆向生成图像
# 只保留50个特征值
eigvals[50:] = 0
image2 = np.mat(eigvecs)\
         * np.diag(eigvals)\
         * np.mat(eigvecs).I
print(image2.shape)

# 奇异值分解
U, sv, V = np.linalg.svd(image)
# 抹掉部分奇异值，生成新图片
sv[10:] = 0
image3 = np.mat(U) * np.mat(np.diag(sv))\
                   * np.mat(V) 
# 绘制
mp.figure('EIG Image')
mp.subplot(221)
mp.imshow(image, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.subplot(222)
mp.imshow(image2.real, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.subplot(224)
mp.imshow(image3.real, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()
mp.show()








