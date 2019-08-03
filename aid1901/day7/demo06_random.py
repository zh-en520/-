# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_random.py  随机数
"""
import numpy as np

# 二项分布
# 命中率 0.8  
r = np.random.binomial(10, 0.8, 10)
print(r)
# 求：命中率0.8时，投10球进8球的概率
# 投100000轮，看看有多少轮进了8个球?
r = np.random.binomial(10, 0.8, 100000)
print((r==1).sum()/100000)
print((r==2).sum()/100000)
print((r==3).sum()/100000)
print((r==4).sum()/100000)
print((r==5).sum()/100000)
print((r==6).sum()/100000)
print((r==7).sum()/100000)
print((r==8).sum()/100000)
print((r==9).sum()/100000)
print((r==10).sum()/100000)

# 超几何分布  
# 7个坏的3个好的，模3个球，返回好球的个数
r = np.random.hypergeometric(7, 3, 3, 100000)
print((r==0).sum()/100000)
print((r==1).sum()/100000)
print((r==2).sum()/100000)
print((r==3).sum()/100000)
a = np.random.normal(1,0.1,9)
print('a:',a)
