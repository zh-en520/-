"""
demo08_random.py 概率分布
"""
import numpy as np
# 投篮球案例  命中率0.8
a = np.random.binomial(10, 0.8, 100000)
print((a==8).sum() / 100000)

# 超几何分布
b = np.random.hypergeometric(6, 4, 3, 100000)
print((b==0).sum() / 100000)
print((b==1).sum() / 100000)
print((b==2).sum() / 100000)
print((b==3).sum() / 100000)

# 正态分布
print(np.random.normal(0, 1, 10))
# 平均分布
print(np.random.uniform(1, 10, 10))