"""
demo09_sort.py  排序
"""
import numpy as np

pros =['Apple','Huawei','Mi','Oppo','Vivo']
prices=[8888, 4999, 2999, 3999, 3999]
v =np.array([100, 70, 60, 50, 40])

# 普通排序
print(np.msort(v))
# 联合间接排序
indices = np.lexsort((-v, prices))
pros = np.array(pros)
print(pros[indices])
# 复数数组排序  先实部  后虚部
c = [1+2j, 1-3j, 2+2j]
print(np.sort_complex(c))
#插入排序
a = np.array([1,2,3,5,7,9])
b = np.array([6,8])
# 把b中的元素都插入a数组中
indices = np.searchsorted(a, b)
print(indices)
d = np.insert(a, indices, b)
print(d)
