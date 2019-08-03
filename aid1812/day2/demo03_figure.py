"""
demo03_figure.py  窗口操作
"""
import matplotlib.pyplot as mp

mp.figure('Figure A', facecolor='gray')
mp.plot([1,2,3,2,3,1,7])
mp.figure('Figure B', facecolor='lightgray')
mp.plot([10,20,30,20,30,10,70])
mp.figure('Figure A')
mp.plot([6,3,6,4,8,3,1])
# 窗口常见的属性参数设置方法
mp.title('Figure A', fontsize=16)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

#mp.tight_layout()  # 使用紧凑布局
mp.show()

