"""
demo05_grid.py  网格布局
"""
import matplotlib.pyplot as mp
import matplotlib.gridspec as mg

mp.figure('Grid Layout', facecolor='lightgray')
mg = mp.GridSpec(3, 3)
# 0行的前两个单元格合并构建子图
mp.subplot(mg[0, :2])
mp.text(0.5, 0.5, 1, alpha=0.5, fontsize=36,
	ha='center', va='center')
mp.xticks([])
mp.yticks([])

mp.subplot(mg[:2, 2])
mp.text(0.5, 0.5, 2, alpha=0.5, fontsize=36,
	ha='center', va='center')
mp.xticks([])
mp.yticks([])

mp.subplot(mg[1, 1])
mp.text(0.5, 0.5, 3, alpha=0.5, fontsize=36,
	ha='center', va='center')
mp.xticks([])
mp.yticks([])

mp.subplot(mg[1:, 0])
mp.text(0.5, 0.5, 4, alpha=0.5, fontsize=36,
	ha='center', va='center')
mp.xticks([])
mp.yticks([])

mp.subplot(mg[2, 1:])
mp.text(0.5, 0.5, 5, alpha=0.5, fontsize=36,
	ha='center', va='center')
mp.xticks([])
mp.yticks([])
mp.tight_layout()


mp.figure('Flow Layout', facecolor='lightgray')
mp.axes([0.03, 0.03, 0.5, 0.94])
mp.axes([0.56, 0.03, 0.4, 0.54])

mp.show()