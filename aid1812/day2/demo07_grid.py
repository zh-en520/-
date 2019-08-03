"""
demo07_grid.py  刻度网格线
"""
import matplotlib.pyplot as mp
import numpy as np

y = np.array([1, 10, 100, 1000, 100, 10, 1])

mp.figure('GridLine', facecolor='lightgray')

mp.subplot(211)
mp.title('GridLine', fontsize=14)
mp.ylabel('Y', fontsize=12)
# 设置刻度定位器与网格线
ax = mp.gca()
ax.xaxis.set_major_locator(
			mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(
			mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(
			mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(
			mp.MultipleLocator(50))
ax.grid(which='major', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.75)
ax.grid(which='minor', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.25)

mp.plot(y, 'o-', color='dodgerblue')

mp.subplot(212)
mp.ylabel('Y', fontsize=12)
# 设置刻度定位器与网格线
ax = mp.gca()
ax.xaxis.set_major_locator(
			mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(
			mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(
			mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(
			mp.MultipleLocator(50))
ax.grid(which='major', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.75)
ax.grid(which='minor', axis='both',
	color='orange', linestyle='-', 
	linewidth=0.25)
mp.semilogy(y, 'o-', color='dodgerblue')

mp.show()
