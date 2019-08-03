"""
demo05_poly.py 多项式演示
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-20, 20, 1000)
y = 3*x**3 + 3*x**2 - 1000*x + 1
# 
P = [3, 3, -1000, 1]
Q = np.polyder(P)
xs = np.roots(Q)
print(xs)
ys = np.polyval(P, xs)

mp.plot(x, y)
mp.scatter(xs, ys, s=60, c='red', marker='o')
mp.show()

