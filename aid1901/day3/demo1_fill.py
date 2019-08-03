import matplotlib.pyplot as mp
import numpy as np

n = 1000
x = np.linspace(0,8*np.pi,n)
sinx = np.sin(x)
cosx = np.cos(x/2)/2

mp.figure('Fill',facecolor='lightgray')
mp.title('Fill',fontsize=18)
mp.grid(linestyle=':')
mp.plot(x,sinx,color='dodgerblue',label='sinx')
mp.plot(x,cosx,color='orangered',label='cosx')

#填充
mp.fill_between(x,sinx,cosx,sinx<cosx,color='dodgerblue',alpha=0.5)
mp.fill_between(x,sinx,cosx,sinx>cosx,color='orangered',alpha=0.5)

mp.legend()
mp.show()