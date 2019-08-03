import matplotlib.pyplot as mp
import numpy as np


x = np.arange(12)
apples=[92,34,75,32,96,52,36,10,23,41,22,35]
oranges=[23,43,46,58,23,74,56,72,38,95,63,9]

mp.figure('Fill',facecolor='lightgray')
mp.title('Fill',fontsize=18)
mp.grid(linestyle=':')
mp.bar(x-0.2,apples,0.3,0,color='limegreen',label='apples',align='center')
mp.bar(x+0.2,oranges,0.45,0,color='orange',label='orange',align='center')
mp.xticks(x,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

mp.legend()
mp.show()