import matplotlib.pyplot as mp

mp.figure('FigureA',facecolor='lightgray')
mp.figure('FigureB',facecolor='gray')
mp.title('Figure B',fontsize=18)
mp.xlabel('X label',fontsize=14)
mp.ylabel('Y label',fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.tight_layout()
mp.show()