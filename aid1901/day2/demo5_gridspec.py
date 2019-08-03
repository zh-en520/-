import matplotlib.gridspec as mg
import matplotlib.pyplot as mp

mp.figure('123')
gs = mg.GridSpec(3,3)

mp.subplot(gs[0,:2])
mp.text(0.5,0.5,1,ha='center',va='center',size=36)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(gs[:2,2])
mp.text(0.5,0.5,2,ha='center',va='center',size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[1,1])
mp.text(0.5,0.5,3,ha='center',va='center',size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[1:,0])
mp.text(0.5,0.5,4,ha='center',va='center',size=36)
mp.xticks([])
mp.yticks([])

mp.subplot(gs[2,1:])
mp.text(0.5,0.5,5,ha='center',va='center',size=36)
mp.xticks([])
mp.yticks([])

mp.plot()
mp.show()