import matplotlib.pyplot as mp

mp.figure('Subplot',facecolor='lightgray')
#绘制子图
for i in range(1,10):
    mp.subplot(3,3,i)
    mp.text(0.5,0.5,i,size=36,alpha=0.5,ha='center',va='center')
    mp.xticks([])
    mp.yticks([])
mp.tight_layout()
mp.show()