import matplotlib.pyplot as mp

locators = ['mp.MultipleLocator(1)',
            'mp.NullLocator()',
            'mp.MaxNLocator(nbins=4)',
            'mp.AutoLocator()'
            ]

mp.figure('Locators',facecolor='lightgray')


for i,locator in enumerate(locators):
    mp.subplot(len(locators),1,i+1)
    ax = mp.gca()
    #设置坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    mp.ylim(-1,1)
    mp.xlim(0,10)
    ax.spines['bottom'].set_position(('data',0))
    mp.xticks([])
    mp.yticks([])

    #多点定位器 eval()
    loc1 = eval(locator)
    #设置主刻度定位器对象
    ax.xaxis.set_major_locator(loc1)

    loc2 = mp.MultipleLocator(0.1)
    #设置主刻度定位器对象
    ax.xaxis.set_minor_locator(loc2)

    mp.tight_layout()
mp.show()