import matplotlib.pyplot as mp

xarray = [1,2,3,4,5]
yarray = [98,12,83,64,35]
mp.plot(xarray,yarray)
mp.vlines(3.5,20,80)
mp.hlines(40,2,5)
mp.show()