import numpy as np
import matplotlib.mlab as norm
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
    
#生成需要的数据

p_w1 = 0.35
p_w2 = 1 - p_w1

x = np.linspace(-10, 25, 500)
y1 = 0.2 * norm.normpdf(x, 0, 1) + \
     0.4 * norm.normpdf(x, 4, 0.8) + \
     0.4 * norm.normpdf(x, 7, 3)
y2 = 0.2 * norm.normpdf(x, 3, 2) + \
     0.8 * norm.normpdf(x, 9, 3.5)
#z1 = p_w1 * y1 / (p_w1 * y1 + p_w2 * y2)
#print(p_w1 * y1 + p_w2 * y2)
z1 = p_w1 * y1
z2 = p_w2 * y2
plt.plot(x,z1,label='w1')
plt.plot(x,z2,label='w2')
# plt.show()

y = np.linspace(0, 0.1, 500)
xv, yv = np.meshgrid(x, y)
X = np.c_[xv.ravel(), yv.ravel()]
# print(xv.shape,yv.shape,X.shape)
z = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    #for j in range(100):
    tx,ty = X[i]
    if (ty<z1[i%500] or ty<z2[i%500]):
        z[i] = 1
    if (ty<z1[i%500] and ty<z2[i%500])and(z1[i%500] < z2[i%500]):
        z[i] = 2
    if (ty<z1[i%500] and ty<z2[i%500])and(z2[i%500] < z1[i%500]):
        z[i] = 3      
z = z.reshape(xv.shape)

c = mcolors.ColorConverter().to_rgb
mc = make_colormap([c('white'), 0.25, c('yellow'), 0.5, c('deepskyblue'),0.75,c('lawngreen')])
plt.contourf(xv,yv,z,cmap=mc)
plt.colorbar()
plt.legend()
plt.show()

        
