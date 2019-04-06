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
#print(y1.sum(),y2.sum(),z1.sum(),z2.sum())

l = y1/y2
pe_1 = 0.2
plt.plot(x,l)
plt.show()
dl = np.linspace(0,l.max(),500)
for t in range(500):
    e_y1 = y1[l < dl[t]]
    e_y2 = y2[l > dl[t]]
    if (e_y1.sum()*0.07 > 0.15 and e_y1.sum()*0.07 < 0.25):
        plt.fill(x[l < dl[t]],l[l < dl[t]])
        plt.fill(x[l > dl[t]],l[l > dl[t]])
        plt.show()
        print(t,dl[t],e_y1.sum()*0.07,e_y2.sum()*0.07)
#print(y1[l<0.64].sum(),y2[l>0.64].sum())
        
