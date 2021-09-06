import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tsseg.utils import *
from tsseg.greed import *
from tsseg.omslr import *

def get_dataset():
    lines = list(filter(None, open('gen_ds_noise.txt').read().split('\n')))
    ds = []
    for line in lines:
        ds.append([float(i) for i in line.split(', ')])
    return ds

def draw_segmentation(T, pivots, sub=None, **kwargs):
    if sub: plt.subplot(sub)
    for pvt in pivots:
        plt.vlines(pvt-0.5, min(T), max(T), 'r', alpha=0.5)
    plt.plot(T, '.-')
    if 'title' in kwargs:
        plt.title(kwargs['title'])

ds = get_dataset()
t = ds[50]
# t = t[:28]

for ps in range(1, 6):
    print('len(p): {}'.format(ps))
    fwn = open('fwn{}.txt'.format(ps), 'w')
    fwo = open('fwo{}.txt'.format(ps), 'w')
    for i, t in enumerate(ds):
        print('ds: {}'.format(i))
        sigma, beta, alpha = iter_sigma(t)
        pvts = naive_minmax(t, ps, sigma)
        pvt1 = pvts
        pvts = [0] + pvts + [len(t)]
        err1 = max([sigma[pvts[i], pvts[i+1]-1] for i in range(len(pvts)-1)])
        
        gamma, rho = omslr_minmax(t, ps+1, sigma)
        pvts = get_pivots(gamma)
        pvt2 = pvts
        pvts = [0] + pvts + [len(t)]
        err2 = max([sigma[pvts[i], pvts[i+1]-1] for i in range(len(pvts)-1)])
        fwn.write('{}\t{}\n'.format(pvt1, err1))
        fwo.write('{}\t{}\n'.format(pvt2, err2))
    fwn.close()        
    fwo.close()    
# l, r = [], []
# for i in range(1, 27):

#     l.append(sigma[0,i-1])
#     r.append(sigma[i,-1])
# plt.plot(l, 'b.-')
# plt.plot(r, 'r.-')
# plt.show()


# print(sigma[0,8])
# print(regression_error(t[:9]))
# print(sigma[9,-1])
# print(regression_error(t[9:]))

# print()


# print(sigma[0,5])
# print(regression_error(t[:6]))
# print(sigma[6,-1])
# print(regression_error(t[6:]))

# print(sigma[0,6])
# print(regression_error(t[:7]))
# print(sigma[7,-1])
# print(regression_error(t[7:]))
# plt.show()