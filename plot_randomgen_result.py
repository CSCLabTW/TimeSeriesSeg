import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tsseg.utils import *
from tsseg.greed import *
from tsseg.omslr import *

def load_file(fname):
    lines = list(filter(None, open(fname).read().split('\n')))
    li = [float(line.split('\t')[1]) for line in lines]
    return li

def load_data():
    randgen = []
    optimal = []
    for i in range(1, 6):
        randgen.append(load_file('fwn{}.txt'.format(i)))
        optimal.append(load_file('fwo{}.txt'.format(i)))
    randgen = np.array(randgen)
    optimal = np.array(optimal)
    return randgen, optimal

def load_pivots_info(num_seg=5):
    lines = list(filter(None, open('seg_testing_1_with_noise/gen_ds_noise.txt').read().split('\n')))
    ds = [eval('[{}]'.format(line)) for line in lines]

    slines = list(filter(None, open('seg_testing_1_with_noise/fwo{}.txt'.format(num_seg)).read().split('\n')))
    cols = [line.split('\t') for line in slines]
    pvts, errs = np.array([[eval(col[0]), eval(col[1])] for col in cols]).transpose().tolist()

    return ds, pvts, errs



ds, pvts, errs = load_pivots_info()
for i in range(100):
    plt.figure(figsize=(16, 12), dpi=300)
    sigma, beta, alpha = iter_sigma(ds[i])
    plt.plot(ds[i], '.-')
    curr_pvts = [0] + pvts[i] + [len(ds[i])]
    errors = [sigma[curr_pvts[i], curr_pvts[i+1]-1] for i in range(len(curr_pvts)-1)]
    plt.ylim(0,1050)
    for p in pvts[i]:
        plt.vlines(p-0.5, min(ds[i]), max(ds[i]), 'r')
    for j, err in enumerate(errors):
        plt.text((curr_pvts[j]+curr_pvts[j+1])/2-1, 50, '{:6.4f}'.format(err))
    plt.savefig('seg_testing_1_with_noise/segments_viz/{}.png'.format(i))
    plt.cla()
    plt.clf()