import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tsseg.utils import *
from tsseg.omslr import *
from tsseg.greed import *


def call_by_sys():
    if len(sys.argv) < 3:
        print('python plot_segmentation.py exp_number seg_count')
        exit(-1)
    exp_no = int(sys.argv[1])   # experiment number [int]
    seg_cn = int(sys.argv[2])   # segmentation count [int]
    exp_type = 'thesis_dot_global'
    if len(sys.argv) == 4:
        filename = sys.argv[3]
    else:
        filename = 'out.png'
    draw(exp_no, seg_cn, exp_type, filename)

def call_by_tasks():
    
    pass

def draw(exp_no, seg_cn, exp_type, filename):
    dataset = [eval(line) for line in list(filter(None, open('{}/realdataset.txt'.format(exp_type)).read().split('\n')))]
    gamma = np.load('{}/exp/{}/omslr_gamma.npy'.format(exp_type, exp_no))

    tsinfo = dataset[exp_no]
    pvts = get_pivots(gamma[:seg_cn+1])

    plt.figure(figsize=(20, 6), dpi=150)
    plt.plot(tsinfo, '.-')
    for pvt in pvts:
        if 'line' in exp_type:
            plt.vlines(pvt-0.5, min(tsinfo), max(tsinfo), 'r')
        else:
            plt.vlines(pvt-1, min(tsinfo), max(tsinfo), 'r')
    
    plt.savefig(filename)

def main():
    pass

if __name__ == '__main__':
    main()