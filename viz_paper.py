import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tsseg.omslr import *

import tsseg.greed
import tsseg.omslr
import tsseg.utils

import tsseg_dot.greed
import tsseg_dot.omslr
import tsseg_dot.utils

line_style = '-'

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def viz_seg(tsinfo, pvts, exp_no, seg_diff=0.5, savepath=''):
    '''
    input:
        tsinfo[list<numeric>]: time series data info
        pvts[list<int>]: pivots of segments
        seg_diff: adjust the position of pivots, the position would move to left
            for line segmentation, the position is 0.5
            for dot segmentation, the position is 1
    '''
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=150)
    ax.plot(tsinfo, line_style)
    for pvt in pvts:
        ax.vlines(pvt-seg_diff, min(tsinfo), max(tsinfo), 'r')
    fig.tight_layout()
    d, b = os.path.dirname(savepath), os.path.basename(savepath)
    fig.savefig(f'{d}/png/{exp_no}/{b}.png', format='png')
    fig.savefig(f'{d}/eps/{exp_no}/{b}.eps', format='eps')
    fig.clear()
    plt.close(fig)

def viz_errors(errors, savepath=''):
    '''
    input:
        errors[dict<label: list>]:
            a format for multiple plot of errors
            the key of dictionary would be label
            color code is default 
            tip: use collections.OrderdDict to control order of curves
    '''
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    xlen = min([len(v) for v in errors.values()])
    for k, v in errors.items():
        plt.plot(v[:xlen], line_style, label=k)
    fig.legend()
    fig.tight_layout()
    d, b = os.path.dirname(savepath), os.path.basename(savepath)
    fig.savefig(f'{d}/png/{b}.png', format='png')
    fig.savefig(f'{d}/eps/{b}.eps', format='eps')
    fig.clear()
    plt.close(fig)

def opt_subplot(position, tsinfo, title, pvts, delta):
    plt.subplot(position)
    plt.title(title, fontsize=18)
    plt.plot(tsinfo, line_style)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    for pvt in pvts:
        plt.vlines(pvt-delta, min(tsinfo), max(tsinfo), 'r')


def viz_kseg_result(exp_no, k):
    makedir(f'result/ucr_omslr/viz/png')
    makedir(f'result/ucr_omslr/viz/eps')
    dataset = load_dataset('result/ucr_omslr/dataset.txt')
    # dsnames = eval(open(f'result/{folder}/{dname}_name.txt').read())
    tsinfo = dataset[exp_no]

    get_pivot = tsseg_dot.omslr.get_pivots
    gamma_line_minmax = np.load(f'result/ucr_omslr/{exp_no}/gamma_line_minmax.npy')
    gamma_line_gmse = np.load(f'result/ucr_omslr/{exp_no}/gamma_line_gmse.npy')
    gamma_dot_minmax = np.load(f'result/ucr_omslr/{exp_no}/gamma_dot_minmax.npy')
    gamma_dot_gmse = np.load(f'result/ucr_omslr/{exp_no}/gamma_dot_gmse.npy')

    plt.figure(figsize=(12,12), dpi=300)
    opt_subplot(221, tsinfo, 'Seg by Line, Optimal MinMax', get_pivot(gamma_line_minmax[:k+1]), 0.5)
    opt_subplot(222, tsinfo, 'Seg by Line, Optimal GMSE',   get_pivot(gamma_line_gmse[:k+1]), 0.5)
    opt_subplot(223, tsinfo, 'Seg by Dot, Optimal MinMax',  get_pivot(gamma_dot_minmax[:k+1]), 1)
    opt_subplot(224, tsinfo, 'Seg by Dot, Optimal GMSE',    get_pivot(gamma_dot_gmse[:k+1]), 1)
    plt.tight_layout()
    plt.savefig(f'result/ucr_omslr/viz/png/{exp_no}_{k+1}.png', format='png')
    plt.savefig(f'result/ucr_omslr/viz/eps/{exp_no}_{k+1}.eps', forpat='ess')
    plt.cla()
    plt.clf()
    plt.close()

FIG_SIZE = (16,12)
FIG_SIZE = (8,6)

def load_dataset(path):
    return [eval(line) for line in filter(None, open(path).read().split('\n'))]

def load_exp(exp_name, exp_no):
    exp_dir = f'result/{exp_name}/exp/{exp_no}/'
    op_errs = eval(open(exp_dir + 'omslr.txt').read())
    
    bu_exp = open(exp_dir + 'bu.txt').read().split('\n')
    bu_errs, bu_pvts = eval(bu_exp[0]), eval(bu_exp[1])
    
    td_exp = open(exp_dir + 'td.txt').read().split('\n')
    td_errs, td_pvts = eval(td_exp[0]), eval(td_exp[1])

    bu_errs = [float(e) for e in bu_errs]
    td_errs = [float(e) for e in td_errs]
    return dict(op_errs=op_errs, bu_errs=bu_errs, bu_pvts=bu_pvts,td_errs=td_errs, td_pvts=td_pvts )

def draw_seg_and_err(exp_name):
    makedir(f'result/{exp_name}/viz/3errs/png')
    makedir(f'result/{exp_name}/viz/3errs/eps')
    dataset = load_dataset(f'result/{exp_name}/dataset.txt')
    for i in range(len(dataset)):
        makedir(f'result/{exp_name}/viz/segment/png/{i}')
        makedir(f'result/{exp_name}/viz/segment/eps/{i}')
        print(i)
        result = load_exp(exp_name, i)
        gamma = np.load(f'result/{exp_name}/exp/{i}/gamma.npy')
        for k in range(4):
            continue
            pvts = get_pivots(gamma[:k+2])
            viz_seg(dataset[i], pvts, exp_no=i, savepath=f'result/{exp_name}/viz/segment/op_{i}_{k}')
            pvts = result['td_pvts'][:k+1]
            viz_seg(dataset[i], pvts, exp_no=i, savepath=f'result/{exp_name}/viz/segment/td_{i}_{k}')
            pvts = result['bu_pvts'][:k+1]
            viz_seg(dataset[i], pvts, exp_no=i, savepath=f'result/{exp_name}/viz/segment/bu_{i}_{k}')

        viz_errors({'OMSLR': result['op_errs'], 
                    'Top Down': result['td_errs'],
                    'Bottom Up': result['bu_errs']}, 
                    savepath=f'result/{exp_name}/viz/3errs/{i}')

def draw_omslr_criteria():
    viz_kseg_result(0, 10)
    viz_kseg_result(0, 12)
    viz_kseg_result(4, 6)
    viz_kseg_result(5, 8)
    viz_kseg_result(6, 6)
    viz_kseg_result(6, 9)
    viz_kseg_result(7, 20)
    viz_kseg_result(8, 11)
    viz_kseg_result(11, 12)


def main():
    # viz_seg([2,6,1,3,7,3,2,1,5,8], [4,8])
    viz_errors(dict(a=[1,2,6], b=[2,1,3], c=[7,8,3]))
    plt.show()

if __name__ == '__main__':
    # main()
    draw_seg_and_err('ucr_3algo')
    draw_seg_and_err('simup')
    draw_seg_and_err('simun')
    # draw_omslr_criteria()