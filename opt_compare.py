#! /usr/bin/python3 

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
import logging

import tsseg.greed
import tsseg.omslr
import tsseg.utils

import tsseg_dot.greed
import tsseg_dot.omslr
import tsseg_dot.utils

def load_dataset(folder, dname):
    dataset = [eval(line) for line in list(filter(None, open(f'result/{folder}/{dname}.txt').read().split('\n')))]
    return dataset

def calc_omslr_compare():
    fw = open('result/omslr/log.txt', 'w')
    folder = 'thesis_line_minmax'
    dname = 'realdataset'
    dataset = load_dataset(folder, dname)
    dsnames = eval(open(f'result/{folder}/{dname}_name.txt').read())
    for i, tsinfo in enumerate(dataset):
        print(i)
        if not os.path.isdir(f'result/omslr/{i}'):
            os.makedirs(f'result/omslr/{i}')
        op_k = min(len(tsinfo)//3, 200)
        fw.write(f'exp_no: {i}, k: {op_k}\n')
        sigma_minm, sigma_gmse = iter_sigma_mats(tsinfo)
        np.save(f'result/omslr/{i}/sigma_minmax.npy', sigma_minm)
        np.save(f'result/omslr/{i}/sigma_gmse.npy', sigma_gmse)
        
        # line minmax
        gamma, rho = tsseg.omslr.omslr_minmax(tsinfo, op_k, sigma_minm)
        np.save(f'result/omslr/{i}/gamma_line_minmax.npy', gamma)
        np.save(f'result/omslr/{i}/rho_line_minmax.npy', rho)
        # line gmse
        gamma, rho = tsseg.omslr.omslr_gmse(tsinfo, op_k, sigma_gmse)
        np.save(f'result/omslr/{i}/gamma_line_gmse.npy', gamma)
        np.save(f'result/omslr/{i}/rho_line_gmse.npy', rho)
        # dot minmax
        gamma, rho = tsseg_dot.omslr.omslr_minmax(tsinfo, op_k, sigma_minm)
        np.save(f'result/omslr/{i}/gamma_dot_minmax.npy', gamma)
        np.save(f'result/omslr/{i}/rho_dot_minmax.npy', rho)
        # dot gmse
        gamma, rho = tsseg_dot.omslr.omslr_gmse(tsinfo, op_k, sigma_gmse)
        np.save(f'result/omslr/{i}/gamma_dot_gmse.npy', gamma)
        np.save(f'result/omslr/{i}/rho_dot_gmse.npy', rho)
        fw.write('\n')

    fw.close()

def iter_sigma_mats(t):
    sigma_minm, beta_minm, alpha_minm = tsseg.omslr.iter_sigma(t, get_mean=True)
    sigma_gmse, beta_gmse, alpha_gmse = tsseg.omslr.iter_sigma(t, get_mean=False)
    return sigma_minm, sigma_gmse

def viz_kseg_result(exp_no, k):
    folder = 'thesis_line_minmax'
    dname = 'realdataset'
    dataset = load_dataset(folder, dname)
    dsnames = eval(open(f'result/{folder}/{dname}_name.txt').read())
    tsinfo = dataset[exp_no]

    get_pivot = tsseg_dot.omslr.get_pivots
    gamma_line_minmax = np.load(f'result/omslr/{exp_no}/gamma_line_minmax.npy')
    gamma_line_gmse = np.load(f'result/omslr/{exp_no}/gamma_line_gmse.npy')
    gamma_dot_minmax = np.load(f'result/omslr/{exp_no}/gamma_dot_minmax.npy')
    gamma_dot_gmse = np.load(f'result/omslr/{exp_no}/gamma_dot_gmse.npy')

    plt.figure(figsize=(24,12), dpi=300)
    opt_subplot(221, tsinfo, 'Seg by Line, Optimal MinMax', get_pivot(gamma_line_minmax[:k+1]), 0.5)
    opt_subplot(222, tsinfo, 'Seg by Line, Optimal GMSE',   get_pivot(gamma_line_gmse[:k+1]), 0.5)
    opt_subplot(223, tsinfo, 'Seg by Dot, Optimal MinMax',  get_pivot(gamma_dot_minmax[:k+1]), 1)
    opt_subplot(224, tsinfo, 'Seg by Dot, Optimal GMSE',    get_pivot(gamma_dot_gmse[:k+1]), 1)
    plt.tight_layout()
    plt.savefig(f'result/omslr/viz/png/{exp_no}_{k}.png', format='png')
    plt.savefig(f'result/omslr/viz/eps/{exp_no}_{k}.eps', forpat='ess')
    plt.cla()
    plt.clf()
    
def opt_subplot(position, tsinfo, title, pvts, delta):
    plt.subplot(position)
    plt.title(title)
    plt.plot(tsinfo, '.-')
    for pvt in pvts:
        plt.vlines(pvt-delta, min(tsinfo), max(tsinfo), 'r')

def viz_sample():
    viz_kseg_result(0, 10)
    viz_kseg_result(0, 12)
    viz_kseg_result(4, 6)
    viz_kseg_result(5, 8)
    viz_kseg_result(6, 6)
    viz_kseg_result(6, 9)
    viz_kseg_result(7, 20)
    viz_kseg_result(8, 11)
    viz_kseg_result(11, 12)

if __name__ == '__main__':
    # calc_omslr_compare()
    viz_sample()