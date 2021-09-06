"""
Real Data Extractor
extract UCR time series dataset from 15 dataset
each dataset would be extracted one record to real_data_set.txt

Note:
    drop Trajectory category cause the dataset contain lots of '?' data
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import OrderedDict

from tsseg.utils import *
from tsseg.greed import *
from tsseg.omslr import *


dnames = ['Device:SmallKitchenAppliances','ECG:ECGFiveDays','EOG:EOGVerticalSignal','EPG:InsectEPGRegularTrain','Hemodynamics:PigCVP','HRM:Fungi','Image:MiddlePhalanxOutlineAgeGroup','Motion:UWaveGestureLibraryAll','Power:PowerCons','Sensor:Plane','Simulated:CBF','Spectro:Ham','Spectrum:Rock','Traffic:Chinatown']
def extract():
    dataset_list = OrderedDict({
        'Device': 'SmallKitchenAppliances',
        'ECG': 'ECGFiveDays',
        'EOG': 'EOGVerticalSignal',
        'EPG': 'InsectEPGRegularTrain',
        'Hemodynamics': 'PigCVP',
        'HRM': 'Fungi',
        'Image': 'MiddlePhalanxOutlineAgeGroup',
        'Motion': 'UWaveGestureLibraryAll',
        'Power': 'PowerCons',
        'Sensor': 'Plane',
        'Simulated': 'CBF',
        'Spectro': 'Ham',
        'Spectrum': 'Rock',
        'Traffic': 'Chinatown'
    })

    fw = open('realdataset.txt', 'w')
    for k in dataset_list:
        dname = dataset_list[k]
        df = pd.read_csv('dataset/{0}/{0}_TRAIN.csv'.format(dname))
        df = df.drop('target', axis=1)
        # print(df.iloc[0].values.tolist())
        fw.write('{}\n'.format(df.iloc[0].values.tolist()))
    fw.close()

def omslr_testing():
    dataset = [eval(line) for line in list(filter(None, open('realdataset.txt').read().split('\n')))]
    t = dataset[2]

    for i, t in enumerate(dataset):
        sigma, beta, alpha = iter_sigma(t)
        gamma, rho = omslr_minmax(t, 3, sigma)
        pivots = get_pivots(gamma)
        err = rho[-1][-1]
        pvts_td = seg2pivots(top_down(t, err))
        pvts_bu = seg2pivots(bottom_up(t, err))
        with open('k_seg tesint.txt', 'a') as fa:
            fa.write('{}:\n'.format(dnames[i]))
            fa.write('error={}\n'.format(err))
            fa.write('OMSLR MinMax\n')
            fa.write('{}\n'.format(pivots))
            fa.write('Top-Down')
            fa.write('{}\n'.format(pvts_td))
            fa.write('Bottom-Up')
            fa.write('{}\n\n'.format(pvts_bu))

def plot_dataset():
    dataset = [eval(line) for line in list(filter(None, open('realdataset.txt').read().split('\n')))]
    L = len(dataset)
    plt.figure(figsize=(16,90), dpi=150)
    for i, ds in enumerate(dataset):
        plt.subplot(L, 1, i+1)
        plt.plot(ds, '.-', label=dnames[i])
        plt.title(dnames[i])
    plt.tight_layout()
    plt.savefig('all_dataset')

extract()
plot_dataset()
# omslr_testing()
