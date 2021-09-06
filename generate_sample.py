import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rd

from tsseg.greed import *
from tsseg.utils import *
from tsseg.omslr import *

# np.random.seed(1)

def generate_random_segment(length, num_segs, lower=0, upper=1, voli=0.8, noise=0):
    '''
    generate_random_segment(length, num_segs, lower=0, upper=1, voli=0.8, noise=0)
    input:
        length [int]: lenght for output
        num_segs [int]: count of segments (need to less than length)
        lower, upper [numeric]: bound for normalize
        voli [float]: volilidity value, larger voli could avoid same trend's slope in continual (range: 0-1)
        noise [float]: 
    '''
    # pivots = np.array(list(range(5, length-5, length//num_segs//2)))
    pivots = np.arange(5, length-5, length//num_segs//2)
    np.random.shuffle(pivots)
    pivots = pivots[:num_segs-1]
    pivots.sort()
    print(pivots)
    pivots = [0] + pivots.tolist() + [length]
    # slopes = np.random.rand(num_segs) * voli + (1-voli)
    # slopes -= 0.5
    slopes = [rd.randrange(-5,5)]
    for i in range(num_segs-1):
        slp = slopes[-1]
        slp = rd.randrange(-5,5) - slp
        # slp += rd.randrange(5,10) - 10
        # while slp > 1:
        #     slp -= 1
        # while slp < -1:
        #     slp += 1
        slopes.append(slp)
        


    slp = slopes[0]
    pvt = pivots[0]
    npvt = pivots[1]
    alp = 0
    idx = 1

    print(pivots)
    print(slopes)
    ts_data = [0]
    for i in range(1, length):
        if i == npvt:
            alp = t
            slp = slopes[idx]
            pvt = pivots[idx]
            idx += 1
            npvt = pivots[idx]
        t = alp + slp*(i-pvt)
        ts_data.append(t)
    
    ts_data = np.array(ts_data)
    ts_data = (ts_data - ts_data.min()) / (ts_data.max() - ts_data.min())
    ts_data = ts_data * (upper - lower) + lower
    ts_data += np.random.rand(length) * 2 * noise - noise
    return ts_data.tolist()

def generate_random_segment_fordot(length, num_segs, lower=0, upper=1, voli=0.8, noise=0):
    '''
    generate_random_segment(length, num_segs, lower=0, upper=1, voli=0.8, noise=0)
    input:
        length [int]: lenght for output
        num_segs [int]: count of segments (need to less than length)
        lower, upper [numeric]: bound for normalize
        voli [float]: volilidity value, larger voli could avoid same trend's slope in continual (range: 0-1)
        noise [float]: 
    '''
    # pivots = np.array(list(range(5, length-5, length//num_segs//2)))
    pivots = np.arange(5, length-5, length//num_segs//2)
    np.random.shuffle(pivots)
    pivots = pivots[:num_segs-1]
    pivots.sort()
    print(pivots)
    pivots = [0] + pivots.tolist() + [length]
    slopes = np.random.rand(num_segs) * voli + (1-voli)

    slopes -= 0.5

    slp = slopes[0]
    pvt = pivots[0]
    npvt = pivots[1]
    alp = 0
    idx = 1

    print(pivots)
    print(slopes)
    ts_data = [0]
    for i in range(1, length):
        t = alp + slp*(i-pvt)
        ts_data.append(t)
        
        if i == npvt:
            alp = t
            slp = slopes[idx]
            pvt = pivots[idx]
            idx += 1
            npvt = pivots[idx]
        
    
    ts_data = np.array(ts_data)
    ts_data = (ts_data - ts_data.min()) / (ts_data.max() - ts_data.min())
    ts_data = ts_data * (upper - lower) + lower
    ts_data += np.random.rand(length) * 2 * noise - noise
    return ts_data.tolist()

def generate_simulation():
    fc = open('simu_without_noise.txt', 'w')
    fn = open('simu_with_noise.txt', 'w')
    noise = 5
    for i in range(10):
        # T = generate_random_segment_fordot(100, 6, 100, 1000, voli=0.9, noise=10)
        T = generate_random_segment(80, 5, 0, 100, voli=0.9, noise=0)
        fc.write(str([float(f'{t:4.2f}') for t in T]))
        fc.write('\n')
        Tn = T
        Tn += np.random.rand(len(T)) * 2 * noise - noise
        fn.write(str([float(f'{t:4.2f}') for t in Tn]))
        fn.write('\n')
    fc.close()
    fn.close()

if __name__ == '__main__':
    generate_simulation()