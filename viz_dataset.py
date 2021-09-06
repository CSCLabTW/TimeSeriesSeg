import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makedir(path):
    if not os.path.isdir(path): os.makedirs(path)

def load_dataset(path):
    return [eval(l) for l in list(filter(None, open(path).read().split('\n')))]

def savefig(path):
    plt.savefig(f'{path}.png', format='png')
    plt.savefig(f'{path}.eps', format='eps')

def ucr():
    makedir('dataset/ucr/')
    ds = load_dataset('ucr.txt')

    dn = ['ECG','EOG','EPG','Hemodynamics','HRM','Image','Motion','Power','Sensor','Simulated','Spectro','Spectrum']
    plt.figure(figsize=(16,12), dpi=300)
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.title(dn[i])
        plt.plot(ds[i], '-')

    plt.tight_layout()
    savefig('dataset/ucr_data')
    
    for i in range(12):
        plt.figure(figsize=(4, 3))
        plt.plot(ds[i])
        plt.tight_layout()
        savefig(f'dataset/ucr/{i}')
        plt.close()


def simu():
    makedir('dataset/simup/')
    makedir('dataset/simun/')
    simup = load_dataset('simup.txt')
    simun = load_dataset('simun.txt')
    plt.figure(figsize=(8, 15))
    for i in range(10):
        plt.subplot(10, 2, i*2+1)
        plt.plot(simup[i])
        plt.subplot(10, 2, i*2+2)
        plt.plot(simun[i])
    plt.tight_layout()
    savefig('dataset/simulation_data')
    plt.close()
    for i in range(10):
        plt.figure(figsize=(4, 3))
        plt.plot(simup[i])
        plt.tight_layout()
        savefig(f'dataset/simup/{i}')
        plt.close()
        plt.figure(figsize=(4, 3))
        plt.plot(simun[i])
        plt.tight_layout()
        savefig(f'dataset/simun/{i}')
        plt.close()
def simul():
    makedir('dataset/simul/')

    simul = load_dataset('simul.txt')[0]
    plt.figure(figsize=(4, 3))
    plt.plot(simul)
    plt.tight_layout()
    savefig(f'dataset/simul/simul')
    plt.close()
        


# simu()
simul()
# ucr()