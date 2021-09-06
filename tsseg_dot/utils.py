
def seg2pivots(segments):
    pivots = []
    pivot = 0
    for seg in segments[:-1]:
        pivot += len(seg)
        pivots.append(pivot)
    return pivots

def get_segments(x, pivots, sigma, beta):
    intervals =[]
    pivots = [0]+pivots
    for i in range(len(pivots)-1):
         intervals.append(pivots[i:i+2])
    intervals.append([pivots[-1], len(x)])
    segments, durations, slopes, errors = [], [], [], []
    for i in intervals:
        segments.append(x[i[0]:i[1]])
        durations.append(i[1]-i[0])
        slopes.append(beta[i[0],i[1]-1])
        errors.append(sigma[i[0],i[1]-1])
    return segments, durations, slopes, errors

def get_phi(x, pivots, beta, alpha):
    pvt = [0] + pivots + [len(x)]
    est = []
    for i in range(len(pvt)-1):
        seg = (pvt[i], pvt[i+1]-1)
        b, a = beta[seg], alpha[seg]
        est.extend(list(map(lambda x: b*x+a ,list(range(pvt[i], pvt[i+1])))))
    residuals = np.array(x)-np.array(est)
    return residuals.sum()