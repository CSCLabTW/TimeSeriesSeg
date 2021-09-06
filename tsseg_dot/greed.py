import numpy as np
from itertools import combinations 

def naive(x, k, sigma, criterion='gmse'):
    if criterion == 'gmse':
        return naive_gmse(x, k, sigma)
    elif criterion == 'minmax':
        return naive_minmax(x, k, sigma)
    else:
        print('criterion not match, should be gmse|minmax')
        return None

def naive_gmse(x, k, sigma):
    pivot_candidate = list(range(2, len(x)-1))
    min_error = 10E10
    best_pivot = list(range(0, k+1))

    for c in combinations(pivot_candidate, k):
        pivots = [0] + list(c) + [len(x)]
        case_error = 0
        # if 1 in list(np.array(c[1:]) - np.array(c[:-1])): continue
        
        case_error = sum([sigma[pivots[i], pivots[i+1]-1] for i in range(len(pivots)-1)])
        # print(c, case_error)
        if case_error < min_error:
            min_error = case_error
            best_pivot = c
    return list(best_pivot)


def naive_minmax(x, k, sigma):
    pivot_candidate = list(range(2, len(x)-1))
    min_error = 10E10
    best_pivot = list(range(0, k+1))

    for c in combinations(pivot_candidate, k):
        pivots = [0] + list(c) + [len(x)]
        # case_error = 0
        
        case_error = max([sigma[pivots[i], pivots[i+1]-1] for i in range(len(pivots)-1)])
        # print(c, case_error)
        if case_error < min_error:
            min_error = case_error
            best_pivot = c
    return list(best_pivot)


def bottom_up_k(x, k, sigma):
    segments = [x[i:i+2] for i in range(0, len(x), 2)]
    cost=[]
    for i in range(len(segments)-1):
        cost.append(regression_error(segments[i] + segments[i+1]))
    if cost == []:
        return segments
    # print('cost:', cost)
    while len(segments) > k+1:
        i = cost.index(min(cost))
        segments[i] = segments[i] + segments[i+1]
        del segments[i+1]
        del cost[i]
        if cost == []:
            break
        if i > 0: cost[i-1] = regression_error(segments[i-1]+segments[i])
        if i < len(segments)-1: cost[i] = regression_error(segments[i]+segments[i+1])
    seg_lens = np.array([len(p) for p in segments])
    pivots = seg_lens.cumsum()
    return pivots.tolist()[:-1]

def bottom_up(T, max_err):
    segments = [T[i:i+2] for i in range(0, len(T), 2)]
    cost = []
    for i in range(len(segments)-1):
        cost.append(regression_error(segments[i] + segments[i+1]))
    # print(cost)
    if cost == []:
        return segments
    while min(cost) < max_err:
        i = cost.index(min(cost))
        segments[i] = segments[i] + segments[i+1]
        del segments[i+1]
        del cost[i]
        if cost == []:
            break
        if i > 0: cost[i-1] = regression_error(segments[i-1]+segments[i])
        if i < len(segments)-1: cost[i] = regression_error(segments[i]+segments[i+1])
    return segments

def bottom_up_iter(x):
    segments = [x[i:i+2] for i in range(0, len(x), 2)]
    cost=[]
    for i in range(len(segments)-1):
        cost.append(regression_error(segments[i] + segments[i+1]))
    if cost == []:
        return segments

    while len(segments) > 1:
        i = cost.index(min(cost))
        segments[i] = segments[i] + segments[i+1]
        del segments[i+1]
        del cost[i]
        if cost == []:
            break
        if i > 0: cost[i-1] = regression_error(segments[i-1]+segments[i])
        if i < len(segments)-1: cost[i] = regression_error(segments[i]+segments[i+1])
        seg_lens = np.array([len(p) for p in segments])
        pivots = seg_lens.cumsum()[:-1].tolist()
        error = max_error(x, pivots)
        yield error, pivots
        
    # return pivots.tolist()[:-1]

def top_down(T, max_err):
    # if len(T) < 4:
    #     return [T]
    if regression_error(T) > max_err:
        best_so_far = float('inf')
        for i in range(1, len(T)-1):
            err = large_mse(T[:i], T[i:])
            if  err < best_so_far:
                break_point = i
                best_so_far = err
        if 'i' not in locals():
            return [T]
        # return [T[:break_point], T[break_point:]]
        segments = []
        segments.extend(top_down(T[:break_point], max_err))
        segments.extend(top_down(T[break_point:], max_err))
        return segments
    else:
        return [T]

def create_tdtree(T, max_err, begin_point=0, depth=0):
    '''
    create_tdtree(T, max_err, begin_point=0, depth=0)
    create top down tree with dictionary-based tree structure
    '''
    if regression_error(T) > max_err:
        pivots = {}
        best_so_far = float('inf')
        for i in range(1, len(T)-1):
            err = large_mse(T[:i], T[i:])
            if  err < best_so_far:
                break_point = i
                best_so_far = err
        if 'i' not in locals():
            return [break_point]
        left  = create_tdtree(T[:break_point], max_err, 0, depth=depth+1)
        right = create_tdtree(T[break_point:], max_err, begin_point+break_point, depth=depth+1)
        
        lerr = regression_error(T[:break_point])
        rerr = regression_error(T[break_point:])
        pivots['pvt'] = begin_point+break_point
        pivots['err']  = regression_error(T)
        pivots['lerr'] = lerr
        pivots['rerr'] = rerr
        pivots['depth'] = depth
        if left != {} : pivots['left']  = left
        if right != {}: pivots['right'] = right
        return pivots
    else:
        return {}


def traverse(node, flatten_nodes):
    print(node)
    depth =  node['depth']
    if len(flatten_nodes) < depth + 1:
        flatten_nodes.append([])

    nd = { k: node[k] for k in ['pvt', 'lerr', 'rerr', 'err', 'depth'] }
    flatten_nodes[depth].append(nd)
    if 'left' in node:
        traverse(node['left'], flatten_nodes)
    if 'right' in node:
        traverse(node['right'], flatten_nodes)
    return flatten_nodes

def mse(y, y_bar):
    if len(y) < 2:
        return 0.0
    s = 0.0
    for i in range(len(y)):
        s += (y[i] - y_bar[i])**2
    # err = (s**.5)/len(y)
    err = s/len(y)
    return err

def regression(y):
    if len(y) == 0: return []
    if len(y) == 1: return y
    x = list(range(1,len(y)+1))
    fit = np.polyfit(x, y, 1)
    regr = np.poly1d(fit)
    r = regr(x)
    return r

def regression_error(T):
    r = regression(T)
    err = mse(T, r)
    return err

def max_error(t, pivots):
    pivots = [0] + pivots + [len(t)]
    return max([regression_error(t[pivots[i]:pivots[i+1]+1]) for i in range(len(pivots)-1)])

def large_mse(seg1, seg2):
    return max(regression_error(seg1), regression_error(seg2))
