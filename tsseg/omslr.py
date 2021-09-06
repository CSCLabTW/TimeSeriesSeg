import numpy as np

def iter_sigma(x, get_mean=True, get_alpha_beta=True):
    '''
    iter_sigma(x, [get_alpha_beta=False])
    iteractive sigma caculation, given a time-series data vector
    return the 2D matrix with sub-vector's sigma (mean-square-error)

    input:
        x [list]: Time-series data
        get_alpha_beta [bool default False]: return more detail matrix or not
    return:
        sigma: 2D matrix for error

    '''
    def x_bar(i,j): return sum(x[i:j+1])/len(x[i:j+1])
    def t_bar(i,j): return (i+j)/2
    N = len(x)
    q_mat, p_mat, r_mat = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    sigma, a_mat, b_mat = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    xb_mat, xi_mat = np.zeros((N, N)), np.zeros((N, N))
    delta_mat, zeta_mat = np.zeros((N, N)), np.zeros((N, N))

    for i in range(N):
        xb_mat[i,i]    = x[i]
        a_mat[i,i]     = x[i]
        xi_mat[i,i]    = i**2
        delta_mat[i,i] = x[i]*i
        zeta_mat[i,i]  = x[i]**2
        
    for i in range(N-1):
        for j in range(i+1, N):
            xb_mat[i,j]= ((xb_mat[i,j-1] * (j-i)) + x[j]) / (j-i+1)
            xi_mat[i,j]    = xi_mat[i,j-1]  + j**2
            delta_mat[i,j] = delta_mat[i,j-1]   + x[j]*j
            zeta_mat[i,j]  = zeta_mat[i,j-1] + x[j]**2
            p_mat[i,j] = p_mat[i,j-1] \
                        + (j-i)*(xb_mat[i,j-1]-xb_mat[i,j])*(t_bar(i,j-1)-t_bar(i,j)) \
                        + (x[j]-xb_mat[i,j])*(j-t_bar(i,j))
            q_mat[i,j] = q_mat[i,j-1] \
                        + (j-t_bar(i,j-1))**2 + 2*(t_bar(i,j-1)-t_bar(i,j))*(j-t_bar(i,j-1)) \
                        + (j-i+1)*(t_bar(i,j-1)-t_bar(i,j))**2
            b_mat[i,j] = p_mat[i,j]/q_mat[i,j]
            
            a_mat[i,j] = xb_mat[i,j]-b_mat[i,j]*t_bar(i,j)
            r_mat[i,j] = zeta_mat[i,j] \
                        + (b_mat[i,j]**2)*(xi_mat[i,j]) \
                        + 2*a_mat[i,j]*b_mat[i,j]*((j+i)*(j-i+1)/2) \
                        + (j+1-i)*(a_mat[i,j]**2) \
                        - 2*b_mat[i,j]*delta_mat[i,j] \
                        - 2*a_mat[i,j]*(j-i+1)*xb_mat[i,j]
            if get_mean:
                sigma[i,j] = r_mat[i,j] /(j-i+1)
            else:
                sigma[i,j] = r_mat[i,j]
    # sigma = np.abs(np.round(sigma, 4))
    sigma = sigma.clip(min=0)
    if get_alpha_beta:
        return sigma, b_mat, a_mat
    else:
        # return xb_mat
        return sigma

def omslr_gmse(x, max_k, sigma):
    i = 0
    rho = []
    gamma = []
    r_arr = [0] * len(x)
    g_arr = [0] * len(x)
    for j in range(0, len(x)):
        r_arr[j] = sigma[0][j]
        g_arr[j] = j+1
    rho.append(r_arr)
    i += 1
    gamma.append(g_arr)
    while i < max_k+1:
        r_arr = [float('inf')] * len(x)
        g_arr = [0] * len(x)
        for j in range(0, len(x)):
            rs = [float('inf')] * len(x)
            for m in range(i-1, j):
                rs[m] = rho[i-1][m]+sigma[m+1][j]
            g = 0
            min_rs = min(rs)
            g = rs.index(min_rs)
            g_arr[j] = g + 1 
            r = rho[i-1][g] + sigma[g+1][j]
            r_arr[j] = min_rs
        gamma.append(g_arr)
        rho.append(r_arr)
        i += 1
    return gamma, rho


def omslr_minmax(x, max_k, sigma):
    i = 0
    rho = []
    gamma = []
    r_arr = [0] * len(x)
    g_arr = [0] * len(x)
    for j in range(0, len(x)):
        r_arr[j] = sigma[0][j]
        g_arr[j] = j+1
    rho.append(r_arr)
    gamma.append(g_arr)
    # i = 0
    i += 1
    while i < max_k+1:
        r_arr = [float('inf')] * len(x)
        g_arr = [0] * len(x)
        for j in range(0, len(x)):
            rs = [float('inf')] * len(x)
            # print(i,j)
            for m in range(i-1, j):
                # print(m, sigma[i][m-1], sigma[m][j])
                rs[m] = max(rho[i-1][m], sigma[m+1][j])
            g = 0
            min_rs = min(rs)
            g = rs.index(min_rs)
            g_arr[j] = g+1
            r_arr[j] = min_rs
        gamma.append(g_arr)
        rho.append(r_arr)
        i += 1
    return gamma, rho

def get_pivots(gamma, p=-1):
    p = gamma[p][-1]
    pivots = [p]
    for i in reversed(range(1, len(gamma)-1)):
        p = gamma[i][p-1]
        pivots += [p]
    pivots.sort()
    return pivots