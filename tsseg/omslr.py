import numpy as np

def iter_sigma(x:list, get_mean:bool=True, get_alpha_beta=True):
    '''
    iter_sigma(x, [get_alpha_beta=False])
    iteractive sigma caculation, given a time-series data vector
    return the 2D matrix with sub-vector's sigma (mean-square-error)

    input:
        x [list]: Time-series data
        get_mean [bool]: when it's True, sigman would be mean-square-error of sub-time-series data. 
                        If False, sigma is sum-square-error
        get_alpha_beta [bool default False]: return more detail matrix or not
    return:
        sigma: 2D matrix for error

    example:
        sigma = iter_sigma([8,1,5,4,3,6,9], True, False) # get mse matrix simply
        ### or ###
        sigma, beta, alpha = iter_sigma([8,1,5,4,3,6,9], True, True) # for trace more information
       
    Tip:
        get_mean usage:
        when we set get_mean as False, the matrix presents sum-square-error
        which means the error of a longer segment would be larger in usually
        
        once we set get_mean sa True, the matrix presents mean-square-error normally
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

    sigma = sigma.clip(min=0)
    if get_alpha_beta:
        return sigma, b_mat, a_mat
    else:
        return sigma

def omslr_gmse(x:list, max_k:int, sigma:numpy):
    '''
    omslr_gmse(x:list, max_k:int, sigma:numpy.array)
    Optimal Multi-Segmentation Linear Regression with global mean-square-error policy:
    optimized with minimizing the sum
    input:
        x[list] - time series data, should be and array of numeric data
        max_k[int]: number of segmentation, notice that all the k less than max_k are finished after this procedure
        sigma[numpy.array]: sigma matrix, calculated from iter_sigma
    return 
        gamma[list<2D>]: segmentation pivot matrix
        rho[list<2D>]: segmentation error matrix
        
    example:
        tsdata = [8,1,5,4,3,6,9]
        sigma = iter_sigma(tsdata, get_mean=False, get_alpha_beta=False) # get mse matrix simply
        gamma, rho = omslr_gmse(tsdata, 3, sigma)
        ## get_mean=False cause it's "global", that means sumation contain call the residuals 
        ## (can devided by len(tsdata) but not necessary)
    '''
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


def omslr_minmax(x:list, max_k:int, sigma:numpy.array):
    '''
    omslr_minmax(x:list, max_k:int, sigma:numpy.array)
    Optimal Multi-Segmentation Linear Regression with min-max policy:
    optimized with minimizing the max-error in segments
    input:
        x[list] - time series data, should be and array of numeric data
        max_k[int]: number of segmentation, notice that all the k less than max_k are finished after this procedure
        sigma[numpy.array]: sigma matrix, calculated from iter_sigma
    return 
        gamma[list<2D>]: segmentation pivot matrix
        rho[list<2D>]: segmentation error matrix
    example:
        tsdata = [8,1,5,4,3,6,9]
        sigma = iter_sigma(tsdata, get_mean_True, get_alpha_beta=False) # get mse matrix simply
        gamma, rho = omslr_gmse(tsdata, 3, sigma)
    '''
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
    '''
    get_pivots(gamma)
    gamma[numpy.array(2D)]: generate from omslr_gmse function
    p[int]: get less segmentation pivots (p should less than number of rows in gamma)
    return[list(1D)]:
        segmentation pivots
    '''
    p = gamma[p][-1]
    pivots = [p]
    for i in reversed(range(1, len(gamma)-1)):
        p = gamma[i][p-1]
        pivots += [p]
    pivots.sort()
    return pivots
