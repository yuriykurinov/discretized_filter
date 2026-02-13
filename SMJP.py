import numpy as np
from numba import jit, njit#, prange

from config import *
from utils import get_moments

#генерация марковской цепи и наблюдений (по скачкам)
@njit(fastmath=True)
def choice(a, p):
    return a[np.searchsorted(np.cumsum(p), np.random.uniform())]

#генерация Y по theta
#равномерные
@njit(fastmath=True)
def get_y_uniform(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.uniform(y_interval[state][0], y_interval[state][1])
        )
    return res

@njit(fastmath=True)
def get_y_triangular(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.triangular(y_interval[state][0], 
                                (y_interval[state][1] + y_interval[state][0]) / 2, 
                                 y_interval[state][1])
        )
    return res

@njit(fastmath=True)
def get_y_triangular_2(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.triangular(y_interval[state][0], 
                                 y_interval[state][1], 
                                 y_interval[state][1])
        )
    return res

###########
# @njit(fastmath=True)
# def get_y_3point(state):
#     return [choice(points3_1[state], p_3point),
#             choice(points3_2[state], p_3point)]

# @njit(fastmath=True)
# def arcsine_dist_standard():
#     return (np.sin(np.random.uniform(0, 2*np.pi)) + 1)/2
# @njit(fastmath=True)
# def arcsine_dist(a, b):
#     return arcsine_dist_standard() * (b-a) + a
# @njit(fastmath=True)
# def get_y_arcsine(state):
#      return [arcsine_dist(y1_intervals[state][0], y1_intervals[state][1]),
#              arcsine_dist(y2_intervals[state][0], y2_intervals[state][1])]
###########

@jit(nopython=True, fastmath=True)
def sparse_mc(p0, Lambda, lam, T, get_y, y_intervals):
    #начальные условия и т.п.
    res_theta = []
    res_y = []
    res_t = []
    #O = np.empty(p0.shape[0])
    #J = np.zeros(Lambda.shape)
    k = np.arange(p0.shape[0])

    state = choice(k, p0)
    t = 0.
    Y = get_y(state, y_intervals)
    res_theta.append(state)
    res_y.append(Y)

    pr = [-Lambda[state][k != state] / lam[state] for state in k]
    state_exc = [k[k != state] for state in k]

    while True:
        #генерирую скачок
        tmp = np.random.exponential(-1/lam[state])
        t += tmp
        if t <= T:
            pr_state = state
            state = choice(state_exc[pr_state], pr[pr_state])
            Y = get_y(state, y_intervals)
            res_theta.append(state)
            res_y.append(Y)
            #добавляю момент ИЗМЕНЕНИЯ состояния
            res_t.append(t)
        else:
            res_t.append(T)
            break
    return np.array(res_theta, dtype=np.int64), np.array(res_y), np.array(res_t)

@njit(nogil=True, cache=True)
def make_discretized_xi(t_net_filtering, g, sigma, theta, Y, smjp_jumps, xi_dim):
    dxi = np.zeros((t_net_filtering.shape[0], xi_dim))
    G = np.zeros((t_net_filtering.shape[0], xi_dim))
    S = np.zeros((t_net_filtering.shape[0], xi_dim))
    for t in range(G.shape[0]):
        G[t] = g(t, Y[t:t+1], theta[t])[0] #TODO theta, t
        S[t] = sigma(t, Y[t:t+1], theta[t])[0]**2 #TODO theta, t

    smjp_pos = 0

    for i in range(1, t_net_filtering.shape[0]):
        mean, var, smjp_pos = get_moments(G, S, i, t_net_filtering, smjp_pos, smjp_jumps)
        dxi[i] = mean + np.sqrt(var) * np.random.normal(0, 1, size=mean.shape[0])

    return dxi

# @njit(nogil=True, cache=True) 
def make_discretized_eta(t_net_filtering, h, theta, Y, smjp_jumps):
    deta = np.empty(t_net_filtering.shape[0])
    H = h(-1, Y, -1)

    smjp_pos = 0

    deta[0] = 0

    for i in range(1, t_net_filtering.shape[0]):
        mean, var, smjp_pos = get_moments(H, H, i, t_net_filtering, smjp_pos, smjp_jumps)
        deta[i] = np.random.poisson(mean) 

    return deta
