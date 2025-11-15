import numpy as np
from numba import jit, njit#, prange
from math import ceil

from config import *
from utils import get_moments

#генерация марковской цепи и наблюдений (по скачкам)
@njit(fastmath=True)
def choice(a, p):
    return a[np.searchsorted(np.cumsum(p), np.random.uniform())]

#генерация Y по theta
#равномерные
@njit(fastmath=True)
def get_y_uniform(state):
     return [np.random.uniform(y1_intervals[state][0], y1_intervals[state][1]),
             np.random.uniform(y2_intervals[state][0], y2_intervals[state][1])]

@njit(fastmath=True)
def get_y_triangular(state):
    return [np.random.triangular(y1_intervals[state][0], 
                                (y1_intervals[state][1] + y1_intervals[state][0]) / 2, 
                                 y1_intervals[state][1]),
            np.random.triangular(y2_intervals[state][0], 
                                (y2_intervals[state][1] + y2_intervals[state][0]) / 2, 
                                 y2_intervals[state][1])]
@njit(fastmath=True)
def get_y_triangular_2(state):
    return [np.random.triangular(y1_intervals[state][0], 
                                 y1_intervals[state][1], 
                                 y1_intervals[state][1]),
            np.random.triangular(y2_intervals[state][0], 
                                 y2_intervals[state][1], 
                                 y2_intervals[state][1])]

@njit(fastmath=True)
def get_y_3point(state):
    return [choice(points3_1[state], p_3point),
            choice(points3_2[state], p_3point)]

@njit(fastmath=True)
def arcsine_dist_standard():
    return (np.sin(np.random.uniform(0, 2*np.pi)) + 1)/2
@njit(fastmath=True)
def arcsine_dist(a, b):
    return arcsine_dist_standard() * (b-a) + a
@njit(fastmath=True)
def get_y_arcsine(state):
     return [arcsine_dist(y1_intervals[state][0], y1_intervals[state][1]),
             arcsine_dist(y2_intervals[state][0], y2_intervals[state][1])]


@jit(nopython=True, fastmath=True)
def sparse_mc(p0, Lambda, lam, T, get_y):
    #начальные условия и т.п.
    res_theta = []
    res_y = []
    res_t = []
    #O = np.empty(p0.shape[0])
    #J = np.zeros(Lambda.shape)
    k = np.arange(p0.shape[0])

    state = choice(k, p0)
    t = 0.
    Y = get_y(state)
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
            Y = get_y(state)
            res_theta.append(state)
            res_y.append(Y)
            #добавляю момент ИЗМЕНЕНИЯ состояния
            res_t.append(t)
        else:
            res_t.append(T)
            break
    return np.array(res_theta, dtype=np.int64), np.array(res_y), np.array(res_t)

#переводит кусочно-постоянное представление в дискретное
@njit(fastmath=True)
def to_discrete(jumps, time, T, h):
    res = np.empty((ceil(T/h),) + np.shape(jumps[0]), dtype=jumps.dtype)
    j = 0
    i = 0
    for s in time:
        for k in range(j, ceil(s/h)):
            res[k] = jumps[i]
        j = k + 1
        i += 1
    for k in range(j, ceil(T/h)):
        res[k] = jumps[-1]
    return res

@njit(nogil=True, cache=True)
def make_xi(g, dty, dtmc, ht, ht0, T, sigma):
    xi = np.empty((dty.shape[0], R))
    G = g(-1, dty.T, -1).T
    xi[0] = G[0]*ht
    for t in range(1, dty.shape[0]):
        xi[t] = xi[t-1] + G[t-1]*ht + sigma*ht0*np.random.normal(0, 1, G.shape[1])
    return xi

@njit(fastmath=True, nogil=True, cache=True)
def make_eta(h, dtmc, dty, ht, T):
    eta_jumps = []
    pr = ht*h(-1, dty, -1)
    a = np.array([True, False])
    for t in range(ceil(T/ht)):
        if choice(a, np.array([pr[t], 1-pr[t]])):
            eta_jumps.append(t*ht)
    return np.array(eta_jumps)

@njit(nogil=True, cache=True)
def make_discretized_xi(t_net_filtering, g, sigma, theta, Y, smjp_jumps):
    dxi = np.empty(t_net_filtering.shape[0]) #TODO dim
    G = g(-1, Y, -1) #TODO theta, t
    S = sigma(-1, Y, -1)**2 #TODO theta, t

    smjp_pos = 0

    dxi[0] = 0

    for i in range(1, t_net_filtering.shape[0]):
        mean, var, smjp_pos = get_moments(S, G, i, t_net_filtering, smjp_pos, smjp_jumps)
        dxi[i] = np.random.normal(mean, np.sqrt(var))

    return dxi

@njit(nogil=True, cache=True)
def make_discretized_eta(t_net_filtering, h, dtmc, dty, ht, T):
    eta = make_eta(h, dtmc, dty, ht, T)
    disc_eta = np.empty(t_net_filtering.shape[0])
    eta_pos = 0

    for i in range(t_net_filtering.shape[0]):
        eta_increment = 0

        while (eta_pos < eta.shape[0]) and (t_net_filtering[i] > eta[eta_pos]):
            eta_increment += 1
            eta_pos += 1
        
        disc_eta[i] = eta_increment

    return disc_eta
