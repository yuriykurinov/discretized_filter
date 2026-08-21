"""Генерация марковской цепи и наблюдений (по скачкам)."""
import numpy as np
from numba import jit, njit#, prange

from discretized_filter.utils.distributions import pareto_std
from discretized_filter.utils.grids import get_moments

@njit(fastmath=True)
def choice(a, p):
    return a[np.searchsorted(np.cumsum(p), np.random.uniform())]

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
    for t in range(min(G.shape[0], Y.shape[0])):
        G[t] = g(t, Y[t:t+1], theta[t])[0] #TODO theta, t
        S[t] = sigma(t, Y[t:t+1], theta[t])[0]**2 #TODO theta, t

    smjp_pos = 0

    for i in range(1, t_net_filtering.shape[0]):
        mean, var, smjp_pos = get_moments(G, S, i, t_net_filtering, smjp_pos, smjp_jumps)
        dxi[i] = mean + np.sqrt(var) * np.random.normal(0, 1, size=mean.shape[0])

    return dxi

@njit(nogil=True, cache=True)
def make_discretized_pareto(t_net, loc_fn, scale_fn, alpha_fn, theta, Y, smjp_jumps):
    """Наблюдение loc + scale*eps на шаге -- смесь по временам пребывания
    внутри шага (Парето не безгранично делима, накапливать параметры
    линейно, как для NORMAL/POISSON, нельзя)."""
    n = t_net.shape[0]
    L = np.zeros(n)
    Sc = np.zeros(n)
    Al = np.zeros(n)
    # Y/theta заданы по скачкам: их длина меньше числа узлов t_net
    for t in range(min(n, Y.shape[0])):
        L[t] = loc_fn(t, Y[t:t+1], theta[t])[0, 0]
        Sc[t] = scale_fn(t, Y[t:t+1], theta[t])[0, 0]
        Al[t] = alpha_fn(t, Y[t:t+1], theta[t])[0, 0]

    dx = np.zeros(n)
    pos = 0
    # продвижение pos -- те же условия while, что в get_moments, иначе
    # индексация компонент разойдётся с остальными каналами шага
    for i in range(1, n):
        ht = t_net[i] - t_net[i-1]
        u = np.random.random() * ht
        acc = 0.0
        prev_t = t_net[i-1]
        chosen = -1
        while (pos < smjp_jumps.shape[0]) and (t_net[i] > smjp_jumps[pos]):
            acc += smjp_jumps[pos] - prev_t
            if chosen < 0 and u < acc:
                chosen = pos
            prev_t = smjp_jumps[pos]
            pos += 1
        if chosen < 0:
            chosen = pos
        z = pareto_std(1, Al[chosen])[0]
        dx[i] = L[chosen] + Sc[chosen] * (1.0 + z / np.sqrt(12.0))
    return dx

def make_xi_generator(noise):
    """Аналог make_discretized_xi с приращением mean + sqrt(var) * noise(xi_dim)
    вместо mean + sqrt(var) * N(0,1); noise -- джитованная функция со средним 0
    и дисперсией 1 (см. utils.distributions.pareto_std)."""
    @njit(nogil=True)
    def xi_generator(t_net_filtering, g, sigma, theta, Y, smjp_jumps, xi_dim):
        dxi = np.zeros((t_net_filtering.shape[0], xi_dim))
        G = np.zeros((t_net_filtering.shape[0], xi_dim))
        S = np.zeros((t_net_filtering.shape[0], xi_dim))
        for t in range(min(G.shape[0], Y.shape[0])):
            G[t] = g(t, Y[t:t+1], theta[t])[0] #TODO theta, t
            S[t] = sigma(t, Y[t:t+1], theta[t])[0]**2 #TODO theta, t

        smjp_pos = 0

        for i in range(1, t_net_filtering.shape[0]):
            mean, var, smjp_pos = get_moments(G, S, i, t_net_filtering, smjp_pos, smjp_jumps)
            dxi[i] = mean + np.sqrt(var) * noise(mean.shape[0])

        return dxi

    return xi_generator

# @njit(nogil=True, cache=True)
def make_discretized_eta(t_net_filtering, h, theta, Y, smjp_jumps):
    deta = np.empty(t_net_filtering.shape[0])
    H = h(-1, Y, -1).squeeze()

    smjp_pos = 0

    deta[0] = 0

    for i in range(1, t_net_filtering.shape[0]):
        mean, var, smjp_pos = get_moments(H, H, i, t_net_filtering, smjp_pos, smjp_jumps)
        deta[i] = np.random.poisson(mean) 

    return deta
