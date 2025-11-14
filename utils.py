import numpy as np
from numba import njit

@njit()
def set_seed(seed):
    np.random.seed(seed)


#декартово произведение
def cartesian_product(arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


#плотности
@njit(fastmath=True, nogil=True, cache=True)
def norm(x, mu, sigma_sq):
    return np.exp(-(x - mu)**2 / (2*sigma_sq)) / np.sqrt(2*np.pi*sigma_sq)
@njit(fastmath=True, nogil=True, cache=True)
def unif(x, a, b):
    return (x <= b) * (x >= a) / (b - a)
@njit(fastmath=True, nogil=True, cache=True)
def unif(x, a, b):
    return (x <= b) * (x >= a) / (b - a)

def triang1(x, a, b, delta):
    res = 2 * (x >= a) * (x <= b) * (x - a) / (b - a)**2
    return res / (res.sum()*delta)

def triang2(x, a, b, delta):
    res = 2 * (x >= a) * (x <= b) * (b - x) / (b - a)**2
    return res / (res.sum()*delta)

#@np.vectorize()
def triang(x, a, b, delta):
    c = (a + b) / 2
    res = (x <= b) * (x >= a) * np.abs(b - c - np.abs(c - x)) / (b - c)**2
    return res / (res.sum() * delta)

@njit(fastmath=True)
def arcsine_dist_pdf(net, a, b, delta):
    res = np.zeros(net.shape)
    idx_special = []
    for i in range(net.shape[0]):
        s1 = net[i] - a
        s2 = b - net[i]
        if (s1 > 0) and (s2 > 0):
            res[i] = np.exp(- 0.5 * (np.log(s1) + np.log(s2)))
        elif (s1 < 0) or (s2 < 0):
            continue
        else:
            idx_special.append(i)
    if len(idx_special) > 0:
        for i in idx_special:
            res[i] = np.max(res)

    return res / (res.sum() * delta)


def get_distributions(N, M_net, net1, net2, y1_intervals, y2_intervals, delta1, delta2):
    #условная плотность y при условии theta
    pi_uniform = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp1 = unif(net1, y1_intervals[n][0], y1_intervals[n][1])
        tmp2 = unif(net2, y2_intervals[n][0], y2_intervals[n][1])
        pi_uniform[n] = cartesian_product([tmp1, tmp2]).prod(axis=1)

    pi_3point = np.zeros((N, M_net.shape[0]))
    for n in range(N):
        tmp1 = np.zeros(net1.shape[0])
        tmp2 = np.zeros(net2.shape[0])
        tmp1[np.argmin(np.abs(net1 - y1_intervals.T[0][n]))] = 1
        tmp1[np.argmin(np.abs(net1 - y1_intervals.T[1][n]))] = 1
        tmp1[
            np.argmin(np.abs(net1 - (y1_intervals.T[0][n] + y1_intervals.T[1][n])/2))
        ] = 1
        tmp1 /= 3*delta1
        tmp2[np.argmin(np.abs(net2 - y2_intervals.T[0][n]))] = 1
        tmp2[np.argmin(np.abs(net2 - y2_intervals.T[1][n]))] = 1
        tmp2[np.argmin(np.abs(net2 - (y2_intervals.T[0][n] + y2_intervals.T[1][n])/2))] = 1
        tmp2 /= 3*delta2
        pi_3point[n] = cartesian_product([tmp1, tmp2]).prod(axis=1)

    pi_triangular = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp1 = triang(net1, y1_intervals[n][0], y1_intervals[n][1], delta1)
        tmp2 = triang(net2, y2_intervals[n][0], y2_intervals[n][1], delta2)
        pi_triangular[n] = cartesian_product([tmp1, tmp2]).prod(axis=1)

    pi_triangular_2 = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp1 = triang1(net1, y1_intervals[n][0], y1_intervals[n][1], delta1)
        tmp2 = triang1(net2, y2_intervals[n][0], y2_intervals[n][1], delta2)
        pi_triangular_2[n] = cartesian_product([tmp1, tmp2]).prod(axis=1)

    pi_arcsine = np.zeros((N, M_net.shape[0]))
    for n in range(N):
        tmp1 = arcsine_dist_pdf(net1, y1_intervals[n][0], y1_intervals[n][1], delta1)
        tmp2 = arcsine_dist_pdf(net2, y2_intervals[n][0], y2_intervals[n][1], delta2)
        pi_arcsine[n] = cartesian_product([tmp1, tmp2]).prod(axis=1)

    #средние y
    y_means_uniform = np.array(
        [
            [(y1_intervals[n][1] + y1_intervals[n][0])/2 for n in range(N)],
            [(y2_intervals[n][1] + y2_intervals[n][0])/2 for n in range(N)]
        ]
    )

    y_means_arcsine = y_means_uniform.copy()

    y_means_triang = y_means_uniform.copy()

    y_means_3point = y_means_uniform.copy()

    y_means_2point = y_means_uniform.copy()

    return pi_uniform, pi_3point, pi_triangular_2, pi_arcsine, y_means_uniform,\
        y_means_arcsine, y_means_triang, y_means_3point, y_means_2point

@njit()
def get_index(eta, t_net, t_net0):
    index = []
    for t in range(t_net.shape[0]):
        eta_flag = t_net[t] in eta[:,0]
        index.append(eta_flag and not (t_net[t] in t_net0))
    return np.array(index)


@njit(fastmath=True, nogil=True)
def get_moments(S, G, i, t_net, smjp_pos, smjp_jumps):
    mean = 0
    var = 0
    prev_t = t_net[i-1]
    # собираю все скачки на (t_net[i-1], t_net[i]]
    while (smjp_pos < smjp_jumps.shape[0]) and (t_net[i] > smjp_jumps[smjp_pos]):
        #тут smjp_jumps[-1] == T, поэтому t_net[-1] всегда <= smjp_jumps[-1]
        #TODO сделать нормально
        dt = (smjp_jumps[smjp_pos] - prev_t)

        mean += G[smjp_pos] * dt
        var += S[smjp_pos] * dt
        prev_t = smjp_jumps[smjp_pos]
        smjp_pos += 1

    dt = (t_net[i] - prev_t)
    mean += G[smjp_pos] * dt
    var += S[smjp_pos] * dt
    
    return mean, var, smjp_pos
