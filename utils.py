import numpy as np
from numba import njit
import pickle

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


def get_distributions(N, M_net, nets, y_intervals, deltas):
    #условная плотность y при условии theta
    pi_uniform = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp = []
        for y_intervals_ith_dim, net in zip(y_intervals, nets):
            tmp.append(
                unif(net, y_intervals_ith_dim[n][0], y_intervals_ith_dim[n][1])
            )
        pi_uniform[n] = cartesian_product(tmp).prod(axis=1)

    pi_3point = np.zeros((N, M_net.shape[0]))
    for n in range(N):
        tmp = []
        for y_intervals_ith_dim, net, delta in zip(y_intervals, nets, deltas):
            tmp.append(
                np.zeros(net.shape[0])
            )
            tmp[-1][np.argmin(np.abs(net - y_intervals_ith_dim[n][0]))] = 1
            tmp[-1][np.argmin(np.abs(net - y_intervals_ith_dim[n][1]))] = 1
            tmp[-1][
                np.argmin(
                    np.abs(
                        net - (y_intervals_ith_dim[n][0] + y_intervals_ith_dim[n][1])/2
                    )
                )
            ] = 1
            tmp[-1] /= 3*delta

        pi_3point[n] = cartesian_product(tmp).prod(axis=1)

    pi_triangular = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp = []
        for y_intervals_ith_dim, delta, net in zip(y_intervals, deltas, nets):
            tmp.append(
                triang(net, y_intervals_ith_dim[n][0], y_intervals_ith_dim[n][1], delta)
            )
        pi_triangular[n] = cartesian_product(tmp).prod(axis=1)

    pi_triangular_2 = np.empty((N, M_net.shape[0]))
    for n in range(N):
        tmp = []
        for y_intervals_ith_dim, delta, net in zip(y_intervals, deltas, nets):
            tmp.append(
                triang1(net, y_intervals_ith_dim[n][0], y_intervals_ith_dim[n][1], delta)
            )

        pi_triangular_2[n] = cartesian_product(tmp).prod(axis=1)

    pi_arcsine = np.zeros((N, M_net.shape[0]))
    for n in range(N):
        tmp = []
        for y_intervals_ith_dim, delta, net in zip(y_intervals, deltas, nets):
            tmp.append(
                arcsine_dist_pdf(net, y_intervals_ith_dim[n][0], y_intervals_ith_dim[n][1], delta)
            )

        pi_arcsine[n] = cartesian_product(tmp).prod(axis=1)

    #средние y
    y_means_uniform = np.array(
        [
            [(y_intervals_ith_dim[n][1] + y_intervals_ith_dim[n][0])/2 for n in range(N)]\
            for y_intervals_ith_dim in y_intervals
        ]
    )

    y_means_arcsine = y_means_uniform.copy()

    y_means_triang = y_means_uniform.copy()

    y_means_3point = y_means_uniform.copy()

    return pi_uniform, pi_3point, pi_triangular_2, pi_arcsine,\
        y_means_uniform, y_means_arcsine, y_means_triang, y_means_3point

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
        dt = (smjp_jumps[smjp_pos] - prev_t)

        mean += G[smjp_pos] * dt
        var += S[smjp_pos] * dt
        prev_t = smjp_jumps[smjp_pos]
        smjp_pos += 1

    dt = (t_net[i] - prev_t)
    mean += G[smjp_pos] * dt
    var += S[smjp_pos] * dt
    
    return mean, var, smjp_pos


def load_saved_path(exp_id):
    with open(f'saved_path/t_{exp_id}.pkl', 'rb') as f:
        t = pickle.load(f)
    with open(f'saved_path/theta_{exp_id}.pkl', 'rb') as f:
        theta = pickle.load(f)
    with open(f'saved_path/y_{exp_id}.pkl', 'rb') as f:
        y = pickle.load(f)
    with open(f'saved_path/theta_est_{exp_id}.pkl', 'rb') as f:
        theta_est = pickle.load(f)
    with open(f'saved_path/y_est_{exp_id}.pkl', 'rb') as f:
        y_est = pickle.load(f) 
    with open(f'saved_path/dxi_{exp_id}.pkl', 'rb') as f:
        dxi = pickle.load(f)
    with open(f'saved_path/deta_{exp_id}.pkl', 'rb') as f:
        deta = pickle.load(f) 
    return theta, y, t, theta_est, y_est, dxi, deta
    
