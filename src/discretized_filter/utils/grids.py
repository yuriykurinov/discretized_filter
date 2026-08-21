import numpy as np
import numba as nb
from numba import njit


@njit()
def set_seed(seed):
    np.random.seed(seed)


def cartesian_product(arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


@njit()
def get_index(eta, t_net, t_net0):
    index = []
    for t in range(t_net.shape[0]):
        eta_flag = t_net[t] in eta[:,0]
        index.append(eta_flag and not (t_net[t] in t_net0))
    return np.array(index)

def to_discrete(jumps, time, T, h):
    from math import ceil
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


@njit(
# TODO почему-то не работает, когда явно указаны типы
    # nb.types.Tuple(
    #     (nb.float64[:], nb.float64[:], nb.uintp)
    # )(
    #     nb.float64[:, :, :],
    #     nb.float64[:, :, :],
    #     nb.uintp,
    #     nb.float64[:],
    #     nb.uintp,
    #     nb.float64[:]
    # ),
    fastmath=True,
    nogil=True
)
def get_moments(G, S, i, t_net, smjp_pos, smjp_jumps):
    mean = np.zeros(G.shape[1:])
    var = np.zeros(S.shape[1:])
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
