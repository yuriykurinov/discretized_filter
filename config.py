import numpy as np
import numba as nb
from numba import njit
from math import ceil
from utils import set_seed, cartesian_product, get_distributions
from SMJP import make_discretized_xi, make_discretized_eta

exp_id = 'example_for_arc_new'

two_jumps = False

n_points = 1 # time integration

#правая граница временного промежутка
T = 5

ht = 0.5 #шаг фильтрации


seed = 321

N = 4

np.random.seed(seed)
rng = np.random.default_rng(seed=seed)
set_seed(seed)

@njit(fastmath=True, nogil=True, cache=True)
def h(t, y, theta):
    return 1000 * (y[:, 1:2] / y[:, 0:1])

@njit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.int64
    ),
    nogil=True, 
    cache=True
)
def g(t, y, theta):
    return y[:, 0:1]

sigma_obs = 0.1#0.0005
@njit(nogil=True, cache=True)
def sigma(t, y, theta, b=sigma_obs):
    return b*np.sqrt(y[:, 0:1])

Lambda = np.array(
    [
        [   0, 0.45,   0, 0.05],
        [0.25,    0, 0.1, 0.05],
        [0.75,    0,   0, 0.05],
        [   1,    0,   0,    0]
    ]
)

for i in range(Lambda.shape[0]):
    Lambda[i][i] = -np.sum(Lambda[i])

#правая граница временного промежутка
T = 10

ht = 1e-1 #шаг фильтрации
#сетка
t_net_filtering = np.array([t*ht for t in range(ceil(T/ht))])

_, _, vh = np.linalg.svd(Lambda.T)
p0 = vh[-1]
p0 /= np.sum(p0)

#интервалы для Y
y1_intervals = np.array([[0.01 , 5 + 0.01 ],
                         [0.015, 5 + 0.05 ],
                         [0.02 , 5 + 0.02 ],
                         [0.022, 5 + 0.022]])

y2_intervals = np.array([[0.001, 0.03 ],
                         [0.01 , 0.05 ],
                         [0.04 , 0.08 ],
                         [0.06 , 0.10 ]])

y_intervals = [y1_intervals, y2_intervals]


#параметры сетки
num1 = 81 #число узлов

# TODO normalno
delta1 = (y1_intervals[0, 1] - y1_intervals[0, 0]) / (num1 - 1) #шаг по координате
delta2 = (y2_intervals[0, 1] - y2_intervals[0, 0]) / (num1 - 1)


deltas = [delta1, delta2]
delta = np.prod(deltas)


points3_1 = np.array([[y1_intervals[state][0],
                    (y1_intervals[state][0] + y1_intervals[state][1])/2,
                     y1_intervals[state][1]] for state in range(N)])
# points3_2 = np.array([[y2_intervals[state][0],
#                     (y2_intervals[state][0] + y2_intervals[state][1])/2,
#                      y2_intervals[state][1]] for state in range(N)])

p_3point = np.array([1/3, 1/3, 1/3])

nets = [
    [
        np.linspace(y1_intervals[n, 0], y1_intervals[n, 1], num=num1), 
        np.linspace(y2_intervals[n, 0], y2_intervals[n, 1], num=num1)
    ] for n in range(N)
]

M_net = np.array([cartesian_product(net) for net in nets])

M = 2

R = 1
L = 1

lam = np.diagonal(Lambda).copy()
Lam = Lambda - np.diag(lam)


pi_uniform, pi_3point, pi_triangular_2, pi_arcsine,\
    y_means_uniform, y_means_arcsine, y_means_triang, y_means_3point =\
        get_distributions(N, M_net, nets, y_intervals, deltas)

def gamma_d(x, alpha, beta, shift):
    from scipy.special import gammaln
    res = np.zeros_like(x)
    s = x[x >= shift]
    res[x >= shift] = np.exp(
        alpha * np.log(beta) + (alpha - 1) * np.log(s) - beta * s - gammaln(alpha)
    )
    return res

def cond_d(x, a, b, x1):
    if (x < a) or (x > b):
        return 0.
    return 1/(b - a) * (2/np.pi * np.atan(x1) * (x - (a + b) / 2) + 1)

pi_dependent = np.zeros_like(pi_uniform)
for n in range(N):
    _pi_y1_n = gamma_d(M_net[n, :, 0], alpha=10 + n * 5 if n != 3 else 22, beta=2000 if n != 3 else 1000, shift=y1_intervals[n][0])
    for i in range(pi_dependent.shape[1]):
        pi_dependent[n, i] = _pi_y1_n[i] * cond_d(M_net[n, i, 1], y2_intervals[n][0], y2_intervals[n][1], M_net[n, i, 0])
    

F = np.array([np.hstack([g(-1, M_net[n], -1), h(-1, M_net[n], -1)]) for n in range(N)])
G = np.array([np.hstack([sigma(-1, M_net[n], -1)**2, h(-1, M_net[n], -1)]) for n in range(N)])

def get_obs(t_net_filtering, theta, y, t):
    dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t, 1)
    deta = make_discretized_eta(t_net_filtering, h, theta, y, t)
    deta = deta.reshape(-1, 1)
    return np.stack([dxi[1:], deta[1:]], axis=-1)

# def get_obs(t_net_filtering, theta, y, t):
#     dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t, R)
#     return np.stack([dxi[1:], ], axis=-1)
