import numpy as np
import numba as nb
from numba import njit
from math import ceil
from utils import set_seed, cartesian_product, get_distributions
from SMJP import make_discretized_xi, make_discretized_eta

exp_id = 'example_for_ia_6'

two_jumps = False

n_points = 1 # time integration

#правая граница временного промежутка
T = 10

ht = 0.5 #шаг фильтрации


seed = 321

N = 3 # theta dim

np.random.seed(seed)
rng = np.random.default_rng(seed=seed)
set_seed(seed)

@njit(fastmath=True, nogil=True, cache=True)
def h(t, y, theta):
    return (y[:, 1] / y[:, 0])


@njit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.int64
    ),

)
def g(t, y, theta):
    return 0 * y[:, 0:1] #/ (1 - y[:, 0:1])
#np.stack((y[:, 0], y[:, 1] / y[:, 0]), axis=-1)

@njit(
    nb.float64[:, :](
        nb.float64,
        nb.float64[:, :],
        nb.float64
    ), 
)
def sigma(t, y, theta):
    return np.sqrt(y[:, 0:1])
#np.stack((b*y[:, 0], np.sqrt(y[:, 1] / y[:, 0])), axis=-1)

@njit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.int64
    ),
    nogil=True, 
    cache=True
)
def sigma(t, y, theta):
    res = np.sqrt(y[:, 0:2])
    if theta == 0:
        res[:, 1] = 2*res[:, 1]
    else:
        res[:, 0] = 2*res[:, 0]
    return res

Lambda = np.array(
    [
        [0   ,  0.2, 0.2 ],
        [0.01,  0  , 0.05],
        [0.15,  0.1 , 0   ],
    ]
)

for i in range(Lambda.shape[0]):
    Lambda[i][i] = -np.sum(Lambda[i])

#правая граница временного промежутка
T = 100

ht = 1e-2 #шаг фильтрации
#сетка
t_net_filtering = np.array([t*ht for t in range(ceil(T/ht))])

_, _, vh = np.linalg.svd(Lambda.T)
p0 = vh[-1]
p0 /= np.sum(p0)

#интервалы для Y
y1_intervals = np.array([[0.2, 0.2],
                         [0.3, 0.5],
                         [0.4, 0.6]])


y_intervals = [y1_intervals, y2_intervals]


#параметры сетки
num1 = 101 #число узлов

# TODO normalno
delta1 = (y1_intervals[0, 1] - y1_intervals[0, 0]) / (num1 - 1) #шаг по координате
delta2 = (y2_intervals[0, 1] - y2_intervals[0, 0]) / (num1 - 1)


deltas = [delta1, delta2]


nets = [
    [
        np.linspace(y1_intervals[n, 0], y1_intervals[n, 1], num=num1), 
        np.linspace(y2_intervals[n, 0], y2_intervals[n, 1], num=num1)
    ] for n in range(N)
]

M_net = np.array([cartesian_product(net) for net in nets])

M = 2

R = 2
L = 0

lam = np.diagonal(Lambda).copy()
Lam = Lambda - np.diag(lam)

delta = np.prod(deltas)
# TODO deltas[n]

pi_uniform, pi_3point, pi_triangular_2, pi_arcsine,\
    y_means_uniform, y_means_arcsine, y_means_triang, y_means_3point =\
        get_distributions(N, M_net, nets, y_intervals, deltas)

F = np.array([[g(-1, M_net[n], -1), ] for n in range(N)]).squeeze(1)
G = np.array([[sigma(-1, M_net[n], -1)**2, ] for n in range(N)]).squeeze(1)

def get_obs(t_net_filtering, theta, y, t):
    #dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t, R)
    import pickle
    obs = pickle.load(open('saved_path_example_for_arc3_0.001/observations.pkl', 'rb'))
    dxi = [obs[0]]
    window = int(ht / 0.001)
    for i in range(1, obs.shape[0] // window):
        dxi.append(obs[i * window : (i+1) * window].sum(axis=0))
    return np.array(dxi)

# def get_obs(t_net_filtering, theta, y, t):
#     dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t, R)
#     return dxi #np.stack([dxi[1:], ], axis=-1)

