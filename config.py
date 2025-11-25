import numpy as np
from numba import njit
from math import ceil
from utils import set_seed, cartesian_product, get_distributions

seed = 123

N = 4

np.random.seed(seed)

@njit(fastmath=True, nogil=True, cache=True)
def h(t, y, theta):
    return (y[:, 1] / y[:, 0])

@njit(nogil=True, cache=True)
def g(t, y, theta):
    return y[:, 0]

sigma_obs = 0.01#0.0005
@njit(nogil=True, cache=True)
def sigma(t, y, theta, b=sigma_obs):
    return b*y[:, 0]

rng = np.random.default_rng(seed=seed)
set_seed(seed)

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

#количество состояний
N = Lambda.shape[0]

#правая граница временного промежутка
T = 100

ht = 1e-2 #шаг фильтрации
ht0 = np.sqrt(ht)
#сетка
t_net_filtering = np.array([t*ht for t in range(ceil(T/ht))])

_, _, vh = np.linalg.svd(Lambda.T)
p0 = vh[-1]
p0 /= np.sum(p0)

#интервалы для Y
y1_intervals = np.array([[0.02,  0.0285],
                         [0.0285, 0.036],
                         [0.036, 0.0435],
                         [0.0435, 0.070]])

y2_intervals = np.array([[0.001, 0.02],
                         [0.02,  0.045],
                         [0.045,  0.07],
                         [0.07,  0.10]]) * 10000

y_intervals = [y1_intervals, y2_intervals]


#параметры сетки
num1 = 25 #число узлов
a1 = np.min(y1_intervals)
b1 = np.max(y1_intervals)
net1 = np.linspace(a1, b1, num=num1)
delta1 = (b1 - a1) / (num1 - 1) #шаг по координате

num2 = 100
a2 = np.min(y2_intervals)
b2 = np.max(y2_intervals)
net2 = np.linspace(a2, b2, num=num2)
delta2 = (b2 - a2) / (num2 - 1)

deltas = [delta1, delta2]


points3_1 = np.array([[y1_intervals[state][0],
                    (y1_intervals[state][0] + y1_intervals[state][1])/2,
                     y1_intervals[state][1]] for state in range(N)])
points3_2 = np.array([[y2_intervals[state][0],
                    (y2_intervals[state][0] + y2_intervals[state][1])/2,
                     y2_intervals[state][1]] for state in range(N)])

p_3point = np.array([1/3, 1/3, 1/3])

nets = [net1, net2]

M_net = cartesian_product(nets)

M = 2

R = 1
L = 1

lam = np.diagonal(Lambda).copy()
Lam = Lambda - np.diag(lam)

delta = np.prod(deltas)

pi_uniform, pi_3point, pi_triangular_2, pi_arcsine,\
    y_means_uniform, y_means_arcsine, y_means_triang, y_means_3point =\
        get_distributions(N, M_net, nets, y_intervals, deltas)

