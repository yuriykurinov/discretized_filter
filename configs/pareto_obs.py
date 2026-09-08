"""Точный фильтр по каналу PARETO: наблюдение X = loc + scale*eps, eps -- Парето."""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, PARETO

exp_id = 'pareto_obs_exact'

ALPHA = 2.5

two_jumps = False

n_points = 2

T = 24 * 3600

ht = 1.0

seed = 20260803

N = 4

Lambda = np.full((4, 4), 1.0 / (3 * 3600))
np.fill_diagonal(Lambda, 0.0)

y1_intervals = np.array([[1., 5.], [5., 10.], [10., 15.], [15., 20.]])
y2_intervals = np.array([[10., 20.], [20., 30.], [30., 40.], [40., 50.]])
y_intervals = [y1_intervals, y2_intervals]

num1 = [20, 20]

pi_family = 'uniform'

@njit(nogil=True, cache=True)
def loc(t, y, theta):
    return y[:, 0:1]


@njit(nogil=True, cache=True)
def scale(t, y, theta):
    return y[:, 1:2]


@njit(nogil=True, cache=True)
def alpha(t, y, theta):
    return np.full((y.shape[0], 1), ALPHA)


channels = [
    ObsChannel(kind=PARETO, loc=loc, scale=scale, alpha=alpha),
]
