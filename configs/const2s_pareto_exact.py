"""Точный канал PARETO: X = Y1 + Y2*eps, постоянный сигнал, 2 состояния."""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, PARETO

exp_id = 'const2s_pareto_exact'

two_jumps = False

n_points = 2

# Постоянный сигнал: 2 состояния, Lambda = 0; T не используется как горизонт.
T = 24 * 3600

ht = 1.0

seed = 20260803

N = 2

Lambda = np.zeros((2, 2))

p0 = np.array([0.5, 0.5])

y1_intervals = np.array([[5., 10.], [10., 15.]])
y2_intervals = np.array([[10., 15.], [15., 20.]])
y_intervals = [y1_intervals, y2_intervals]

num1 = [50, 50]

pi_family = 'uniform'



ALPHA = 2.5

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
