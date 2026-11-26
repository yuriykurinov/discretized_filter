"""Точный канал X = p + q*tau, tau = 1 + (|Z| - sqrt(2/pi))/(sqrt(12)*sqrt(1-2/pi)), Z ~ N(0, 1); постоянный сигнал."""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, ONESIDED_NORMAL

exp_id = 'const1s_halfnormal_exact'

two_jumps = False

n_points = 2

# Постоянный сигнал: 1 состояние, Lambda = 0; T не используется как горизонт.
T = 24 * 3600

ht = 1.0

seed = 20260803

N = 1

Lambda = np.zeros((1, 1))

p0 = np.array([1.0])

y1_intervals = np.array([[5., 10.]])
y2_intervals = np.array([[10., 15.]])
y_intervals = [y1_intervals, y2_intervals]

num1 = [50, 50]

pi_family = 'uniform'


@njit(nogil=True, cache=True)
def loc(t, y, theta):
    return y[:, 0:1]


@njit(nogil=True, cache=True)
def scale(t, y, theta):
    return y[:, 1:2]


channels = [
    ObsChannel(kind=ONESIDED_NORMAL, loc=loc, scale=scale),
]
