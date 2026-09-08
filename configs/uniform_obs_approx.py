"""Гауссовская аппроксимация UNIFORM по двум моментам, ht=10.

Одно наблюдение фильтра является суммой 10 шагов ``uniform_obs``.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import NORMAL, ObsChannel
from discretized_filter.utils.distributions import uniform_std

exp_id = 'uniform_obs_approx'

two_jumps = False

n_points = 2

T = 24 * 3600

ht = 10.0

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
def drift(t, y, theta):
    return y[:, 0:1] + y[:, 1:2]


@njit(nogil=True, cache=True)
def var(t, y, theta):
    return y[:, 1:2]**2 / 12.0


@njit(nogil=True)
def noise(size):
    return uniform_std(size)


channels = [
    ObsChannel(kind=NORMAL, drift=drift, var=var, noise=noise),
]
