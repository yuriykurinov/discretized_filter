"""Лёгкий пример: треугольная метка Y и аддитивные гауссовские наблюдения.

Два состояния цепи, две координаты метки, постоянное СКО шума 0.7.
Шаг ht относится к фильтру; наблюдения в ноутбуке генерируются с шагом 0.001.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import NORMAL, ObsChannel

exp_id = 'toy_continuous_additive'
T = 4.0
ht = 0.02
seed = 20260906
N = 2
num1 = 6
n_points = 2
two_jumps = False
pi_family = 'triangular'

Lambda = np.array([[0.0, 0.4], [0.3, 0.0]])
y_intervals = [
    np.array([[-1.5, 0.5], [0.5, 2.5]]),
    np.array([[0.0, 2.0], [2.0, 4.0]]),
]


@njit(nogil=True, cache=True)
def drift_first(t, y, theta):
    return y[:, 0:1]


@njit(nogil=True, cache=True)
def drift_second(t, y, theta):
    return y[:, 1:2]


@njit(nogil=True, cache=True)
def constant_variance(t, y, theta):
    return np.full((y.shape[0], 1), 0.7 ** 2)


channels = [
    ObsChannel(kind=NORMAL, drift=drift_first, var=constant_variance),
    ObsChannel(kind=NORMAL, drift=drift_second, var=constant_variance),
]
