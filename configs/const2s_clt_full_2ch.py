"""ЦПТ-модель (постоянный сигнал, 2 состояния). Сумма S и знакопеременная сумма Z: NORMAL(mu, v), NORMAL(0, v).
Единичные моменты X: mu = Y1+Y2, v = Y2**2/12. Скрипт умножает копию C на B
и переопределяет ht = B. План: plans/20261109-channel-clt-constant.md.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'const2s_clt_full_2ch'

two_jumps = False

n_points = 2

# Постоянный сигнал: 2 состояния, Lambda = 0; T не используется как горизонт.
T = 24 * 3600

ht = 10.0

seed = 20260803

N = 2

Lambda = np.zeros((2, 2))

p0 = np.array([0.5, 0.5])

y1_intervals = np.array([[5., 10.], [10., 15.]])
y2_intervals = np.array([[10., 15.], [15., 20.]])
y_intervals = [y1_intervals, y2_intervals]

num1 = [50, 50]

pi_family = 'uniform'


@njit(nogil=True, cache=True)
def drift_full(t, y, theta):
    return y[:, 0:1] + y[:, 1:2]


@njit(nogil=True, cache=True)
def var_full(t, y, theta):
    return y[:, 1:2]**2 / 12.0


@njit(nogil=True, cache=True)
def drift_zero(t, y, theta):
    return np.zeros((y.shape[0], 1))


channels = [
    ObsChannel(kind=NORMAL, drift=drift_full, var=var_full),
    ObsChannel(kind=NORMAL, drift=drift_zero, var=var_full),
]
