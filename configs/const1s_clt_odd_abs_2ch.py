"""ЦПТ-модель (постоянный сигнал, 1 состояние). Нечётная сумма и модуль знакопеременной чётной: NORMAL(mu/2, v/2), HALF_NORMAL(v/2).
Единичные моменты X: mu = Y1+Y2, v = Y2**2/12. Скрипт умножает копию C на B
и переопределяет ht = B. План: plans/20261109-channel-clt-constant.md.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL, HALF_NORMAL

exp_id = 'const1s_clt_odd_abs_2ch'

two_jumps = False

n_points = 2

# Постоянный сигнал: 1 состояние, Lambda = 0; T не используется как горизонт.
T = 24 * 3600

ht = 10.0

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
def drift_half(t, y, theta):
    return (y[:, 0:1] + y[:, 1:2]) / 2.0


@njit(nogil=True, cache=True)
def var_half(t, y, theta):
    return y[:, 1:2]**2 / 24.0


channels = [
    ObsChannel(kind=NORMAL, drift=drift_half, var=var_half),
    ObsChannel(kind=HALF_NORMAL, var=var_half),
]
