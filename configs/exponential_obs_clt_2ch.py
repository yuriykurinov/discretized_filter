"""ЦПТ-модель полной суммы S и знакопеременной суммы Z.

Z = sum_even X - sum_odd X. Единичные моменты (mu,v),(0,v);
для DiscreteFilter копия C умножается на B один раз; шаг прогноза ht=B. Независимость S,Z — ЦПТ-приближение.
Наблюдения строятся из одного исходного потока, не get_obs.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'exponential_obs_clt_2ch'

two_jumps = False

n_points = 2

# Параметры скрытого процесса совпадают с exponential_obs.py.
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
def drift_sum(t, y, theta):
    return y[:, 0:1] + y[:, 1:2]


@njit(nogil=True, cache=True)
def var_sum(t, y, theta):
    return y[:, 1:2]**2 / 12.0


@njit(nogil=True, cache=True)
def drift_alt(t, y, theta):
    # знакопеременная сумма центрирована: снос сокращается по парам слагаемых
    return np.zeros((y.shape[0], 1))


@njit(nogil=True, cache=True)
def var_alt(t, y, theta):
    # знаки не влияют на дисперсию, поэтому она та же, что у прямой суммы
    return y[:, 1:2]**2 / 12.0


channels = [
    ObsChannel(kind=NORMAL, drift=drift_sum, var=var_sum),
    ObsChannel(kind=NORMAL, drift=drift_alt, var=var_alt),
]
