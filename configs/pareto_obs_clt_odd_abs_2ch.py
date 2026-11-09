"""ЦПТ-модель: сумма нечётных O и модуль знакопеременной суммы чётных |W|.

Единичные моменты одного исходного наблюдения X: mu = Y1+Y2, v = Y2**2/12.
Канал 1: O = sum X_{нечётные} ~ N((B/2) mu, (B/2) v); конфиг (mu/2, v/2).
Канал 2: W = X_2 - X_4 + X_6 - ... ~ N(0, (B/2) v) при B % 4 == 0; в фильтр
подаётся |W| с полунормальной плотностью, конфиг: снос 0, дисперсия v/2.
O и W построены по непересекающимся наблюдениям, условно независимы по
скрытой траектории. Для DiscreteFilter копия C умножается на B один раз,
шаг прогноза ht=B. Наблюдения строятся из одного исходного потока, не get_obs.
План: plans/20261109-channel-clt-odd-abs.md.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL, HALF_NORMAL

exp_id = 'pareto_obs_clt_odd_abs_2ch'

two_jumps = False

n_points = 2

# Параметры скрытого процесса совпадают с pareto_obs.py.
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
def drift_half(t, y, theta):
    return (y[:, 0:1] + y[:, 1:2]) / 2.0


@njit(nogil=True, cache=True)
def var_half(t, y, theta):
    return y[:, 1:2]**2 / 24.0


channels = [
    ObsChannel(kind=NORMAL, drift=drift_half, var=var_half),
    ObsChannel(kind=HALF_NORMAL, var=var_half),
]
