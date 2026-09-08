"""Гауссовская ЦПТ-аппроксимация канала PARETO с двумя каналами наблюдений.

Блок из чётного числа 2n исходных наблюдений ``xi_k`` (шаг ``pareto_obs``,
``ht=1``) сворачивается в два числа:

    S1 = sum_{k=1}^{2n} xi_k,      S2 = sum_{k=1}^{2n} (-1)^k xi_k.

При постоянном состоянии ``(theta, Y)`` на блоке ``E xi_k = Y1 + Y2``,
``Var xi_k = Y2^2/12``, поэтому

    E S1 = 2n (Y1 + Y2),   Var S1 = 2n Y2^2/12,
    E S2 = 0,              Var S2 = 2n Y2^2/12,
    Cov(S1, S2) = Y2^2/12 * sum_k (-1)^k = 0.

Знакопеременный канал несёт ту же дисперсию, но нулевое среднее, и не
коррелирован с первым, так что совместное распределение аппроксимируется
``N([m, 0], diag(v, v))`` -- произведением двух независимых нормальных
плотностей, ровно как их перемножает ``make_obs_density``.

``Filter`` умножает снос и дисперсию на переданный ``ht``, поэтому ``drift``
и ``var`` здесь заданы в единицах ОДНОГО исходного наблюдения: при
``Filter(..., ht=2n, ...)`` получаются выписанные выше моменты.

Конфиг предназначен только для фильтрации по уже готовым наблюдениям:
``get_obs`` породил бы два независимых канала, тогда как S2 -- детерминированная
функция тех же ``xi_k``.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL
from discretized_filter.utils.distributions import pareto_std

exp_id = 'pareto_obs_approx_2ch'

ALPHA = 2.5

two_jumps = False

n_points = 2

# T, ht, seed, N, Lambda, y_intervals, num1 должны совпадать с pareto_obs.py
# и pareto_obs_approx.py
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


@njit(nogil=True)
def noise(size):
    return pareto_std(size, ALPHA)


channels = [
    ObsChannel(kind=NORMAL, drift=drift_sum, var=var_sum, noise=noise),
    ObsChannel(kind=NORMAL, drift=drift_alt, var=var_alt, noise=noise),
]
