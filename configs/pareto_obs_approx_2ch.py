"""Гауссовская ЦПТ-аппроксимация PARETO со знакопеременным каналом.

Для чётного B из одной исходной выборки X_i строятся
    A = sum X_i / B,      Z = sum (-1)^i X_i, i=1..B.
При постоянном состоянии блока mu=Y1+Y2, v=Y2^2/12:
    E A=mu, Var A=v/B, E Z=0, Var Z=B*v, Cov(A,Z)=0.
Произведение независимых нормальных плотностей — ЦПТ-приближение.
При скачке внутри блока нулевые среднее Z и ковариация уже приближённые;
блоковое правдоподобие не учитывает такие скачки точно.

Коэффициенты конфига остаются единичными: (mu,v) и (0,v).
``DiscreteFilter.update`` использует экспозицию 1; ``ht`` влияет только
на прогноз. Для пары (A,Z) надо копировать C и менять дисперсии:
    C[..., 0, 1] /= B,    C[..., 1, 1] *= B.
Средние остаются mu и 0; шаг прогноза равен B * pareto_obs.ht.

Для прежнего ``Filter`` с парой (sum X_i,Z) и исходным шагом 1
единичные коэффициенты совместимы с ``ht=B``: прежнее правдоподобие
умножает и снос, и дисперсию на экспозицию.

Конфиг предназначен для фильтрации готовых преобразований одной выборки.
``get_obs`` породил бы два независимых канала и не строит пару (A,Z).
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'pareto_obs_approx_2ch'

two_jumps = False

n_points = 2

# T, seed, N, Lambda, y_intervals, num1 должны совпадать с pareto_obs.py
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


channels = [
    ObsChannel(kind=NORMAL, drift=drift_sum, var=var_sum),
    ObsChannel(kind=NORMAL, drift=drift_alt, var=var_alt),
]
