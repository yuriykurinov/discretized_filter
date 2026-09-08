"""Фильтр по среднему блока (ЦПТ-осреднение) вместо суммы блока; ht=10, как в
``pareto_obs_approx``, но params масштабированы на 1/ratio -- тождественен
``pareto_obs_approx`` по апостериорному распределению."""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL
from discretized_filter.utils.distributions import pareto_std

exp_id = 'pareto_obs_clt'

ALPHA = 2.5

# число исходных наблюдений (шаг pareto_obs) на один шаг этого фильтра;
# замыкается джитованными drift/var ниже, менять после первого вызова
# этих функций бесполезно (numba вмораживает глобалы при компиляции)
ratio = 10

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
def drift(t, y, theta):
    return (y[:, 0:1] + y[:, 1:2]) / ratio


@njit(nogil=True, cache=True)
def var(t, y, theta):
    return y[:, 1:2]**2 / (12.0 * ratio**2)


@njit(nogil=True)
def noise(size):
    return pareto_std(size, ALPHA)


channels = [
    ObsChannel(kind=NORMAL, drift=drift, var=var, noise=noise),
]
