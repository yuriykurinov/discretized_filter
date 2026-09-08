"""
Конфиг example_poisson_eta -- идентичен example_for_arc_new.py, но канал
eta объявлен POISSON: фильтр использует точное пуассоновское правдоподобие
(exp(x*log(intensity) - intensity - lgamma(x+1))).

Это ЭТАЛОН ДЛЯ СРАВНЕНИЯ, а не замена: в example_for_arc_new.py тот же канал
фильтруется нормальной плотностью со средним = дисперсией, и это не упрощение,
а диффузионная аппроксимация -- самостоятельный предмет исследования
(см. results/diffusion_approx_*). Истинный порождающий процесс наблюдения
(get_obs, make_discretized_eta) у обоих конфигов один и тот же, поэтому при
одном seed они видят одну и ту же траекторию и одни и те же наблюдения, и
разница в оценках -- это ровно цена диффузионной аппроксимации.
"""
import numpy as np
import numba as nb
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL, POISSON

exp_id = 'example_poisson_eta'

two_jumps = False

n_points = 1  # интегрирование по времени скачка

# правая граница временного промежутка
T = 100

ht = 0.01  # шаг фильтрации

seed = 321

N = 4


@njit(fastmath=True, nogil=True, cache=True)
def h(t, y, theta):
    return 1000 * (y[:, 1:2] / y[:, 0:1])


@njit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.int64
    ),
    nogil=True,
    cache=True
)
def g(t, y, theta):
    return y[:, 0:1]


sigma_obs = 0.1  # 0.0005


@njit(nogil=True, cache=True)
def sigma(t, y, theta, b=sigma_obs):
    return b * np.sqrt(y[:, 0:1])


@njit(nogil=True, cache=True)
def var_g(t, y, theta):
    return sigma(t, y, theta)**2


# внедиагональные интенсивности перехода; диагональ строит сборщик конфига
# (lambda_ii = -sum_{j != i} lambda_ij)
Lambda = np.array(
    [
        [   0, 0.45,   0, 0.05],
        [0.25,    0, 0.1, 0.05],
        [0.75,    0,   0, 0.05],
        [   1,    0,   0,    0]
    ]
)

# интервалы для Y
y1_intervals = np.array([[0.01 , 0.01 + 0.01 ],
                         [0.015, 0.01 + 0.015],
                         [0.02 , 0.01 + 0.02 ],
                         [0.022, 0.03 + 0.022]])

y2_intervals = np.array([[0.001, 0.03 ],
                         [0.01 , 0.05 ],
                         [0.04 , 0.08 ],
                         [0.06 , 0.10 ]])

y_intervals = [y1_intervals, y2_intervals]

# параметры сетки
num1 = 64  # число узлов по каждой координате Y

pi_family = 'uniform'

channels = [
    ObsChannel(kind=NORMAL, drift=g, var=var_g),
    ObsChannel(kind=POISSON, intensity=h),
]
