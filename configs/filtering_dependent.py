"""
Конфиг filtering_dependent -- то, что реально считает notebooks/filtering.ipynb:
та же модель, что в example_for_arc_new.py, но условная плотность Y при
условии theta берётся из семейства 'dependent_gamma' (координата y2 зависит
от y1), а не равномерная.

ВАЖНО. Прежде плотность этого семейства собиралась вручную в config.py
(gamma_d x cond_d), а разыгрывающий её генератор -- в ячейке самого
ноутбука (get_y_dependent), и они не совпадали: в gamma_d не вычитался сдвиг
shift, из-за чего вместо сдвинутого гамма-распределения бралcя хвост
несдвинутого (L1-расстояние между старой и исправленной формой ~1.86 из
максимальных 2). Семейство 'dependent_gamma' в utils/distributions.py --
исправленная и согласованная пара «плотность + генератор», поэтому
результаты, сохранённые прежней версией, этим конфигом не воспроизводятся.
"""
import numpy as np
import numba as nb
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'filtering_dependent'

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


sigma_obs = 0.1


@njit(nogil=True, cache=True)
def sigma(t, y, theta, b=sigma_obs):
    return b * np.sqrt(y[:, 0:1])


@njit(nogil=True, cache=True)
def var_g(t, y, theta):
    return sigma(t, y, theta)**2


# внедиагональные интенсивности перехода; диагональ строит сборщик конфига
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

pi_family = 'dependent_gamma'

channels = [
    ObsChannel(kind=NORMAL, drift=g, var=var_g),
    ObsChannel(kind=NORMAL, drift=h, var=h, intensity=h),
]
