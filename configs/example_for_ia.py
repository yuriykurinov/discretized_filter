"""
Конфиг example_for_ia -- порт сохранённой копии конфига, которым считался
пример для статьи (results/example_for_ia/config.py).

Отличия от example_for_arc_new, ради которых сборщик конфига поддерживает
два дополнительных параметра:

* ``shared_grid = True`` -- одна общая сетка по Y на все состояния,
  построенная по объединению носителей, а не своя сетка на каждое состояние;
* ``num1 = [20, 100]`` -- своё число узлов по каждой координате Y.

Оба канала наблюдения непрерывные (NORMAL), считающего канала здесь нет:
наблюдается двумерный диффузионный процесс со сносом
(y1, y2/y1) и диффузией (b*y1, sqrt(y2/y1)).

ВНИМАНИЕ: прежний get_obs разыгрывал оба канала одним вызовом
make_discretized_xi с xi_dim=2, новый -- отдельным вызовом на канал.
Распределение наблюдений то же самое (каналы независимы, матрица диффузии
диагональна), но порядок обращений к ГПСЧ другой, поэтому сохранённые ранее
траектории этим конфигом бит-в-бит не воспроизводятся.
"""
import numpy as np
import numba as nb
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'example_for_ia'

two_jumps = False

n_points = 2  # интегрирование по времени скачка

# правая граница временного промежутка
T = 100

ht = 1e-1  # шаг фильтрации

seed = 123

N = 4

sigma_obs = 0.01


# --- канал 1: снос y1, диффузия b*y1 ---
@njit(nogil=True, cache=True)
def drift_1(t, y, theta):
    return y[:, 0:1]


@njit(nogil=True, cache=True)
def var_1(t, y, theta, b=sigma_obs):
    return (b * y[:, 0:1])**2


# --- канал 2: снос y2/y1, диффузия sqrt(y2/y1) ---
@njit(nogil=True, cache=True)
def drift_2(t, y, theta):
    return y[:, 1:2] / y[:, 0:1]


@njit(nogil=True, cache=True)
def var_2(t, y, theta):
    return y[:, 1:2] / y[:, 0:1]


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
y1_intervals = np.array([[0.02  , 0.0285],
                         [0.0285, 0.036 ],
                         [0.036 , 0.0435],
                         [0.0435, 0.070 ]])

y2_intervals = np.array([[0.001, 0.02 ],
                         [0.02 , 0.045],
                         [0.045, 0.07 ],
                         [0.07 , 0.10 ]])

y_intervals = [y1_intervals, y2_intervals]

# параметры сетки: своё число узлов по каждой координате, общая сетка
# на все состояния
num1 = [20, 100]
shared_grid = True

pi_family = 'uniform'

channels = [
    ObsChannel(kind=NORMAL, drift=drift_1, var=var_1),
    ObsChannel(kind=NORMAL, drift=drift_2, var=var_2),
]
