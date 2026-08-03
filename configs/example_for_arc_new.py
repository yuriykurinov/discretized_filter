"""
Конфиг example_for_arc_new -- порт прежнего плоского config.py на систему
configs/ + ObsChannel. Оба канала наблюдения объявлены NORMAL (канал xi:
drift=g, var=sigma**2; канал eta: drift=h, var=h).

Нормальная плотность на считающем канале eta (среднее = дисперсия = h) --
это ДИФФУЗИОННАЯ АППРОКСИМАЦИЯ пуассоновского приращения, то есть предмет
исследования, а не упрощение по недосмотру. Она сохранена в точности.

Канал eta дополнительно несёт intensity=h: это не влияет на плотность
правдоподобия фильтра (она берётся из kind=NORMAL, drift, var), но говорит
get_obs, что истинный порождающий процесс наблюдения -- считающий, как и было
в исходном коде (make_discretized_eta). Благодаря этому example_poisson_eta.py
задаёт ту же порождающую модель и ту же траекторию, отличаясь только видом
плотности этого канала, -- что и позволяет сравнить диффузионную
аппроксимацию с точным пуассоновским правдоподобием количественно.
"""
import numpy as np
import numba as nb
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'example_for_arc_new'

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
    ObsChannel(kind=NORMAL, drift=h, var=h, intensity=h),
]
