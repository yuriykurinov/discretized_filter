"""
Игрушечная модель для сравнения диффузионной аппроксимации с точным
пуассоновским правдоподобием: theta -- 2 состояния, Y -- одномерное,
два канала наблюдения (непрерывный и считающий).

Здесь считающий канал объявлен NORMAL со средним = дисперсией = h -- это
ДИФФУЗИОННАЯ АППРОКСИМАЦИЯ. Парный конфиг toy_exact_poisson.py задаёт для
него точное пуассоновское правдоподобие; порождающая модель у конфигов
одна и та же, поэтому при одном seed они видят одну траекторию и одни и те
же наблюдения.

Интенсивность подобрана так, чтобы за шаг сетки набегало ~0.2-0.5 отсчёта:
в этом режиме пуассоновское приращение заметно не гауссово, и цена
аппроксимации видна.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'toy_diffusion_approx'

two_jumps = False   # в ноутбуке переопределяется при создании Filter

n_points = 2

T = 10

ht = 0.05

seed = 20260803

N = 2


# --- непрерывный канал: снос y, диффузия 0.3*sqrt(y) ---
@njit(nogil=True, cache=True)
def drift_c(t, y, theta):
    return y[:, 0:1]


@njit(nogil=True, cache=True)
def var_c(t, y, theta):
    return 0.09 * y[:, 0:1]


# --- считающий канал: интенсивность 5*y ---
@njit(nogil=True, cache=True)
def intensity(t, y, theta):
    return 5.0 * y[:, 0:1]


# внедиагональные интенсивности перехода; диагональ строит сборщик конфига
Lambda = np.array([
    [0.0, 1.0],
    [0.8, 0.0],
])

# носители Y по состояниям (одна координата)
y_intervals = [
    np.array([[0.5, 1.0],
              [1.5, 2.5]])
]

num1 = 64

pi_family = 'uniform'

channels = [
    ObsChannel(kind=NORMAL, drift=drift_c, var=var_c),
    ObsChannel(kind=NORMAL, drift=intensity, var=intensity, intensity=intensity),
]
