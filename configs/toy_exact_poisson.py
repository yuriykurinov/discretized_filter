"""
Парный к toy_diffusion_approx.py конфиг: та же модель, но считающий канал
объявлен POISSON -- фильтр использует точное пуассоновское правдоподобие.

Порождающая модель (get_obs -> make_discretized_eta) у обоих конфигов
одинакова, отличается только вид плотности, которым фильтр моделирует
правдоподобие этого канала.
"""
import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL, POISSON

exp_id = 'toy_exact_poisson'

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
    ObsChannel(kind=POISSON, intensity=intensity),
]
