"""Плотности приращения наблюдений.

Приращение наблюдения по каналу ``k`` на отрезке длины ``h`` имеет плотность
с вектором параметров

    params[k, :] = sum_j u_j * C[i_j, y_j, k, :],

где ``u_j`` -- времена пребывания траектории ``(theta, Y)`` в состояниях
``(i_j, y_j)``, а ``C[n, y, k, p]`` -- скорость накопления p-го параметра.
Ключевое (и единственное) требование к семейству плотностей: параметры
накапливаются **линейно по временам пребывания** -- это выполнено для любого
безгранично делимого приращения. Формулы (transition_kernel:1-2) из
``article/my_notes/discretized_filter.tex``: нормальный случай -- частный
вариант с P = 2 слотами, ``C[..., 0] = f`` (снос), ``C[..., 1] = gg^T``
(дисперсия).

Интерфейс плотности, которую ждёт ``core.filter``:

    obs_density(obs: float64[:], params: float64[:, :]) -> float64
        obs.shape    == (K,)      наблюдённое приращение по каналам
        params.shape == (K, P)    накопленные параметры
        возвращает совместную плотность prod_k p_k(obs[k]; params[k])

Пользователь может передать в фильтр свою джитованную функцию с такой же
сигнатурой -- ``make_obs_density`` лишь удобная фабрика для встроенных видов
каналов.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numba as nb

# Коды встроенных видов каналов.
NORMAL = 0
POISSON = 1

# Число используемых слотов параметров для каждого вида канала.
_N_PARAMS = {
    NORMAL: 2,   # (снос, дисперсия)
    POISSON: 1,  # (интенсивность,)
}


def n_params_for(kind):
    """Число слотов параметров, которые использует канал вида ``kind``.

    Обычная (не джитованная) функция: вызывается сборщиком конфига, чтобы
    определить P = max_k n_params_for(kinds[k]); лишние слоты заполняются
    нулями.
    """
    kind = int(kind)
    if kind not in _N_PARAMS:
        raise ValueError(
            f'неизвестный вид канала наблюдения: {kind}; '
            f'доступны NORMAL={NORMAL}, POISSON={POISSON}'
        )
    return _N_PARAMS[kind]


@dataclass(frozen=True)
class ObsChannel:
    """Спецификация одного канала наблюдения для сборщика конфига.

    ``drift``/``var``/``intensity`` -- джитованные функции
    ``(t, y, theta) -> float64[:, :]``, как ``g``/``sigma``/``h`` в старом
    ``config.py``. Для ``NORMAL`` заполняются ``drift`` (слот параметров 0,
    снос) и ``var`` (слот 1, дисперсия gg^T); для ``POISSON`` -- только
    ``intensity`` (слот 0).
    """
    kind: int
    drift: Callable = None
    var: Callable = None
    intensity: Callable = None

    def __post_init__(self):
        kind = int(self.kind)
        n_params_for(kind)  # проверка кода вида канала
        if kind == NORMAL:
            if self.drift is None or self.var is None:
                raise ValueError(
                    'ObsChannel(kind=NORMAL) требует drift и var'
                )
        elif kind == POISSON:
            if self.intensity is None:
                raise ValueError(
                    'ObsChannel(kind=POISSON) требует intensity'
                )


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath=True, cache=True)
def normal_pdf(x, mu, var):
    """Плотность N(mu, var) в точке x."""
    return math.exp(-(x - mu)**2 / (2 * var)) / math.sqrt(2 * math.pi * var)


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def poisson_pmf(x, lam):
    """Вероятность Pois(lam) в точке x (x предполагается целым >= 0).

    Вырожденный случай lam <= 0 -- это точечная масса в нуле; он возникает,
    когда интенсивность канала обнуляется на всей траектории, и обрабатывается
    явно, чтобы не получить log(0) и NaN.
    """
    if lam <= 0.0:
        return 1.0 if x == 0.0 else 0.0
    return math.exp(x * math.log(lam) - lam - math.lgamma(x + 1.0))


def make_obs_density(kinds):
    """Собрать джитованную ``obs_density`` для набора видов каналов.

    ``kinds`` -- последовательность целых кодов (NORMAL / POISSON), по одному
    на канал. Возвращается замыкание по однородному кортежу кодов: numba
    допускает динамическую индексацию UniTuple, поэтому цикл по каналам
    компилируется без ``literal_unroll``.
    """
    kinds = tuple(int(k) for k in kinds)
    for kind in kinds:
        n_params_for(kind)  # проверка кода вида канала

    @nb.njit(nb.float64(nb.float64[:], nb.float64[:, :]), fastmath=True)
    def obs_density(obs, params):
        res = 1.0
        for k in range(len(kinds)):
            kind = kinds[k]
            if kind == NORMAL:
                res *= normal_pdf(obs[k], params[k, 0], params[k, 1])
            elif kind == POISSON:
                res *= poisson_pmf(obs[k], params[k, 0])
        return res

    return obs_density
