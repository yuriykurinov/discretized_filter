"""Плотности приращения наблюдений и фабрика ``obs_density`` для ``core.filter``.

Интерфейс: ``obs_density(obs[K], u[R], comp[R, K, P], n_comp) -> float64``, где
``u[r]`` -- время пребывания в r-й компоненте (r < n_comp, sum u = ht), а
``comp[r, k, :]`` -- параметры k-го канала в этой компоненте.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numba as nb

# Коды встроенных видов каналов.
NORMAL = 0
POISSON = 1
PARETO = 2
EXPONENTIAL = 3
UNIFORM = 4

# Число используемых слотов параметров для каждого вида канала.
_N_PARAMS = {
    NORMAL: 2,   # (снос, дисперсия)
    POISSON: 1,  # (интенсивность,)
    PARETO: 3,   # (loc, scale, alpha)
    EXPONENTIAL: 2,  # (loc, scale)
    UNIFORM: 2,      # (loc, scale)
}


def n_params_for(kind):
    """Число слотов параметров, которые использует канал вида ``kind``."""
    kind = int(kind)
    if kind not in _N_PARAMS:
        raise ValueError(
            f'неизвестный вид канала наблюдения: {kind}; '
            f'доступны NORMAL={NORMAL}, POISSON={POISSON}, PARETO={PARETO}, '
            f'EXPONENTIAL={EXPONENTIAL}, UNIFORM={UNIFORM}'
        )
    return _N_PARAMS[kind]


@dataclass(frozen=True)
class ObsChannel:
    """Спецификация одного канала наблюдения для сборщика конфига.

    Все поля, кроме ``kind`` и ``noise``, -- джитованные функции
    ``(t, y, theta) -> float64[:, :]``. ``kind`` задаёт плотность
    правдоподобия, ``loc``/``scale``/``alpha`` и ``noise`` -- порождающий
    процесс, поэтому они допустимы при любом ``kind``.
    """
    kind: int
    drift: Callable = None
    var: Callable = None
    intensity: Callable = None
    noise: Callable = None
    loc: Callable = None
    scale: Callable = None
    alpha: Callable = None

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
        elif kind == PARETO:
            if self.loc is None or self.scale is None or self.alpha is None:
                raise ValueError(
                    'ObsChannel(kind=PARETO) требует loc, scale и alpha'
                )
        elif kind in (EXPONENTIAL, UNIFORM):
            if self.loc is None or self.scale is None:
                name = 'EXPONENTIAL' if kind == EXPONENTIAL else 'UNIFORM'
                raise ValueError(
                    f'ObsChannel(kind={name}) требует loc и scale'
                )
            if self.alpha is not None:
                name = 'EXPONENTIAL' if kind == EXPONENTIAL else 'UNIFORM'
                raise ValueError(
                    f'ObsChannel(kind={name}) не использует alpha'
                )
        if self.noise is not None and self.intensity is not None:
            raise ValueError(
                'ObsChannel.noise несовместим с intensity'
            )
        pareto_gen = (self.loc, self.scale, self.alpha)
        if kind not in (EXPONENTIAL, UNIFORM) and (
                any(f is not None for f in pareto_gen)
                and any(f is None for f in pareto_gen)):
            raise ValueError(
                'ObsChannel.loc/scale/alpha задаются все вместе или ни одного'
            )


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath=True, cache=True)
def normal_pdf(x, mu, var):
    """Плотность N(mu, var) в точке x."""
    return math.exp(-(x - mu)**2 / (2 * var)) / math.sqrt(2 * math.pi * var)


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
         fastmath=True, cache=True)
def pareto_obs_pdf(x, loc, scale, alpha):
    """Плотность ``x = loc + scale*(1 + Z/sqrt(12))``, Z -- стандартизованное Парето."""
    if scale <= 0.0 or alpha <= 2.0:
        return 0.0
    m = alpha / (alpha - 1.0)
    s = math.sqrt(alpha / (alpha - 2.0)) / (alpha - 1.0)
    j = s * math.sqrt(12.0) / scale
    p = m + j * (x - loc - scale)
    if p < 1.0:
        return 0.0
    return alpha / p**(alpha + 1.0) * j


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64),
         fastmath=True, cache=True)
def exponential_obs_pdf(x, loc, scale):
    """Плотность ``loc + scale*eps``,
    ``eps = Exp(rate=sqrt(12)) + 1 - 1/sqrt(12)``."""
    if scale <= 0.0:
        return 0.0
    sqrt12 = math.sqrt(12.0)
    a = 1.0 - 1.0 / sqrt12
    e = (x - loc) / scale
    if e < a:
        return 0.0
    return sqrt12 / scale * math.exp(-sqrt12 * (e - a))


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64),
         fastmath=True, cache=True)
def uniform_obs_pdf(x, loc, scale):
    """Плотность ``loc + scale*eps``, ``eps ~ U[0.5, 1.5]``."""
    if scale <= 0.0:
        return 0.0
    e = (x - loc) / scale
    if e < 0.5 or e > 1.5:
        return 0.0
    return 1.0 / scale


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def poisson_pmf(x, lam):
    """Вероятность Pois(lam) в точке x (x предполагается целым >= 0)."""
    # lam <= 0 -- точечная масса в нуле; иначе получили бы log(0) и NaN.
    if lam <= 0.0:
        return 1.0 if x == 0.0 else 0.0
    return math.exp(x * math.log(lam) - lam - math.lgamma(x + 1.0))


def make_obs_density(kinds):
    """Собрать джитованную ``obs_density`` для набора видов каналов."""
    # Замыкание по однородному кортежу кодов: numba допускает динамическую
    # индексацию UniTuple, поэтому цикл по каналам обходится без literal_unroll.
    kinds = tuple(int(k) for k in kinds)
    for kind in kinds:
        n_params_for(kind)  # проверка кода вида канала

    @nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :, :],
                        nb.int64), fastmath=True)
    def obs_density(obs, u, comp, n_comp):
        ht = 0.0
        for r in range(n_comp):
            ht += u[r]
        res = 1.0
        for k in range(len(kinds)):
            kind = kinds[k]
            if kind == NORMAL:
                # безгранично делимый случай: параметры линейны по u
                mu = 0.0
                v = 0.0
                for r in range(n_comp):
                    mu += u[r] * comp[r, k, 0]
                    v += u[r] * comp[r, k, 1]
                res *= normal_pdf(obs[k], mu, v)
            elif kind == POISSON:
                lam = 0.0
                for r in range(n_comp):
                    lam += u[r] * comp[r, k, 0]
                res *= poisson_pmf(obs[k], lam)
            elif kind == PARETO:
                # Парето не безгранично делимо: смесь с весами u[r]/ht
                d = 0.0
                for r in range(n_comp):
                    d += (u[r] / ht) * pareto_obs_pdf(
                        obs[k], comp[r, k, 0], comp[r, k, 1], comp[r, k, 2]
                    )
                res *= d
            elif kind == EXPONENTIAL:
                # Точная location-scale модель: смесь по временам пребывания.
                d = 0.0
                for r in range(n_comp):
                    d += (u[r] / ht) * exponential_obs_pdf(
                        obs[k], comp[r, k, 0], comp[r, k, 1]
                    )
                res *= d
            elif kind == UNIFORM:
                # Точная location-scale модель: смесь по временам пребывания.
                d = 0.0
                for r in range(n_comp):
                    d += (u[r] / ht) * uniform_obs_pdf(
                        obs[k], comp[r, k, 0], comp[r, k, 1]
                    )
                res *= d
        return res

    return obs_density
