"""
Плотности π(y|θ) и парные им джитованные генераторы Y.

Реестр семейств: register_pi_family/get_pi_family; build_pi -- точка входа
для конфига.
"""
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

from discretized_filter.utils.grids import cartesian_product


# ---------------------------------------------------------------------------
# Плотность приращения наблюдений — используется в core/filter.py, не трогать
# ---------------------------------------------------------------------------
@njit(fastmath=True, nogil=True)
def norm(x, mu, sigma_sq):
    return np.exp(-(x - mu)**2 / (2*sigma_sq)) / np.sqrt(2*np.pi*sigma_sq)


# ---------------------------------------------------------------------------
# Негауссовский шум для непрерывного канала наблюдения (core/smjp.py).
# ---------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=True)
def pareto_std(size, alpha):
    """size стандартизованных значений Z = (Pareto(x_m=1, alpha) - m)/s."""
    m = alpha / (alpha - 1.0)
    s = np.sqrt(alpha / (alpha - 2.0)) / (alpha - 1.0)
    res = np.empty(size)
    for i in range(size):
        p = (1.0 - np.random.random())**(-1.0 / alpha)
        res[i] = (p - m) / s
    return res


@njit(fastmath=True, nogil=True, cache=True)
def exponential_std(size):
    """Стандартизованный шум Exp(1) - 1 со средним 0 и дисперсией 1."""
    res = np.empty(size)
    for i in range(size):
        res[i] = np.random.exponential(1.0) - 1.0
    return res


@njit(fastmath=True, nogil=True, cache=True)
def uniform_std(size):
    """Стандартизованный шум U[-sqrt(3), sqrt(3)]."""
    res = np.empty(size)
    bound = np.sqrt(3.0)
    for i in range(size):
        res[i] = np.random.uniform(-bound, bound)
    return res


# ---------------------------------------------------------------------------
# Одномерные (по одной координате Y) ненормированные плотности
# ---------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=True)
def unif(x, a, b):
    return (x <= b) * (x >= a) / (b - a)


def triang_1d(x, a, b):
    """Ненормированная симметричная треугольная плотность на [a, b]."""
    c = (a + b) / 2
    return (x <= b) * (x >= a) * np.abs(b - c - np.abs(c - x)) / (b - c)**2


def triang_right_1d(x, a, b):
    """Ненормированная треугольная плотность с модой в правом конце b."""
    return 2 * (x >= a) * (x <= b) * (x - a) / (b - a)**2


@njit(fastmath=True)
def arcsine_1d(x, a, b):
    """Ненормированная плотность арксинус-распределения на [a, b]."""
    res = np.zeros(x.shape)
    idx_special = []
    for i in range(x.shape[0]):
        s1 = x[i] - a
        s2 = b - x[i]
        if (s1 > 0) and (s2 > 0):
            res[i] = np.exp(-0.5 * (np.log(s1) + np.log(s2)))
        elif (s1 < 0) or (s2 < 0):
            continue
        else:
            idx_special.append(i)
    if len(idx_special) > 0:
        for i in idx_special:
            res[i] = np.max(res)
    return res


# ---------------------------------------------------------------------------
# Генераторы Y по θ
# ---------------------------------------------------------------------------
@njit(fastmath=True)
def get_y_uniform(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.uniform(y_interval[state][0], y_interval[state][1])
        )
    return res


@njit(fastmath=True)
def get_y_triangular(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.triangular(y_interval[state][0],
                                (y_interval[state][1] + y_interval[state][0]) / 2,
                                 y_interval[state][1])
        )
    return res


@njit(fastmath=True)
def get_y_triangular_2(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(
            np.random.triangular(y_interval[state][0],
                                 y_interval[state][1],
                                 y_interval[state][1])
        )
    return res


@njit(fastmath=True)
def choice3(a, b, c):
    """Равновероятный выбор одного из трёх значений (для 3-точечного распределения)."""
    u = np.random.uniform()
    if u < 1/3:
        return a
    elif u < 2/3:
        return b
    return c


@njit(fastmath=True)
def get_y_3point(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        a = y_interval[state][0]
        b = y_interval[state][1]
        res.append(choice3(a, (a + b) / 2, b))
    return res


@njit(fastmath=True)
def arcsine_dist_standard():
    return (np.sin(np.random.uniform(0, 2*np.pi)) + 1) / 2


@njit(fastmath=True)
def arcsine_dist(a, b):
    return arcsine_dist_standard() * (b - a) + a


@njit(fastmath=True)
def get_y_arcsine(state, y_intervals):
    res = []
    for y_interval in y_intervals:
        res.append(arcsine_dist(y_interval[state][0], y_interval[state][1]))
    return res


# ---------------------------------------------------------------------------
# «Зависимое» распределение: y1 ~ сдвинутая гамма, y2 | y1 ~ наклонённая
# равномерная плотность.
# ---------------------------------------------------------------------------
def _gamma_d_1d(x, alpha, beta, shift):
    """Плотность Gamma(alpha, rate=beta), сдвинутого на shift: X = shift + Gamma."""
    res = np.zeros_like(x)
    mask = x >= shift
    s = x[mask] - shift
    res[mask] = np.exp(
        alpha * np.log(beta) + (alpha - 1) * np.log(s) - beta * s - math.lgamma(alpha)
    )
    return res


def _gamma_d_params(state):
    """Параметры маргинального закона y1 (гамма) по состоянию — как в config.py."""
    if state == 3:
        return 22, 1000.0
    return 10 + 5 * state, 2000.0


def _cond_d(x, a, b, x1):
    """Условная плотность y2 | y1=x1: наклонённая равномерная на [a, b]."""
    if (x < a) or (x > b):
        return 0.
    return 1 / (b - a) * (2 / np.pi * np.arctan(x1) * (x - (a + b) / 2) + 1)


def _pi_dependent_pdf_grid(state, M_net_n, y_intervals):
    """Совместная плотность (y1, y2) семейства dependent_gamma на сетке."""
    y1_interval, y2_interval = y_intervals
    alpha, beta = _gamma_d_params(state)
    shift = y1_interval[state][0]
    a2 = y2_interval[state][0]
    b2 = y2_interval[state][1]

    y1_vals = M_net_n[:, 0]
    y2_vals = M_net_n[:, 1]

    pdf_y1 = _gamma_d_1d(y1_vals, alpha, beta, shift)
    res = np.empty_like(pdf_y1)
    for i in range(res.shape[0]):
        res[i] = pdf_y1[i] * _cond_d(y2_vals[i], a2, b2, y1_vals[i])
    return res


@njit(fastmath=True)
def get_y_dependent(state, y_intervals):
    """Генератор Y для dependent_gamma, согласованный с _pi_dependent_pdf_grid."""
    if state == 3:
        shape = 22
        scale = 0.001
        loc = 0.022
    else:
        scale = 0.0005
        shape = 10 + 5 * state
        loc = shape * 0.001

    y1 = np.random.gamma(shape, scale) + loc

    a = y_intervals[1][state][0]
    b = y_intervals[1][state][1]
    c = (2. / np.pi) * np.arctan(y1)
    alpha = 0.5 * (1. + c * (b - a) / 2.)
    u = np.random.uniform()
    if u < alpha:
        y2 = np.random.triangular(a, b, b)
    else:
        y2 = np.random.triangular(a, a, b)

    return [y1, y2]


# ---------------------------------------------------------------------------
# 3-точечное распределение: масса в узлах сетки, ближайших к концам и
# середине интервала. Требует сами узлы (для argmin), поэтому не является
# разделимой поточечной функцией — не проходит через separable().
# ---------------------------------------------------------------------------
def _pi_3point_pdf_grid(state, M_net_n, y_intervals):
    tmp = []
    for j, y_interval in enumerate(y_intervals):
        a = y_interval[state][0]
        b = y_interval[state][1]
        net = np.unique(M_net_n[:, j])
        col = np.zeros(net.shape[0])
        for point in (a, (a + b) / 2, b):
            col[np.argmin(np.abs(net - point))] = 1.0
        tmp.append(col)
    return cartesian_product(tmp).prod(axis=1)


# ---------------------------------------------------------------------------
# Реестр семейств π(y|θ)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PiFamily:
    """Семейство π(y|θ): плотность на сетке (pdf_grid) + согласованный
    джитованный генератор (sampler); normalize_on_grid включает нормировку
    pdf по сетке."""
    name: str
    pdf_grid: Callable
    sampler: Callable
    normalize_on_grid: bool


def separable(pdf_1d):
    """Строит совместную плотность на сетке из одномерной pdf_1d(x, a, b)
    в предположении независимости координат Y."""
    def pdf_grid(state, M_net_n, y_intervals):
        res = np.ones(M_net_n.shape[0])
        for j, y_interval in enumerate(y_intervals):
            a = y_interval[state][0]
            b = y_interval[state][1]
            res = res * pdf_1d(M_net_n[:, j], a, b)
        return res
    return pdf_grid


_REGISTRY = {}


def register_pi_family(name_or_family):
    """Регистрация семейства: декоратор над builder() -> PiFamily либо
    прямой вызов с готовым объектом PiFamily."""
    if isinstance(name_or_family, PiFamily):
        _REGISTRY[name_or_family.name] = name_or_family
        return name_or_family

    name = name_or_family

    def decorator(builder):
        family = builder()
        _REGISTRY[name] = family
        return family

    return decorator


def get_pi_family(family):
    """Принимает имя из реестра либо готовый объект PiFamily."""
    if isinstance(family, PiFamily):
        return family
    try:
        return _REGISTRY[family]
    except KeyError:
        raise KeyError(
            f"неизвестное семейство π(y|θ): {family!r}; доступны: {sorted(_REGISTRY)}"
        )


register_pi_family(PiFamily(
    name='uniform',
    pdf_grid=separable(unif),
    sampler=get_y_uniform,
    normalize_on_grid=False,
))

register_pi_family(PiFamily(
    name='triangular',
    pdf_grid=separable(triang_1d),
    sampler=get_y_triangular,
    normalize_on_grid=True,
))

register_pi_family(PiFamily(
    name='triangular_right',
    pdf_grid=separable(triang_right_1d),
    sampler=get_y_triangular_2,
    normalize_on_grid=True,
))

register_pi_family(PiFamily(
    name='arcsine',
    pdf_grid=separable(arcsine_1d),
    sampler=get_y_arcsine,
    normalize_on_grid=True,
))

register_pi_family(PiFamily(
    name='three_point',
    pdf_grid=_pi_3point_pdf_grid,
    sampler=get_y_3point,
    normalize_on_grid=True,
))

register_pi_family(PiFamily(
    name='dependent_gamma',
    pdf_grid=_pi_dependent_pdf_grid,
    sampler=get_y_dependent,
    normalize_on_grid=False,
))


def build_pi(family, N, M_net, nets, y_intervals, delta):
    """Строит π(y|θ) формы (N, n_grid) на сетке для всех N состояний."""
    fam = get_pi_family(family)
    n_grid = M_net.shape[1]
    pi = np.empty((N, n_grid))
    for n in range(N):
        pdf = fam.pdf_grid(n, M_net[n], y_intervals)
        if fam.normalize_on_grid:
            pdf = pdf / (pdf.sum() * delta[n])
        pi[n] = pdf
    return pi
