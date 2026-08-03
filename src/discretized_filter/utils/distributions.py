"""
Плотности π(y|θ) и парные им джитованные генераторы Y.

Каждое встроенное семейство описано объектом PiFamily: плотность на сетке
(pdf_grid) и согласованный с ней генератор (sampler) — джитованная функция
с той же математической моделью. Реестр (register_pi_family/get_pi_family)
позволяет конфигу выбрать семейство по имени либо передать произвольный
объект PiFamily («произвольная плотность»), поэтому набор имён не жёстко
зашит нигде, кроме самой регистрации встроенных семейств.

build_pi(...) — точка входа для конфига, заменяет старый get_distributions.
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
# Генераторы Y по θ (перенесены из core/smjp.py + доведённые до конца
# закомментированные варианты)
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
# равномерная плотность. Перенесено из config.py (gamma_d, cond_d) и
# filtering.ipynb (get_y_dependent) — до переноса плотность и генератор
# жили в разных файлах и, как выяснилось, не совпадали (см. отчёт).
# ---------------------------------------------------------------------------
def _gamma_d_1d(x, alpha, beta, shift):
    """
    Плотность гамма-распределения Gamma(alpha, rate=beta), сдвинутого на
    shift: X = shift + Gamma(alpha, rate=beta). math.lgamma вместо
    scipy.special.gammaln — не тащим scipy в горячий путь.

    ИСПРАВЛЕНО при переносе: в config.py `s = x[x >= shift]` не вычитал
    shift из x, из-за чего плотность фактически была плотностью
    НЕсдвинутой Gamma(alpha, beta), обрезанной по порогу x >= shift, — это
    описывает совершенно другое (и для параметров конфига практически
    нереализуемое) распределение, не совпадающее с генератором. Здесь
    используется s = x - shift, что соответствует генератору.
    """
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
    """
    Условная плотность y2 при заданном y1 = x1 (перенесено из config.py:
    cond_d). Линейно «наклонённая» равномерная плотность на [a, b] с
    наклоном, зависящим от arctan(x1).
    """
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
    """
    Генератор Y для dependent_gamma: y1 = shift + Gamma(shape, scale=1/beta);
    y2 | y1 — смесь двух треугольных распределений (мода в a либо в b) с
    весом alpha, наклонённым в сторону b пропорционально arctan(y1).

    ИСПРАВЛЕНО при переносе (см. отчёт): в filtering.ipynb коэффициент
    наклона был c = (4/pi)*arctan(y1), что вдвое сильнее наклона,
    заданного плотностью cond_d (там коэффициент (2/pi)*arctan(y1)).
    Плотность взята авторитетной — здесь c = (2/pi)*arctan(y1).
    """
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
    """
    Семейство условной плотности π(y|θ): плотность на сетке + согласованный
    с ней джитованный генератор.

    pdf_grid(state, M_net_n, y_intervals) -> np.ndarray (n_grid,)
        ненормированная совместная плотность на сетке M_net_n = M_net[state]
        формы (n_grid, M).
    sampler(state, y_intervals) -> list[float]
        джитованный генератор M координат Y, распределённых согласно
        pdf_grid.
    normalize_on_grid : bool
        True  — нормировать по сетке: pdf /= pdf.sum() * delta[state];
        False — доверять аналитической нормировке pdf_grid (как у uniform).
    """
    name: str
    pdf_grid: Callable
    sampler: Callable
    normalize_on_grid: bool


def separable(pdf_1d):
    """
    Строит совместную ненормированную плотность на сетке из одномерной
    плотности pdf_1d(x, a, b) в предположении независимости координат Y.
    """
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
    """
    Регистрация семейства. Можно вызвать как декоратор над функцией без
    аргументов, строящей PiFamily:

        @register_pi_family('my_family')
        def _build():
            return PiFamily(...)

    либо передать готовый объект: register_pi_family(family).
    """
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
    """
    Строит π(y|θ) на сетке для всех N состояний. Заменяет старый
    get_distributions(N, M_net, nets, y_intervals, deltas).

    family : имя семейства из реестра либо объект PiFamily.
    M_net  : (N, n_grid, M) — совместная сетка по состояниям.
    nets   : список одномерных сеток по состояниям (передаётся не всем
             семействам — большинству достаточно M_net[n] и y_intervals).
    delta  : (N,) — шаг интегрирования по сетке (произведение шагов по
             координатам) для каждого состояния.

    Возвращает pi формы (N, n_grid).
    """
    fam = get_pi_family(family)
    n_grid = M_net.shape[1]
    pi = np.empty((N, n_grid))
    for n in range(N):
        pdf = fam.pdf_grid(n, M_net[n], y_intervals)
        if fam.normalize_on_grid:
            pdf = pdf / (pdf.sum() * delta[n])
        pi[n] = pdf
    return pi
