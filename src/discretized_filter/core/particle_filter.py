"""Бутстрэп-фильтр частиц для той же системы наблюдения, что и ``core.filter``.

Реализуются формулы (3.1)-(3.2) из ``article/my_notes/discretized_filter.tex``
(статья Борисова--Куринова): при фиксированной траектории сигнала
``Z_t = (theta_t, Y_t)`` на шаге ``[t_{k-1}, t_k]`` длины ``ht`` приращение
наблюдения условно гауссовское,

    d_xi ~ N(F, G),
    F = sum_i d_alpha_i * f(theta_i, Y_i),
    G = sum_i d_alpha_i * (g g^T)(theta_i, Y_i)   (диагональ),

где ``d_alpha_i`` -- времена пребывания в состояниях траектории внутри шага.
``f`` и ``gg^T`` -- те же ``ch.drift`` и ``ch.var``, что ``config.py`` кладёт в
``C[..., k, 0]`` и ``C[..., k, 1]``.

Соглашения:
  * только непрерывные наблюдения с гауссовскими шумами (все каналы
    ``kind == NORMAL``); каналы независимы, ковариация диагональна, поэтому
    плотность приращения -- произведение по каналам (логарифмический аналог
    ``core.filter_normal._normal_prod``);
  * предложение (proposal) -- априорная динамика: траектория ``theta`` внутри
    шага разыгрывается ТОЧНО так же, как в ``core.smjp.sparse_mc``
    (экспоненциальные времена, выбор нового состояния по ``Lambda``, метка
    ``Y`` из ``get_y``); число скачков внутри шага не ограничивается -- в
    отличие от сеточного фильтра здесь нет обрыва ряда по числу скачков и нет
    квадратур по ``y`` и по времени скачка;
  * веса ведутся в логарифмах: рабочие дисперсии порядка 1e-8, и линейная
    плотность приращения легко выходит за диапазон float64;
  * ресемплинг ленивый -- в НАЧАЛЕ ``update``, а не в конце, чтобы
    ``estimate()`` между шагами всегда работала со свежими (неравными)
    весами: такая оценка имеет меньшую дисперсию;
  * ``update(obs)``, как и ``Filter.update``, делает переход и обновление по
    наблюдению одновременно (правдоподобие (3.2) уже содержит траекторию
    сигнала на всём шаге), затем нормирует веса;
  * положительность дисперсий -- ответственность вызывающего (как и в
    ``core.filter_normal``, где ``_normal_prod`` не защищена от нулевой
    дисперсии); неконечные веса ловятся при нормировке.

Цикл по частицам последовательный, без ``nb.prange``: у джитованного ГПСЧ
numba состояние потоко-локальное, и с ``parallel=True`` результат перестал бы
воспроизводиться по семени. Это осознанный размен скорости на
воспроизводимость.

Внимание: состояние ГПСЧ numba -- единое на весь процесс, не привязано к
экземпляру ``ParticleFilter``. При сравнении нескольких фильтров на одной
траектории сигнала прогоняйте их последовательно (каждый до конца), пересевая
``set_seed`` перед каждым; чередование вызовов ``update`` разных фильтров
приведёт к расхождению на первом же ресемплинге.
"""

import math

import numpy as np
import numba as nb

from discretized_filter.core.densities import NORMAL
from discretized_filter.utils.grids import set_seed


def make_normal_params(channels):
    """Собирает джитованную ``params(y[1, M], theta, out[K, 2], k0)``.

    Заполняет ``out[k0 + k, 0] = f_k(y)`` и ``out[k0 + k, 1] = (g g^T)_k(y)``
    по всем каналам. Каналы -- питоновский список разнородных джитованных
    функций, вызвать их изнутри njit-ядра циклом нельзя, поэтому список
    разворачивается в одну функцию рекурсивным замыканием.

    Соглашение о вызове ``drift(-1, y, theta)`` с ``t = -1`` и двумерным ``y``
    формы ``(1, M)`` -- ровно то же, что в ``config.py::_build`` при сборке
    ``C`` (там ``ch.drift(-1, M_net[n], -1)``); ``theta`` передаётся
    настоящий, хотя все конфиги репозитория его игнорируют.
    """
    if len(channels) == 0:
        @nb.njit(fastmath=True)
        def base(y, theta, out, k0):
            pass
        return base

    drift = channels[0].drift
    var = channels[0].var
    rest = make_normal_params(channels[1:])

    @nb.njit(fastmath=True)
    def step(y, theta, out, k0):
        out[k0, 0] = drift(-1, y, theta)[0, 0]
        out[k0, 1] = var(-1, y, theta)[0, 0]
        rest(y, theta, out, k0 + 1)

    return step


@nb.njit(fastmath=True)
def particle_step(theta, Y, log_w, obs, Lambda, lam, cum_p, ht,
                  get_y, y_intervals, params, K):
    """Распространение частиц на шаг ``ht`` и добавка логарифма правдоподобия
    (3.2) к ``log_w``.

    ``theta[n_particles]``, ``Y[n_particles, M]`` и ``log_w[n_particles]``
    меняются на месте: после вызова ``(theta[i], Y[i])`` -- значение сигнала в
    правом конце шага.

    ``cum_p[n, :]`` -- кумулятивные вероятности перехода
    ``Lambda[n, m] / (-Lambda[n, n])`` с занулённой диагональю;
    ``params`` -- результат ``make_normal_params``.
    """
    n_particles = theta.shape[0]
    M = Y.shape[1]
    N = Lambda.shape[0]
    two_pi = 2.0 * math.pi

    # буферы -- по одному на вызов ядра, вне цикла по частицам
    y_buf = np.empty((1, M))
    out = np.empty((K, 2))
    F = np.empty(K)
    G = np.empty(K)

    # ПОСЛЕДОВАТЕЛЬНЫЙ цикл: см. замечание о ГПСЧ в докстринге модуля
    for i in range(n_particles):
        state = theta[i]
        for d in range(M):
            y_buf[0, d] = Y[i, d]
        for k in range(K):
            F[k] = 0.0
            G[k] = 0.0

        tau_ost = ht
        while True:
            params(y_buf, state, out, 0)
            # та же параметризация экспоненциального времени, что в sparse_mc
            dt = np.random.exponential(-1.0 / lam[state])
            if dt >= tau_ost:
                for k in range(K):
                    F[k] += tau_ost * out[k, 0]
                    G[k] += tau_ost * out[k, 1]
                break
            for k in range(K):
                F[k] += dt * out[k, 0]
                G[k] += dt * out[k, 1]
            tau_ost -= dt

            # новое состояние m != state с вероятностью Lambda[state, m] / -lam
            u = np.random.random()
            m = 0
            while m < N - 1 and u > cum_p[state, m]:
                m += 1
            state = m
            y_new = get_y(state, y_intervals)
            for d in range(M):
                y_buf[0, d] = y_new[d]

        theta[i] = state
        for d in range(M):
            Y[i, d] = y_buf[0, d]

        # log N(obs; F, G) для диагональной G -- сумма по каналам
        acc = 0.0
        for k in range(K):
            diff = obs[k] - F[k]
            acc += diff * diff / G[k] + math.log(two_pi * G[k])
        log_w[i] -= 0.5 * acc


@nb.njit(fastmath=True)
def systematic_resample(w, theta, Y):
    """Систематический ресемплинг: один ``u ~ U[0, 1/n)``, позиции
    ``u + j / n``, ``j = 0..n-1``.

    Возвращает новые ``(theta, Y)``; веса после ресемплинга полагаются
    равными вызывающим. Аллокации -- две, до цикла.
    """
    n = w.shape[0]
    M = Y.shape[1]
    theta_new = np.empty_like(theta)
    Y_new = np.empty_like(Y)

    u0 = np.random.random() / n
    i = 0
    c = w[0]
    for j in range(n):
        u = u0 + j / n
        # i < n - 1 страхует от выхода за границу при накоплении ошибки в c
        while u > c and i < n - 1:
            i += 1
            c += w[i]
        theta_new[j] = theta[i]
        for d in range(M):
            Y_new[j, d] = Y[i, d]
    return theta_new, Y_new


@nb.njit(fastmath=True)
def init_particles(p0, n_particles, get_y, y_intervals, M):
    """Начальные частицы: ``theta^i ~ p0``, ``Y^i ~ pi^{theta^i}``.

    Это ровно ``pi_init = p0[:, None] * pi`` сеточного фильтра, записанное
    эмпирической мерой. Функция джитована, чтобы пользоваться тем же ГПСЧ
    numba, что и ``particle_step``.
    """
    N = p0.shape[0]
    cum = np.cumsum(p0)
    theta = np.empty(n_particles, dtype=np.int64)
    Y = np.empty((n_particles, M))
    for i in range(n_particles):
        # выбор состояния -- как в smjp.choice
        state = np.searchsorted(cum, np.random.uniform())
        if state >= N:  # страховка от ошибки округления в cum[-1]
            state = N - 1
        theta[i] = state
        y = get_y(state, y_intervals)
        for d in range(M):
            Y[i, d] = y[d]
    return theta, Y


class ParticleFilter(object):
    """Бутстрэп-фильтр частиц для скрытой марковской цепи с метками.

    Оценки те же, что у ``Filter.estimate``, но по эмпирической мере:
    ``theta_est[n] = sum_{i: theta^i = n} w^i``, ``y_est = sum_i w^i Y^i``.
    При ``n_particles -> inf`` сходится к точному решению (3.4)-(3.5): без
    ошибки усечения ряда по числу скачков и без ошибки квадратур, но со
    статистической погрешностью порядка ``1 / sqrt(n_particles)``.
    """

    def __init__(self, p0, Lambda, get_y, y_intervals, channels, ht,
                 n_particles=10_000, resample_threshold=0.5, seed=None):
        if not all(int(ch.kind) == NORMAL for ch in channels):
            raise ValueError(
                'ParticleFilter реализован только для непрерывных '
                'наблюдений с гауссовскими шумами (все каналы NORMAL)'
            )

        if seed is not None:
            # у джитованного ГПСЧ numba (им пользуются ядра) состояние своё,
            # отдельное от глобального RandomState numpy
            set_seed(seed)
            np.random.seed(seed)

        self.Lambda = np.ascontiguousarray(Lambda, dtype=np.float64)
        self.lam = np.ascontiguousarray(np.diagonal(self.Lambda))
        if np.any(self.lam >= 0.0):
            raise ValueError(
                'ParticleFilter: у генератора Lambda есть поглощающее '
                'состояние (Lambda[n, n] == 0) либо неверный знак диагонали; '
                'время до скачка не определено'
            )

        self.N = self.Lambda.shape[0]
        self.M = len(y_intervals)
        self.K = len(channels)
        self.ht = float(ht)
        self.n_particles = int(n_particles)
        self.resample_threshold = float(resample_threshold)

        p0 = np.ascontiguousarray(p0, dtype=np.float64).ravel()
        self.p0 = p0 / p0.sum()

        # кумулятивные вероятности перехода n -> m, диагональ занулена
        jump = self.Lambda.copy()
        np.fill_diagonal(jump, 0.0)
        self.cum_p = np.ascontiguousarray(
            np.cumsum(jump / (-self.lam[:, np.newaxis]), axis=1)
        )

        self.get_y = get_y
        self.y_intervals = list(y_intervals)
        self.channels = list(channels)
        self.params = make_normal_params(self.channels)

        self.theta, self.Y = init_particles(
            self.p0, self.n_particles, self.get_y, self.y_intervals, self.M
        )
        self.log_w = np.full(self.n_particles, -math.log(self.n_particles))
        self.w = np.full(self.n_particles, 1.0 / self.n_particles)
        self.ess = float(self.n_particles)
        self.log_evidence = 0.0

    @classmethod
    def from_config(cls, cfg, n_particles=10_000, resample_threshold=0.5,
                    seed=None):
        """Фильтр по активному конфигу (``config.set_config``).

        ``seed=None`` означает НЕ трогать ГПСЧ: ноутбук сеет один раз в
        ``set_config``, и повторное семя посреди эксперимента сбило бы
        генерацию траектории сигнала и наблюдений. Передавайте ``seed``
        явно, только если хотите переустановить ГПСЧ прямо здесь.
        """
        return cls(
            cfg.p0, cfg.Lambda, cfg.get_y, cfg.y_intervals, cfg.channels,
            cfg.ht, n_particles=n_particles,
            resample_threshold=resample_threshold, seed=seed,
        )

    def update(self, obs):
        """Шаг фильтра по приращению наблюдения ``obs[K]``."""
        obs = np.ascontiguousarray(obs, dtype=np.float64).ravel()
        if obs.shape[0] != self.K:
            raise ValueError(
                f'ожидалось приращение размерности {self.K}, '
                f'получено {obs.shape[0]}'
            )

        # ленивый ресемплинг: в начале шага, по весам предыдущего шага
        if self.ess < self.resample_threshold * self.n_particles:
            self.theta, self.Y = systematic_resample(
                self.w, self.theta, self.Y
            )
            self.log_w[:] = -math.log(self.n_particles)

        particle_step(
            self.theta, self.Y, self.log_w, obs, self.Lambda, self.lam,
            self.cum_p, self.ht, self.get_y, self.y_intervals, self.params,
            self.K
        )

        # нормировка через logsumexp: вычитаем максимум, exp не переполняется
        m = self.log_w.max()
        # NaN != 0 истинно, поэтому проверка на ноль сама по себе NaN не
        # ловит: без явной проверки на конечность веса молча стали бы NaN.
        if not np.isfinite(m):
            raise FloatingPointError(
                f'максимальный логарифм веса не конечен ({m}): скорее всего '
                'нулевая дисперсия или переполнение в плотности наблюдений'
            )
        w = np.exp(self.log_w - m)
        s = w.sum()
        if s == 0:
            raise ZeroDivisionError('вырождение весов частиц')

        self.log_w -= m + math.log(s)
        # приращение логарифма нормировочной константы наблюдений:
        # logsumexp(log_w) - logsumexp(log_w_пред), причём предыдущий
        # logsumexp равен нулю (веса нормированы)
        self.log_evidence += m + math.log(s)

        self.w = w / s
        self.ess = 1.0 / (self.w**2).sum()

    def estimate(self):
        """Оценки ``(theta_est[N], y_est[M])`` -- формы как у ``Filter.estimate``."""
        theta_est = np.bincount(self.theta, weights=self.w, minlength=self.N)
        y_est = self.w @ self.Y
        return theta_est, y_est
