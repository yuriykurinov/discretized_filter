"""Ядра переходного оператора и класс фильтра.

Формулы (transition_kernel:1-2) из ``article/my_notes/discretized_filter.tex``
с усечением ряда по числу скачков на шаге длины ``ht``: r = 0, r = 1 и
(опционально) r = 2; веса времён пребывания -- из представления (repr:rho).

Соглашения:
  * ``C[n, y, k, p]`` -- p-й параметр плотности k-го канала в состоянии
    ``(theta = n, Y = y)``;
  * ``obs_density(obs[K], u[R], comp[R, K, P], n_comp) -> float64`` --
    джитованная совместная плотность приращения (см. ``core.densities``):
    ядро передаёт времена пребывания ``u`` и параметры компонент ``comp``, а
    способ их свёртки (линейное накопление или смесь) выбирает сама плотность;
  * ``Lambda`` -- генератор цепи, строки суммируются в ноль,
    ``lam[m] = Lambda[m, m] <= 0``; ``Lambda[n, m]`` -- интенсивность перехода
    n -> m;
  * ``psi``, ``pi`` -- ненормированная и априорная плотности по сетке ``y``;
    интегрирование по ``y`` -- сумма с весом ``delta[n]``;
  * ``update(obs)`` применяет переход и обновление по наблюдению
    одновременно (ядро уже содержит плотность приращения), затем нормирует.

Буферы ``w`` формы ``(3,)`` и ``comp`` формы ``(3, K, P)`` выделяются один раз
на итерацию ``nb.prange`` (приватны для потока) и прокидываются во все ядра,
чтобы во внутренних циклах не было аллокаций.
"""

import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def zero_jump_kernel(m, y, obs, ht, C, lam, obs_density, w, comp):
    """Слагаемое r = 0: всё время ``ht`` проведено в ``(m, y)``."""
    w[0] = ht
    for k in range(comp.shape[1]):
        for p in range(comp.shape[2]):
            comp[0, k, p] = C[m, y, k, p]
    # вес времени пребывания -- ровно один множитель, а не K штук
    return np.exp(ht * lam[m]) * obs_density(obs, w, comp, 1)


@nb.njit(fastmath=True)
def integrand(tau, m, y, n, v, obs, pi, C, Lambda, ht, obs_density, w, comp):
    """Подынтегральное выражение r = 1: ``tau`` в ``(n, v)``, ``ht - tau`` в ``(m, y)``."""
    if pi[m, y] == 0:
        return 0.0
    else:
        w[0] = tau
        w[1] = ht - tau
        for k in range(comp.shape[1]):
            for p in range(comp.shape[2]):
                comp[0, k, p] = C[n, v, k, p]
                comp[1, k, p] = C[m, y, k, p]
        return (
            pi[m, y]
            * Lambda[n, m]
            * obs_density(obs, w, comp, 2)
            * np.exp(tau * Lambda[n, n] + (ht - tau) * Lambda[m, m])
        )


@nb.njit(fastmath=True)
def single_jump_kernel(
    m, y, n, v, obs, ht, C, Lambda, pi, method, n_points, obs_density, w, comp
):
    """Слагаемое r = 1: интеграл по времени скачка методом прямоугольников."""
    res = 0.0
    step = ht / n_points

    if method == 1:  # left rectangular
        tau = 0.0
    elif method == 2:  # right rectangular
        tau = step
    else:  # method == 0, mid rectangular
        tau = step / 2

    for _ in range(n_points):
        res += integrand(
            tau, m, y, n, v, obs, pi, C, Lambda, ht, obs_density, w, comp
        ) * step
        tau += step
    return res


@nb.njit(fastmath=True)
def integrand2(
    tau1, tau2, m, y, k, z, n, v, obs, pi, C, Lambda, ht, obs_density, w, comp
):
    """Подынтегральное выражение r = 2: ``tau1`` в ``(n, v)``, ``tau2`` в ``(k, z)``, остаток в ``(m, y)``."""
    if pi[m, y] == 0:
        return 0.0
    else:
        w[0] = tau1
        w[1] = tau2
        w[2] = ht - tau1 - tau2
        for kk in range(comp.shape[1]):
            for p in range(comp.shape[2]):
                comp[0, kk, p] = C[n, v, kk, p]
                comp[1, kk, p] = C[k, z, kk, p]
                comp[2, kk, p] = C[m, y, kk, p]
        return (
            pi[m, y] * pi[k, z]
            * Lambda[n, k] * Lambda[k, m]
            * obs_density(obs, w, comp, 3)
            * np.exp(
                tau1 * Lambda[n, n]
                + tau2 * Lambda[k, k]
                + (ht - tau1 - tau2) * Lambda[m, m]
            )
        )


@nb.njit(fastmath=True)
def double_jump_kernel(
    m, y, n, v, obs, ht, C, Lambda, pi, method, n_points, delta,
    obs_density, w, comp
):
    """Слагаемое r = 2: сумма по промежуточному состоянию и двойной интеграл
    по симплексу tau1 + tau2 <= ht."""
    res = 0.0

    step = ht / n_points

    if method == 1:  # left rectangular
        shift = 0.0
    elif method == 2:  # right rectangular
        shift = step
    else:  # method == 0, mid rectangular
        shift = step / 2

    for k in range(pi.shape[0]):
        if (m == k) or (k == n):
            continue
        for z in range(pi.shape[1]):
            for i in range(n_points):
                for j in range(n_points-i):
                    res += integrand2(
                        step*i + shift,
                        step*j + shift,
                        m, y,
                        k, z,
                        n, v,
                        obs,
                        pi,
                        C,
                        Lambda,
                        ht,
                        obs_density,
                        w,
                        comp
                    ) * step * step * delta
    return res


@nb.njit(fastmath=True, parallel=True)
def filter_step(
    psi, obs, C, Lambda, lam, pi, ht, delta, N, n_points, two_jumps,
    obs_density
):
    """Один шаг рекурсии для ненормированной условной плотности ``res[m, y]``."""
    res = np.zeros(psi.shape)
    n_channels = C.shape[2]
    n_slots = C.shape[3]

    for y in nb.prange(psi.shape[1]):
        # буферы времён пребывания и параметров компонент, приватные для потока
        w = np.empty(3)
        comp = np.empty((3, n_channels, n_slots))
        for m in range(N):
            res[m, y] += psi[m, y] * zero_jump_kernel(
                m, y, obs, ht, C, lam, obs_density, w, comp
            )
            for n in range(N):
                if m != n:
                    for v in range(psi.shape[1]):
                        res[m, y] += (
                            single_jump_kernel(
                                m, y, n, v, obs, ht, C, Lambda, pi,
                                0, n_points,  # mid-rectangular method = 0
                                obs_density, w, comp
                            )
                            * psi[n, v]
                            * delta[n]
                        )
                if two_jumps:
                    for v in range(psi.shape[1]):
                        res[m, y] += (
                            double_jump_kernel(
                                m, y, n, v, obs, ht, C, Lambda, pi,
                                0, n_points, delta[n],  # mid-rectangular
                                obs_density, w, comp
                            )
                            * psi[n, v]
                            * delta[n]
                        )
    return res


class Filter(object):
    """Дискретизированный фильтр для скрытой марковской цепи со сносом.

    ``C`` -- ``float64[N, n_grid, K, P]``, параметры плотности приращения
    наблюдений по состояниям; ``obs_density`` -- джитованная функция
    ``(obs[K], u[R], comp[R, K, P], n_comp) -> float64``.
    """

    def __init__(
        self, pi_init, pi, M_net, C,
        N, Lambda, ht, delta, obs_density, n_points=3,
        two_jumps=False,
        filter_step=filter_step
    ):
        self.pi = pi.copy()
        self.Lambda = Lambda.copy()
        self.lam = np.diagonal(Lambda).copy()
        self.M_net = M_net.copy()
        self.ht = ht
        self.N = N
        self.n_points = n_points
        self.delta = delta
        self.filter_step = filter_step

        self.two_jumps = two_jumps

        self.C = C
        self.obs_density = obs_density

        self.psi = pi_init.copy()

    def update(self, obs):
        new_psi = self.filter_step(
            self.psi, obs, self.C,
            self.Lambda, self.lam, self.pi,
            self.ht, self.delta, self.N, self.n_points,
            self.two_jumps, self.obs_density
        )

        normalizer = (self.delta[:, np.newaxis] * new_psi).sum()
        # NaN != 0 истинно, поэтому проверка на ноль сама по себе NaN не ловит:
        # без явной проверки на конечность psi молча стала бы полностью NaN.
        if not np.isfinite(normalizer):
            raise FloatingPointError(
                f'нормировочный множитель не конечен ({normalizer}): '
                'скорее всего нулевая дисперсия или переполнение в плотности '
                'наблюдений'
            )
        if normalizer == 0:
            raise ZeroDivisionError('psi fell to zero')
        self.psi = new_psi / normalizer

    def estimate(self):
        theta_est = self.psi.sum(axis=1) * self.delta
        theta_est = theta_est / theta_est.sum()
        y_est = np.zeros(self.M_net.shape[2])
        for n in range(self.N):
            y_est += self.psi[n] @ self.M_net[n] * self.delta[n]

        #self.psi.sum(axis=0) @ self.M_net * self.delta
        return theta_est, y_est
