"""Специализация переходного ядра для случая, когда все каналы NORMAL.

Считает ровно то же, что связка ``zero_jump_kernel`` / ``integrand`` /
``single_jump_kernel`` / ``integrand2`` / ``double_jump_kernel`` /
``filter_step`` из ``core.filter`` при ``obs_density`` со всеми каналами
NORMAL: та же квадратура средними прямоугольниками (method = 0), тот же
пропуск при ``pi[m, y] == 0``, та же обработка ``two_jumps`` (в том числе при
``n == m``). Отличие только в порядке суммирования (циклы по ``tau`` и ``v``
переставлены) и в том, что параметры гауссианы складываются по временам
пребывания напрямую, без буферов ``w``/``comp`` и косвенного вызова
``obs_density``.

Предполагается ``pi.shape == psi.shape == (N, n_grid)``.
"""

import math

import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:], nb.int64,
                    nb.float64),
         fastmath=True, inline='always')
def _normal_prod(obs, mu, var, K, two_pi_k):
    """prod_k N(obs[k]; mu[k], var[k]) за один exp и один sqrt."""
    q = 0.0
    z = two_pi_k
    for k in range(K):
        d = obs[k] - mu[k]
        q += d * d / var[k]
        z *= var[k]
    return math.exp(-0.5 * q) / math.sqrt(z)


@nb.njit(fastmath=True, parallel=True)
def filter_step_normal(psi, obs, C, Lambda, lam, pi, ht, delta, N, n_points,
                       two_jumps, obs_density):
    # obs_density принимается для совместимости сигнатуры с filter_step и
    # не используется: плотность зашита.
    n_grid = psi.shape[1]
    K = C.shape[2]
    two_pi_k = (2.0 * math.pi)**K
    step = ht / n_points
    shift = step / 2          # средние прямоугольники, как в filter_step

    res = np.zeros(psi.shape)

    for y in nb.prange(n_grid):
        mu = np.empty(K)
        var = np.empty(K)
        base_mu = np.empty(K)
        base_var = np.empty(K)
        mid_mu = np.empty(K)
        mid_var = np.empty(K)

        for m in range(N):
            # r = 0
            for k in range(K):
                mu[k] = ht * C[m, y, k, 0]
                var[k] = ht * C[m, y, k, 1]
            res[m, y] += (
                psi[m, y] * math.exp(ht * lam[m])
                * _normal_prod(obs, mu, var, K, two_pi_k)
            )

            # при pi[m, y] == 0 и integrand, и integrand2 дают ноль
            if pi[m, y] == 0.0:
                continue

            for n in range(N):
                if n != m:
                    # r = 1
                    w_nm = pi[m, y] * Lambda[n, m] * delta[n] * step
                    tau = shift
                    for _ in range(n_points):
                        rest = ht - tau
                        for k in range(K):
                            base_mu[k] = rest * C[m, y, k, 0]
                            base_var[k] = rest * C[m, y, k, 1]
                        w = w_nm * math.exp(
                            tau * Lambda[n, n] + rest * Lambda[m, m]
                        )
                        for v in range(n_grid):
                            pv = psi[n, v]
                            if pv == 0.0:
                                continue
                            for k in range(K):
                                mu[k] = tau * C[n, v, k, 0] + base_mu[k]
                                var[k] = tau * C[n, v, k, 1] + base_var[k]
                            res[m, y] += (
                                w * pv
                                * _normal_prod(obs, mu, var, K, two_pi_k)
                            )
                        tau += step

                if two_jumps:
                    # r = 2; delta[n] дважды -- ровно как в double_jump_kernel
                    # (интегрирование и по v, и по z весом delta[n])
                    for c in range(N):
                        if (c == m) or (c == n):
                            continue
                        w_c = (pi[m, y] * Lambda[n, c] * Lambda[c, m]
                               * delta[n] * delta[n] * step * step)
                        for i in range(n_points):
                            tau1 = step * i + shift
                            for j in range(n_points - i):
                                tau2 = step * j + shift
                                rest = ht - tau1 - tau2
                                w = w_c * math.exp(
                                    tau1 * Lambda[n, n]
                                    + tau2 * Lambda[c, c]
                                    + rest * Lambda[m, m]
                                )
                                for k in range(K):
                                    base_mu[k] = rest * C[m, y, k, 0]
                                    base_var[k] = rest * C[m, y, k, 1]
                                for z in range(n_grid):
                                    pz = pi[c, z]
                                    if pz == 0.0:
                                        continue
                                    for k in range(K):
                                        mid_mu[k] = (tau2 * C[c, z, k, 0]
                                                     + base_mu[k])
                                        mid_var[k] = (tau2 * C[c, z, k, 1]
                                                      + base_var[k])
                                    for v in range(n_grid):
                                        pv = psi[n, v]
                                        if pv == 0.0:
                                            continue
                                        for k in range(K):
                                            mu[k] = (tau1 * C[n, v, k, 0]
                                                     + mid_mu[k])
                                            var[k] = (tau1 * C[n, v, k, 1]
                                                      + mid_var[k])
                                        res[m, y] += (
                                            w * pz * pv
                                            * _normal_prod(obs, mu, var, K,
                                                           two_pi_k)
                                        )
    return res
