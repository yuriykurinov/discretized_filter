import numpy as np
import numba as nb

from utils import norm


@nb.njit(
    nb.float64(
        nb.uintp, nb.uintp, nb.float64, nb.float64,
        nb.float64[:, :], nb.float64[:, :], nb.float64[:]
    ),
    fastmath=True,
    inline='always',
)
def zero_jump_kernel(m, y, obs, h, F, G, lam):
    sigma_sq_2 = 2 * h * G[m, y]
    log_res = (h / lam[m] - 0.5 * np.log(np.pi * sigma_sq_2))\
            - (h * F[m, y] - obs)**2 / sigma_sq_2
    return np.exp(log_res)
    # return np.exp(
    #     h / lam[m] - (h * F[m, y] - obs) ** 2 / sigma_sq_2
    # ) / np.sqrt(np.pi*sigma_sq_2)


@nb.njit(
    nb.float64(
        nb.float64, nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64, nb.float64[:, :], nb.float64[:, :],
        nb.float64[:, :], nb.float64[:, :], nb.float64
    ),
    fastmath=True,
    #inline='always',
)
def integrand(tau, m, y, n, v, obs, pi, F, G, Lambda, h):
    if pi[m, y] == 0:
        return 0
    else:
        return (
            pi[m, y]
            * Lambda[n, m] / Lambda[n, n]**2
            * np.exp(h / Lambda[m, m])
            * norm(
                obs,
                tau * F[n, v] + (h - tau) * F[m, y],
                tau * G[n, v] + (h - tau) * G[m, y],
            )
            * (1 - np.exp(tau / Lambda[n, n]))
            * np.exp(tau * (1 / Lambda[n, n] - 1 / Lambda[m, m]))
        )


@nb.njit(
    nb.float64(
        nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64, nb.float64,
        nb.float64[:, :], nb.float64[:, :],
        nb.float64[:, :], nb.float64[:, :],
        nb.uintp, nb.uintp
    ),
    #inline='always',
    fastmath=True,
)
def single_jump_kernel(m, y, n, v, obs, h, F, G, Lambda, pi, method, n_points):
    #(m, y) -> (n, v)
    res = 0.0
    step = h / n_points
    tau = 0.0

    if method == 0:  # mid rectangular
        tau = step / 2
    elif method == 1:  # left rectangular
        tau = 0
    elif method == 2:  # right rectangular
        tau = step

    for _ in range(n_points):
        res += integrand(tau, m, y, n, v, obs, pi, F, G, Lambda, h) * step
        tau += step
    return res


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :],
        nb.float64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64,
        nb.float64,
        nb.uintp,
        nb.uintp,
    ),
    fastmath=True,
    parallel=True,
)
def filter_step(psi, obs, F, G, Lambda, lam, pi, h, delta, N, n_points):
    res = np.zeros(psi.shape)

    for y in nb.prange(psi.shape[1]):
        for m in range(N):
            res[m, y] += psi[m, y] * zero_jump_kernel(m, y, obs, h, F, G, lam)
            for n in range(N):
                if m != n:
                    for v in range(psi.shape[1]):
                        res[m, y] += (
                            single_jump_kernel(
                                m, y, n, v, obs, h, F, G, Lambda, pi,
                                0, n_points  # mid-rectangular method = 0
                            )
                            * psi[n, v]
                            * delta
                        )
    return res


class Filter(object):
    def __init__(
        self, pi_init, pi, M_net, g, sigma, 
        N, Lambda, ht, delta, n_points=3,
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
        self.F = np.repeat(g(-1, M_net, -1)[np.newaxis, ...], N, axis=0)
        self.G = np.repeat((sigma(-1, M_net, -1)**2)[np.newaxis, ...], N, axis=0)

        self.psi = pi_init.copy()
    
    def update(self, obs):
        new_psi = self.filter_step(
            self.psi, obs, self.F, self.G, 
            self.Lambda, self.lam, self.pi,
            self.ht, self.delta, self.N, self.n_points,
        )

        normalizer = new_psi.sum() * self.delta
        if normalizer != 0:
            self.psi = new_psi / normalizer
        # else:
        #     raise ZeroDivisionError('psi fell to zero')
    
    def estimate(self):
        theta_est = self.psi.sum(axis=1)
        theta_est = theta_est / theta_est.sum()
        y_est = self.psi.sum(axis=0) @ self.M_net * self.delta
        return theta_est, y_est

