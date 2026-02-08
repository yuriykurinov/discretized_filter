import numpy as np
import numba as nb

from utils import norm


@nb.njit(
    nb.float64(
        nb.uintp, nb.uintp, nb.float64[:], nb.float64,
        nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:]
    ),
    fastmath=True,
    inline='always',
)
def zero_jump_kernel(m, y, obs, ht, F, G, lam):
    sigma_sq_2 = 2 * ht * G[m, y]
    log_res = (ht * lam[m] - 0.5 * np.log(np.pi * sigma_sq_2))\
            - (ht * F[m, y] - obs)**2 / sigma_sq_2
    return np.exp(np.sum(log_res))
    # return np.exp(
    #     ht / lam[m] - (ht * F[m, y] - obs) ** 2 / sigma_sq_2
    # ) / np.sqrt(np.pi*sigma_sq_2)


@nb.njit(
    nb.float64(
        nb.float64, nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64[:], nb.float64[:, :], nb.float64[:, :, :],
        nb.float64[:, :, :], nb.float64[:, :], nb.float64
    ),
    fastmath=True,
    #inline='always',
)
def integrand(tau, m, y, n, v, obs, pi, F, G, Lambda, ht):
    if pi[m, y] == 0:
        return 0
    else:
        return (
            pi[m, y]
            * Lambda[n, m] 
            * np.prod(
                norm(
                    obs,
                    tau * F[n, v] + (ht - tau) * F[m, y],
                    tau * G[n, v] + (ht - tau) * G[m, y],
                )
            )
            * np.exp(tau * Lambda[n, n] + (ht - tau) * Lambda[m, m])
        )


@nb.njit(
    nb.float64(
        nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64[:], nb.float64,
        nb.float64[:, :, :], nb.float64[:, :, :],
        nb.float64[:, :], nb.float64[:, :],
        nb.uintp, nb.uintp
    ),
    #inline='always',
    fastmath=True,
)
def single_jump_kernel(m, y, n, v, obs, ht, F, G, Lambda, pi, method, n_points):
    res = 0.0
    step = ht / n_points

    if method == 0:  # mid rectangular
        tau = step / 2
    elif method == 1:  # left rectangular
        tau = 0.0
    elif method == 2:  # right rectangular
        tau = step

    for _ in range(n_points):
        res += integrand(tau, m, y, n, v, obs, pi, F, G, Lambda, ht) * step
        tau += step
    return res

@nb.njit(
    nb.float64(
        nb.float64, nb.float64, 
        nb.uintp, nb.uintp, nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64[:], nb.float64[:, :], nb.float64[:, :, :],
        nb.float64[:, :, :], nb.float64[:, :], nb.float64
    ),
    fastmath=True,
    #inline='always',
)
def integrand2(tau1, tau2, m, y, k, z, n, v, obs, pi, F, G, Lambda, ht):
    if pi[m, y] == 0:
        return 0
    else:
        return (
            pi[m, y] * pi[k, z] 
            * Lambda[n, k] * Lambda[k, m] 
            * np.prod(
                norm(
                    obs,
                    tau1 * F[n, v] + tau2 * F[k, z] + (ht - tau1 - tau2) * F[m, y],
                    tau1 * G[n, v] + tau2 * G[k, z] + (ht - tau1 - tau2) * G[m, y],
                )
            )
            * np.exp(tau1 * Lambda[n, n] + tau2 * Lambda[k, k] + (ht - tau1 - tau2) * Lambda[m, m])
        )

@nb.njit(
    nb.float64(
        nb.uintp, nb.uintp, nb.uintp, nb.uintp,
        nb.float64[:], nb.float64,
        nb.float64[:, :, :], nb.float64[:, :, :],
        nb.float64[:, :], nb.float64[:, :],
        nb.uintp, nb.uintp, nb.float64
    ),
    #inline='always',
    fastmath=True,
)
def double_jump_kernel(m, y, n, v, obs, ht, F, G, Lambda, pi, method, n_points, delta):
    res = 0.0

    step = ht / n_points

    if method == 0:  # mid rectangular
        shift = step / 2
    elif method == 1:  # left rectangular
        shift = 0.0
    elif method == 2:  # right rectangular
        shift = step

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
                        F, G, 
                        Lambda, 
                        ht
                    ) * step * step * delta
    return res


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64,
        nb.float64,
        nb.uintp,
        nb.uintp,
        nb.bool
    ),
    fastmath=True,
    parallel=True,
)
def filter_step(psi, obs, F, G, Lambda, lam, pi, ht, delta, N, n_points, two_jumps=False):
    res = np.zeros(psi.shape)

    for y in nb.prange(psi.shape[1]):
        for m in range(N):
            res[m, y] += psi[m, y] * zero_jump_kernel(m, y, obs, ht, F, G, lam)
            for n in range(N):
                if m != n:
                    for v in range(psi.shape[1]):
                        res[m, y] += (
                            single_jump_kernel(
                                m, y, n, v, obs, ht, F, G, Lambda, pi,
                                0, n_points  # mid-rectangular method = 0
                            )
                            * psi[n, v]
                            * delta
                        )
                if two_jumps:
                    for v in range(psi.shape[1]):
                        res[m, y] += (
                            double_jump_kernel(
                                m, y, n, v, obs, ht, F, G, Lambda, pi,
                                0, n_points, delta  # mid-rectangular method = 0
                            )
                            * psi[n, v]
                            * delta
                        )
    return res


class Filter(object):
    def __init__(
        self, pi_init, pi, M_net, F, G,
        N, Lambda, ht, delta, n_points=3,
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
        self.delta = delta # TODO delta[n]
        self.filter_step = filter_step

        self.two_jumps = two_jumps

        self.F = F
        self.G = G

        self.psi = pi_init.copy()
    
    def update(self, obs):
        new_psi = self.filter_step(
            self.psi, obs, self.F, self.G, 
            self.Lambda, self.lam, self.pi,
            self.ht, self.delta, self.N, self.n_points,
            self.two_jumps
        )

        normalizer = new_psi.sum() * self.delta
        if normalizer != 0:
            self.psi = new_psi / normalizer
        # else:
        #     raise ZeroDivisionError('psi fell to zero')
    
    def estimate(self):
        theta_est = self.psi.sum(axis=1)
        theta_est = theta_est / theta_est.sum()
        y_est = np.zeros(self.M_net.shape[2]) 
        for n in range(self.N):
            y_est += self.psi[n] @ self.M_net[n] * self.delta

        #self.psi.sum(axis=0) @ self.M_net * self.delta
        return theta_est, y_est

