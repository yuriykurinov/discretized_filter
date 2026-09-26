"""Prediction and point-observation correction for a reset-jump state."""

import numpy as np
import numba as nb
from scipy.linalg import expm

from discretized_filter.core.continuous_kernel import predict_density


@nb.njit
def predict_density_article(psi, pi, delta, transition, survival):
    """Apply the printed article formula (3.3)-(3.4) on a common grid."""
    psi = np.asarray(psi, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    survival = np.asarray(survival, dtype=np.float64)
    jumped = np.asarray(transition, dtype=np.float64) - np.diag(survival)
    tolerance = 64 * np.finfo(np.float64).eps * max(1, len(survival))
    if not np.all(np.isfinite(jumped)) or np.any(jumped < -tolerance):
        raise ValueError("Transition minus no-jump survival must be nonnegative")
    jumped = np.maximum(jumped, 0.0)
    theta = np.sum(psi * delta[:, None], axis=1)
    predicted = survival[:, None] * psi + theta[:, None] * (jumped @ pi)
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
        raise FloatingPointError("Article prediction is not a finite density")
    return predicted


@nb.njit
def update_density(psi, obs, C, delta, obs_density):
    """Normalize point likelihoods using grid cell volumes."""
    log_weights = np.full(psi.shape, -np.inf)
    u = np.ones(1)
    comp = np.empty((1, C.shape[2], C.shape[3]))
    largest = -np.inf
    for n in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            if psi[n, j] == 0:
                continue
            comp[0] = C[n, j]
            likelihood = obs_density(obs, u, comp, 1)
            if not np.isfinite(likelihood):
                raise FloatingPointError("Observation likelihood is not finite")
            if likelihood < 0:
                raise ValueError("Observation likelihood is negative")
            if likelihood > 0:
                value = np.log(psi[n, j]) + np.log(likelihood) + np.log(delta[n])
                log_weights[n, j] = value
                largest = max(largest, value)
    if not np.isfinite(largest):
        raise ValueError("Observation is impossible on the predicted support")
    posterior = np.exp(log_weights - largest)
    total = posterior.sum()
    for n in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            posterior[n, j] = posterior[n, j] / total / delta[n]
            if not np.isfinite(posterior[n, j]):
                raise FloatingPointError("Posterior density exceeds float64 range")
    return posterior


class DiscreteFilter:
    """Filter point observations with a Numba-compatible joint density."""

    def __init__(self, pi_init, pi, M_net, C, N, Lambda, ht, delta, obs_density):
        self.obs_density = obs_density
        self.pi = np.array(pi, dtype=float, copy=True)
        self.psi = np.array(pi_init, dtype=float, copy=True)
        self.M_net = np.array(M_net, dtype=float, copy=True)
        self.C = np.array(C, dtype=float, copy=True)
        self.Lambda = np.array(Lambda, dtype=float, copy=True)
        self.delta = np.array(delta, dtype=float, copy=True)
        self.ht = self._interval(ht)
        self.N = N
        if (self.pi.ndim != 2 or self.pi.shape[0] != N or self.psi.shape != self.pi.shape
                or self.M_net.shape[:2] != self.pi.shape
                or self.C.ndim != 4 or self.C.shape[:2] != self.pi.shape
                or self.C.shape[2] == 0 or self.C.shape[3] == 0
                or self.delta.shape != (N,) or self.Lambda.shape != (N, N)):
            raise ValueError("Invalid model array shapes")
        if (not np.all(np.isfinite(self.psi)) or np.any(self.psi < 0)
                or not np.all(np.isfinite(self.pi)) or np.any(self.pi < 0)
                or not np.all(np.isfinite(self.C)) or not np.all(np.isfinite(self.M_net))
                or not np.all(np.isfinite(self.delta)) or np.any(self.delta <= 0)
                or not np.all(np.isfinite(self.Lambda))):
            raise ValueError("Model arrays must be finite with nonnegative densities")
        offdiag = self.Lambda.copy()
        np.fill_diagonal(offdiag, 0)
        if (np.any(offdiag < 0) or np.any(np.diag(self.Lambda) > 0)
                or not np.allclose(self.Lambda.sum(axis=1), 0, atol=1e-12)):
            raise ValueError("Lambda must be a row generator")
        reset_mass = np.sum(self.pi * self.delta[:, None], axis=1)
        initial_mass = np.sum(self.psi * self.delta[:, None])
        if np.any(reset_mass <= 0) or initial_mass <= 0:
            raise ValueError("Initial and reset densities need positive mass")
        self.pi /= reset_mass[:, None]
        self.psi /= initial_mass

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.pi_init, cfg.pi, cfg.M_net, cfg.C, cfg.N,
                   cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)

    @staticmethod
    def _interval(dt):
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        return float(dt)

    def predict(self, dt=None, *, method="reset"):
        """Forecast with the reset model or the printed article formula."""
        if method not in ("reset", "article"):
            raise ValueError("method must be 'reset' or 'article'")
        if method == "article" and (
                not np.all(self.M_net == self.M_net[0])
                or not np.all(self.delta == self.delta[0])):
            raise ValueError("Article prediction requires a common grid and equal cell volumes")
        step = self._interval(self.ht if dt is None else dt)
        transition = expm(self.Lambda * step)
        survival = np.exp(np.diag(self.Lambda) * step)
        kernel = predict_density_article if method == "article" else predict_density
        self.psi = kernel(self.psi, self.pi, self.delta, transition, survival)

    def update(self, obs):
        """Apply only the likelihood of the new point observation."""
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 0:
            obs = obs.reshape(1)
        if obs.shape != (self.C.shape[2],) or not np.all(np.isfinite(obs)):
            raise ValueError("obs must be a finite vector with one value per channel")
        self.psi = update_density(self.psi, obs, self.C, self.delta, self.obs_density)

    def estimate(self):
        mass = self.psi * self.delta[:, None]
        return mass.sum(axis=1), np.einsum("ng,ngm->m", mass, self.M_net)
