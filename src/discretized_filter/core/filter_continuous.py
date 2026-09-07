"""Predict/correct density filter for continuous additive observations."""

import numpy as np
from scipy.linalg import expm

from discretized_filter.core.continuous_kernel import continuous_filter_step
from discretized_filter.core.observations import split_channels


class ContinuousFilter:
    """Time-homogeneous reset-jump model with Gaussian and Poisson channels.

    ``C[n, j, k, :]`` contains drift/variance for NORMAL, intensity for POISSON,
    as in :class:`Filter`. Independent Gaussian variances must be constant
    across states/grid/time. Channel functions are declarations; ``C`` is the
    authoritative parameter array. Time independence is a caller contract.

    Each update predicts exactly over its interval and then applies the
    observation correction. This splitting is not an exact finite-step filter
    for a switching hidden state. Include every counting-event time in the
    update grid when available; binned counts use an endpoint approximation.
    """

    def __init__(self, pi_init, pi, M_net, C, N, Lambda, ht, delta, channels):
        self.channels = tuple(channels)
        self.continuous_indices, self.counting_indices = split_channels(self.channels)
        if not isinstance(N, (int, np.integer)) or N <= 0:
            raise ValueError("N must be a positive integer")
        self.N = int(N)
        self.ht = self._interval(ht)
        self.pi = np.array(pi, dtype=float, copy=True)
        self.psi = np.array(pi_init, dtype=float, copy=True)
        self.M_net = np.array(M_net, dtype=float, copy=True)
        self.C = np.array(C, dtype=float, copy=True)
        self.delta = np.array(delta, dtype=float, copy=True)
        self.Lambda = np.array(Lambda, dtype=float, copy=True)
        if self.pi.ndim != 2 or self.pi.shape[0] != N or self.pi.shape[1] == 0:
            raise ValueError("pi must have shape (N, G) with G > 0")
        shape = self.pi.shape
        if self.psi.shape != shape or self.delta.shape != (N,):
            raise ValueError("pi_init or delta has an invalid shape")
        if (not np.all(np.isfinite(self.delta)) or np.any(self.delta <= 0)
                or not np.all(np.isfinite(self.pi)) or np.any(self.pi < 0)
                or not np.all(np.isfinite(self.psi)) or np.any(self.psi < 0)):
            raise ValueError("Densities must be finite and nonnegative; cell volumes must be positive")
        reset_mass = self.pi.sum(axis=1) * self.delta
        initial_mass = np.sum(self.psi * self.delta[:, None])
        if (not np.all(np.isfinite(reset_mass)) or np.any(reset_mass <= 0)
                or not np.isfinite(initial_mass) or initial_mass <= 0):
            raise ValueError("Every reset density and the joint initial density require positive finite mass")
        self.pi /= reset_mass[:, None]
        self.psi /= initial_mass
        if (self.M_net.ndim != 3 or self.M_net.shape[:2] != shape
                or self.M_net.shape[2] == 0 or not np.all(np.isfinite(self.M_net))):
            raise ValueError("M_net must be a finite array with shape (N, G, M)")
        slots = 2 if len(self.continuous_indices) else 1
        if (self.C.ndim != 4 or self.C.shape[:3] != (*shape, len(self.channels))
                or self.C.shape[3] < slots or not np.all(np.isfinite(self.C))):
            raise ValueError("C has invalid shape or nonfinite parameters")
        if self.Lambda.shape != (N, N) or not np.all(np.isfinite(self.Lambda)):
            raise ValueError("Lambda must be a finite (N, N) row generator")
        off_diagonal = self.Lambda.copy()
        np.fill_diagonal(off_diagonal, 0)
        if (np.any(off_diagonal < 0) or np.any(np.diag(self.Lambda) > 0)
                or not np.allclose(self.Lambda.sum(axis=1), 0, rtol=0, atol=1e-12)):
            raise ValueError("Lambda must have nonnegative off-diagonals and rows summing to zero")
        self.drift = self.C[:, :, self.continuous_indices, 0].copy()
        all_variance = self.C[:, :, self.continuous_indices, 1] if len(self.continuous_indices) else np.empty((*shape, 0))
        self.variance = all_variance[0, 0].copy()
        if (np.any(self.variance <= 0)
                or not np.allclose(all_variance, self.variance, rtol=1e-12, atol=0)):
            raise ValueError("Gaussian variance must be positive and constant across states/grid")
        self.intensity = self.C[:, :, self.counting_indices, 0].copy()
        if np.any(self.intensity < 0):
            raise ValueError("Poisson intensities must be nonnegative")
        self._transition_cache = {}

    @classmethod
    def from_config(cls, cfg, ht=None):
        """Build from the existing configuration object, optionally changing dt."""
        return cls(cfg.pi_init, cfg.pi, cfg.M_net, cfg.C, cfg.N, cfg.Lambda,
                   cfg.ht if ht is None else ht, cfg.delta, cfg.channels)

    @staticmethod
    def _interval(dt):
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite scalar")
        return float(dt)

    @staticmethod
    def _vector(value, length, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (length,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector with shape ({length},)")
        return result

    def update(self, obs=None, dt=None, *, continuous_obs=None, counting_obs=None):
        """Assimilate interval increments, merged or split by channel family.

        Missing counts mean no events. Missing diffusion increments are an
        error when diffusion channels exist. The observed increment is used
        once with the actual ``dt`` (default ``ht``), without subdivision.
        """
        interval = self._interval(self.ht if dt is None else dt)
        if obs is not None:
            if continuous_obs is not None or counting_obs is not None:
                raise ValueError("Provide merged obs or split observations, not both")
            merged = self._vector(obs, len(self.channels), "obs")
            continuous_obs = merged[self.continuous_indices]
            counting_obs = merged[self.counting_indices]
        if continuous_obs is None:
            if len(self.continuous_indices):
                raise ValueError("Continuous observation increments are required")
            continuous_obs = np.empty(0)
        if counting_obs is None:
            counting_obs = np.zeros(len(self.counting_indices))
        dxi = self._vector(continuous_obs, len(self.continuous_indices), "continuous_obs")
        counts = self._vector(counting_obs, len(self.counting_indices), "counting_obs")
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("Counting increments must be nonnegative integers")
        if interval not in self._transition_cache:
            transition = expm(self.Lambda * interval)
            survival = np.exp(np.diag(self.Lambda) * interval)
            # Keep the cache bounded for irregular event-time grids.
            if len(self._transition_cache) >= 128:
                self._transition_cache.pop(next(iter(self._transition_cache)))
            self._transition_cache[interval] = transition, survival
        transition, survival = self._transition_cache[interval]
        self.psi = continuous_filter_step(
            self.psi, self.pi, self.delta, transition, survival, self.drift,
            self.variance, self.intensity, dxi, counts, interval,
        )

    def estimate(self):
        """Return theta probabilities and the posterior mean of Y."""
        mass = self.psi * self.delta[:, None]
        return mass.sum(axis=1), np.einsum("ng,ngm->m", mass, self.M_net)
