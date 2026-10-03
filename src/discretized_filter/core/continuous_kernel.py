"""Nopython predict/correct kernels for an Itô additive observation model.

The hidden chain uses a row generator; each jump resets Y to the destination
conditional density. Prediction is exact for that hidden process. Correction
uses endpoint drifts/intensities and is a splitting approximation. All state
arrays are densities, integrated with the state-specific cell volumes.
Arithmetic uses float64, including likelihood products; nonfinite intermediate
results raise FloatingPointError rather than dropping evidence. Finite rounding
and cancellation can differ from an extended-precision calculation.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def predict(psi, pi, delta, transition, survival):
    """Return the exact hidden-state prediction without modifying inputs.

    ``psi`` and reset ``pi`` have shape ``(N, G)``; ``delta`` and no-jump
    ``survival`` have shape ``(N,)``. ``transition`` is the row-stochastic
    ``(N, N)`` matrix exp(Q*dt). Inputs are validated by the wrapper; each
    row of ``pi`` integrates to one and ``psi`` integrates globally to one.
    """
    psi = np.asarray(psi, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    survival = np.asarray(survival, dtype=np.float64)
    jumped = np.asarray(transition, dtype=np.float64) - np.diag(survival)
    tolerance = 64 * np.finfo(np.float64).eps * max(1, len(survival))
    if not np.all(np.isfinite(jumped)) or np.any(jumped < -tolerance):
        raise ValueError("Transition minus no-jump survival must be nonnegative.")
    # Includes paths that leave and subsequently return to the same state.
    jumped = np.maximum(jumped, 0.0)
    marginal = np.sum(psi * delta[:, None], axis=1)
    predicted = survival[:, None] * psi + (marginal @ jumped)[:, None] * pi
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
        raise FloatingPointError("Hidden-state prediction is not a finite density.")
    return predicted


# Preserve the original public name and its compiled signatures.
predict_density = predict


@njit(cache=True)
def _accumulate_channel(log_likelihood, contribution):
    """Center one channel and the accumulated evidence, checking overflow."""
    if not np.all(np.isfinite(contribution)):
        raise FloatingPointError("Observation log likelihood is not finite.")
    centered = contribution - np.max(contribution)
    if not np.all(np.isfinite(centered)):
        raise FloatingPointError("Observation log likelihood is not finite.")
    log_likelihood += centered
    if not np.all(np.isfinite(log_likelihood)):
        raise FloatingPointError("Observation log likelihood is not finite.")
    log_likelihood -= np.max(log_likelihood)
    if not np.all(np.isfinite(log_likelihood)):
        raise FloatingPointError("Observation log likelihood is not finite.")


@njit(cache=True)
def update(predicted, delta, drift, variance, intensity, dxi, counts, dt):
    """Apply diffusion/counting evidence to a prediction and normalize.

    ``predicted`` is ``(N, G)``, ``delta`` is ``(N,)``, ``drift`` is
    ``(N, G, K)``, ``variance`` and ``dxi`` are ``(K,)``, ``intensity`` is
    ``(N, G, R)``, and nonnegative integer-valued ``counts`` is ``(R,)``.
    Empty channel families are allowed. The wrapper validates finite inputs,
    positive state-independent variance rates, positive volumes, and dt > 0.

    Log likelihood is sum(g*dxi/variance - dt*g**2/(2*variance)) minus
    dt*sum(intensity), plus sum(counts*log(intensity)). State-independent
    constants are omitted. Positive counts at zero intensity exclude a node;
    an observation impossible on the predicted support raises ValueError.
    Inputs are never mutated. The returned float64 density has unit weighted
    mass; unrepresentable arithmetic raises rather than dropping evidence.
    Centering limits loss of small terms, but finite precision cannot preserve
    every cancellation between arbitrarily large, opposing channel evidence.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    dxi = np.asarray(dxi, dtype=np.float64)
    step = np.float64(dt)
    support = predicted > 0
    for n in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            for channel in range(len(counts)):
                if counts[channel] > 0 and intensity[n, j, channel] <= 0:
                    support[n, j] = False
    size = np.count_nonzero(support)
    if size == 0:
        raise ValueError("Observation is impossible on the predicted support.")

    # Explicit gathering preserves the original row-major support order.
    rows = np.empty(size, dtype=np.int64)
    columns = np.empty(size, dtype=np.int64)
    index = 0
    for n in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            if support[n, j]:
                rows[index], columns[index] = n, j
                index += 1
    log_likelihood = np.zeros(size, dtype=np.float64)
    contribution = np.empty(size, dtype=np.float64)
    for channel in range(len(dxi)):
        reference = drift[rows[0], columns[0], channel]
        for index in range(size):
            value = drift[rows[index], columns[index], channel]
            difference = value - reference
            midpoint = step * (value + reference) / 2
            residual = dxi[channel] - midpoint
            product = difference * residual
            if (not np.isfinite(difference) or not np.isfinite(midpoint)
                    or not np.isfinite(residual) or not np.isfinite(product)):
                raise FloatingPointError("Observation log likelihood is not finite.")
            contribution[index] = product / variance[channel]
        _accumulate_channel(log_likelihood, contribution)
    for channel in range(len(counts)):
        reference = intensity[rows[0], columns[0], channel]
        count = np.float64(counts[channel])
        for index in range(size):
            value = intensity[rows[index], columns[index], channel]
            contribution[index] = -step * (value - reference)
            if count > 0:
                contribution[index] += count * (np.log(value) - np.log(reference))
        _accumulate_channel(log_likelihood, contribution)

    # Center evidence before adding prior odds, retaining tied-node weights.
    log_weight = log_likelihood - np.max(log_likelihood)
    volumes = np.empty(size, dtype=np.float64)
    for index in range(size):
        n, j = rows[index], columns[index]
        log_weight[index] += np.log(predicted[n, j])
        volumes[index] = delta[n]
    log_mass = log_weight + np.log(volumes)
    if not np.all(np.isfinite(log_mass)):
        raise FloatingPointError("Observation log weights are not finite.")
    # Weighted log-sum-exp, retaining exact zeros outside the support.
    centered_mass = log_mass - np.max(log_mass)
    if not np.all(np.isfinite(centered_mass)):
        raise FloatingPointError("Observation log weights are not finite.")
    mass = np.exp(centered_mass)
    density = mass / np.sum(mass) / volumes
    if not np.all(np.isfinite(density)):
        raise FloatingPointError("Posterior density exceeds float64 range.")
    posterior = np.zeros_like(predicted)
    for index in range(size):
        posterior[rows[index], columns[index]] = density[index]
    return posterior


@njit(cache=True)
def continuous_filter_step(psi, pi, delta, transition, survival, drift, variance,
                           intensity, dxi, counts, dt):
    """Compose :func:`predict` and :func:`update` for existing callers."""
    predicted = predict(psi, pi, delta, transition, survival)
    return update(predicted, delta, drift, variance, intensity, dxi, counts, dt)
