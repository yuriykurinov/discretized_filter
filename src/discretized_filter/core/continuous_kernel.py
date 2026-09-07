"""Predict/correct kernels for an Itô additive observation model.

The hidden chain uses a row generator; each jump resets Y to the destination
conditional density. Prediction is exact for that hidden process. Correction
uses endpoint drifts/intensities and is a splitting approximation. All state
arrays are densities, integrated with the state-specific cell volumes.
"""

import numpy as np


def predict_density(psi, pi, delta, transition, survival):
    """Return the exact hidden-state prediction without modifying inputs.

    ``psi`` and reset ``pi`` have shape ``(N, G)``; ``delta`` and no-jump
    ``survival`` have shape ``(N,)``. ``transition`` is the row-stochastic
    ``(N, N)`` matrix exp(Q*dt). Inputs are validated by the wrapper; each
    row of ``pi`` integrates to one and ``psi`` integrates globally to one.
    """
    psi = np.asarray(psi, dtype=float)
    pi = np.asarray(pi, dtype=float)
    delta = np.asarray(delta, dtype=float)
    survival = np.asarray(survival, dtype=float)
    jumped = np.asarray(transition, dtype=float) - np.diag(survival)
    tolerance = 64 * np.finfo(float).eps * max(1, len(survival))
    if not np.all(np.isfinite(jumped)) or np.any(jumped < -tolerance):
        raise ValueError("Transition minus no-jump survival must be nonnegative.")
    # Includes paths that leave and subsequently return to the same state.
    jumped = np.maximum(jumped, 0.0)
    marginal = np.sum(psi * delta[:, None], axis=1)
    predicted = survival[:, None] * psi + (marginal @ jumped)[:, None] * pi
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
        raise FloatingPointError("Hidden-state prediction is not a finite density.")
    return predicted


def continuous_filter_step(psi, pi, delta, transition, survival, drift, variance,
                           intensity, dxi, counts, dt):
    """Predict, apply joint diffusion/counting evidence, and normalize.

    State/prediction arguments have the shapes documented in
    :func:`predict_density`. ``drift`` is ``(N, G, K)``, ``variance`` and
    ``dxi`` are ``(K,)``; ``intensity`` is ``(N, G, R)`` and integer
    nonnegative ``counts`` is ``(R,)``. Empty channel families are allowed.
    ``variance`` contains positive, state-independent variance rates, and
    ``dt`` is positive. The wrapper validates these input contracts.

    Log likelihood is sum(g*dxi/variance - dt*g**2/(2*variance)) minus
    dt*sum(intensity), plus sum(counts*log(intensity)). State-independent
    constants are omitted. Positive counts at zero intensity exclude a node;
    an observation impossible on the predicted support raises ValueError.
    Inputs are never mutated. The returned float64 density has unit weighted
    mass; unrepresentable arithmetic raises rather than dropping evidence.
    Centering limits loss of small terms, but finite precision cannot preserve
    every cancellation between arbitrarily large, opposing channel evidence.
    """
    predicted = predict_density(psi, pi, delta, transition, survival)
    support = predicted > 0
    for channel, count in enumerate(counts):
        if count > 0:
            support &= intensity[..., channel] > 0
    if not np.any(support):
        raise ValueError("Observation is impossible on the predicted support.")

    # Extended precision protects finite float64 inputs during products.
    wide = np.longdouble
    g = np.asarray(drift, dtype=wide)[support]
    h = np.asarray(intensity, dtype=wide)[support]
    step = wide(dt)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_likelihood = np.zeros(np.count_nonzero(support), dtype=wide)
        for channel, increment in enumerate(dxi):
            values = g[:, channel]
            reference = values[0]
            # Subtract the reference node's log likelihood before evaluating
            # products, preserving prior odds for identical large drifts.
            contribution = ((values - reference)
                            * (wide(increment) - step * (values + reference) / 2)
                            / wide(variance[channel]))
            # Center each channel before combining evidence, so a huge common
            # Gaussian term cannot erase a smaller counting contribution.
            log_likelihood += contribution - np.max(contribution)
            log_likelihood -= np.max(log_likelihood)
        for channel, count in enumerate(counts):
            values = h[:, channel]
            contribution = -step * (values - values[0])
            if count > 0:
                contribution += wide(count) * (np.log(values) - np.log(values[0]))
            log_likelihood += contribution - np.max(contribution)
            log_likelihood -= np.max(log_likelihood)

        if not np.all(np.isfinite(log_likelihood)):
            raise FloatingPointError("Observation log likelihood is not finite.")
        # Center evidence before adding prior odds: a huge common likelihood
        # would otherwise round away unequal prior weights of tied nodes.
        log_weight = (log_likelihood - np.max(log_likelihood)
                      + np.log(predicted[support].astype(wide)))

        volumes = np.broadcast_to(np.asarray(delta, dtype=wide)[:, None],
                                  predicted.shape)[support]
        log_mass = log_weight + np.log(volumes)
        if not np.all(np.isfinite(log_mass)):
            raise FloatingPointError("Observation log weights are not finite.")
        # Weighted log-sum-exp, retaining exact zeros outside the support.
        with np.errstate(under="ignore"):
            mass = np.exp(log_mass - np.max(log_mass))
        density = (mass / np.sum(mass) / volumes).astype(float)
    if not np.all(np.isfinite(density)):
        raise FloatingPointError("Posterior density exceeds float64 range.")
    posterior = np.zeros_like(predicted)
    posterior[support] = density
    return posterior
