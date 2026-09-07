"""Stationary linear non-Gaussian filter for two-coordinate reset states.

The component-major extended column state is (theta, Y1*theta, Y2*theta).
Lambda is a row generator. Diffusion observations satisfy dX = H V dt +
sqrt(R) dW in the Itô convention; R denotes the variance rate, not its root.
The process bracket rate is unconditional and constant under stationarity.
"""

import numpy as np


def build_stationary_linear_model(pi, M_net, delta, Lambda, p0, variance):
    """Return ``(A, B, H, R, mean0, K0)`` for the stationary reset model.

    ``pi`` has shape (N,G), ``M_net`` (N,G,2), ``delta`` and ``p0`` (N,),
    ``Lambda`` (N,N), and positive constant observation ``variance`` (2,).
    Each row of pi must integrate to one with delta; p0 must be stationary.
    The initial joint law is p0[:,None]*pi. Moments use exactly this grid's
    quadrature, without renormalizing or changing its reset distribution.

    A, B, K0 have shape (3N,3N), H (2,3N), R (2,2), and mean0 (3N,).
    B = -(A S + S A.T) follows from stationarity of the raw second moment S.
    """
    pi, M_net, delta, Lambda, p0, variance = (
        np.asarray(value, dtype=float)
        for value in (pi, M_net, delta, Lambda, p0, variance)
    )
    if pi.ndim != 2 or min(pi.shape) == 0:
        raise ValueError("pi must have nonempty shape (N, G).")
    n, grid_size = pi.shape
    if (M_net.shape != (n, grid_size, 2) or delta.shape != (n,)
            or Lambda.shape != (n, n) or p0.shape != (n,)
            or variance.shape != (2,)):
        raise ValueError("Incompatible stationary two-coordinate model shapes.")
    if not all(np.all(np.isfinite(value))
               for value in (pi, M_net, delta, Lambda, p0, variance)):
        raise ValueError("Model inputs must be finite.")
    if np.any(pi < 0) or np.any(delta <= 0) or np.any(p0 < 0):
        raise ValueError("Densities/probabilities must be nonnegative; delta positive.")
    if np.any(variance <= 0):
        raise ValueError("Observation variance rates must be positive constants.")
    with np.errstate(over="raise", invalid="raise"):
        weights = pi * delta[:, None]
    if (not np.allclose(weights.sum(axis=1), 1, rtol=1e-10, atol=1e-12)
            or not np.isclose(p0.sum(), 1, rtol=1e-10, atol=1e-12)):
        raise ValueError("Each reset density and p0 must be normalized.")
    qoff = Lambda - np.diag(np.diag(Lambda))
    rate_tolerance = 1e-12 * np.max(np.abs(Lambda))
    if (np.any(qoff < 0) or np.any(np.diag(Lambda) > 0)
            or np.any(np.abs(Lambda.sum(axis=1)) > rate_tolerance)):
        raise ValueError("Lambda must be a row generator.")
    if np.any(np.abs(Lambda.T @ p0) > rate_tolerance):
        raise ValueError("p0 must be stationary for Lambda.")

    functions = np.concatenate((np.ones((1, n, grid_size)),
                                M_net.transpose(2, 0, 1)), axis=0)
    with np.errstate(over="raise", invalid="raise"):
        mu = np.einsum("ang,ng->an", functions, weights)
        second = np.einsum("ang,bng,ng->abn", functions, functions, weights)
        mean0 = (mu * p0).reshape(3 * n)
        raw_second = np.zeros((3 * n, 3 * n))
        A = np.zeros_like(raw_second)
        A[:n, :n] = Lambda.T
        for a in range(3):
            row = slice(a * n, (a + 1) * n)
            for b in range(3):
                col = slice(b * n, (b + 1) * n)
                raw_second[row, col] = np.diag(p0 * second[a, b])
            if a > 0:
                A[row, :n] = mu[a, :, None] * qoff.T
                A[row, row] = np.diag(np.diag(Lambda))
        B = -(A @ raw_second + raw_second @ A.T)
        K0 = raw_second - np.outer(mean0, mean0)
        B = (B + B.T) / 2
        K0 = (K0 + K0.T) / 2
    H = np.zeros((2, 3 * n))
    H[0, n:2 * n] = 1
    H[1, 2 * n:3 * n] = 1
    R = np.diag(variance)
    if not all(np.all(np.isfinite(value)) for value in (A, B, mean0, K0)):
        raise FloatingPointError("Extended-state moments exceed numerical range.")
    return A, B, H, R, mean0, K0


def linear_filter_step(mean, covariance, dxi, dt, A, B, H, R):
    """Return one explicit Euler Itô step of the Kalman-Bucy equations.

    Shapes follow :func:`build_stationary_linear_model`; dxi has shape (2,)
    and dt is positive. Both right-hand sides use the old mean/covariance.
    Only roundoff asymmetry is removed: no simplex or eigenvalue projection
    is applied. Euler does not guarantee positive covariance at large dt;
    the caller must check convergence/positivity and choose a small step.
    """
    mean, covariance, dxi, A, B, H, R = (
        np.asarray(value, dtype=float)
        for value in (mean, covariance, dxi, A, B, H, R)
    )
    if mean.ndim != 1 or mean.size == 0 or mean.size % 3:
        raise ValueError("mean must have shape (3N,) for a positive N.")
    size = mean.size
    if (covariance.shape != (size, size) or A.shape != (size, size)
            or B.shape != (size, size) or H.shape != (2, size)
            or R.shape != (2, 2) or dxi.shape != (2,)):
        raise ValueError("Incompatible linear filter shapes.")
    if (np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0
            or not all(np.all(np.isfinite(value))
                       for value in (mean, covariance, dxi, A, B, H, R))):
        raise ValueError("Inputs must be finite and dt must be positive.")
    if np.any(np.diag(R) <= 0) or np.any(R != np.diag(np.diag(R))):
        raise ValueError("R must contain positive diagonal variance rates.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        gain = np.linalg.solve(R, H @ covariance.T).T
        next_mean = mean + (A @ mean) * dt + gain @ (dxi - (H @ mean) * dt)
        next_covariance = covariance + dt * (
            A @ covariance + covariance @ A.T + B - gain @ R @ gain.T
        )
        next_covariance = (next_covariance + next_covariance.T) / 2
    if (not np.all(np.isfinite(next_mean))
            or not np.all(np.isfinite(next_covariance))):
        raise FloatingPointError("Linear filter update exceeds numerical range.")
    return next_mean, next_covariance
