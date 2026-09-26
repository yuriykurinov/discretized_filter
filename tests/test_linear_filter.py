"""Independent jump-model and Euler checks for the stationary linear filter."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from discretized_filter.config import set_config
from discretized_filter.core.filter_continuous import ContinuousFilter
from discretized_filter.core.linear_filter import (
    build_stationary_linear_model, linear_filter_step,
)


def tiny_model(p0=None):
    delta = np.array([0.5, 2.0])
    weights = np.array([[0.25, 0.75], [0.6, 0.4]])
    grid = np.array([[[-1., 0.5], [2., 1.5]], [[0.2, -2.], [1.2, 3.]]])
    generator = np.array([[-0.4, 0.4], [0.3, -0.3]])
    if p0 is None:
        p0 = np.array([3 / 7, 4 / 7])
    arguments = (weights / delta[:, None], grid, delta, generator,
                 p0, np.array([0.49, 0.81]))
    return arguments


class LinearFilterTests(unittest.TestCase):
    def test_extended_drift_and_bracket_match_enumerated_jumps(self):
        arguments = tiny_model()
        pi, grid, delta, generator, p0, variance = arguments
        A, B, H, R, mean, covariance = build_stationary_linear_model(*arguments)
        # Enumerate the four finite hidden states and their extended columns.
        values = np.zeros((4, 6))
        stationary = np.empty(4)
        for n in range(2):
            for j in range(2):
                source = 2 * n + j
                values[source, n] = 1
                values[source, 2 + n] = grid[n, j, 0]
                values[source, 4 + n] = grid[n, j, 1]
                stationary[source] = p0[n] * pi[n, j] * delta[n]
        finite_generator = np.zeros((4, 4))
        bracket = np.zeros((6, 6))
        for n in range(2):
            for j in range(2):
                source = 2 * n + j
                for m in range(2):
                    if m == n:
                        continue
                    for k in range(2):
                        destination = 2 * m + k
                        rate = generator[n, m] * pi[m, k] * delta[m]
                        finite_generator[source, destination] = rate
                        jump = values[destination] - values[source]
                        bracket += stationary[source] * rate * np.outer(jump, jump)
                finite_generator[source, source] = -finite_generator[source].sum()
        assert_allclose(stationary @ finite_generator, 0, atol=1e-15)
        assert_allclose(values @ A.T, finite_generator @ values, atol=1e-14)
        assert_allclose(B, bracket, atol=1e-14)
        expected_mean = stationary @ values
        centered = values - expected_mean
        assert_allclose(mean, expected_mean, atol=1e-15)
        assert_allclose(covariance, centered.T @ (stationary[:, None] * centered), atol=1e-14)
        assert_allclose(values @ H.T, grid.reshape(4, 2))
        assert_array_equal(R, np.diag(variance))

    def test_euler_step_uses_old_mean_and_covariance(self):
        A, B, H, R, mean, covariance = build_stationary_linear_model(*tiny_model())
        mean = mean + np.array([0.03, -0.03, 0.1, -0.2, 0.05, 0.04])
        covariance = covariance * 0.7
        old_mean, old_covariance = mean.copy(), covariance.copy()
        dt, observation = 0.017, np.array([0.21, -0.12])
        gain = covariance @ H.T @ np.diag(1 / np.diag(R))
        expected_mean = mean + dt * A @ mean + gain @ (observation - dt * H @ mean)
        expected_covariance = covariance + dt * (
            A @ covariance + covariance @ A.T + B - gain @ R @ gain.T
        )
        next_mean, next_covariance = linear_filter_step(mean, covariance, observation, dt, A, B, H, R)
        assert_allclose(next_mean, expected_mean, atol=1e-14)
        assert_allclose(next_covariance, expected_covariance, atol=1e-14)
        assert_array_equal(mean, old_mean)
        assert_array_equal(covariance, old_covariance)

    def test_nonstationary_initial_distribution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'stationary'):
            build_stationary_linear_model(*tiny_model(p0=np.array([0.5, 0.5])))

    def test_toy_euler_covariance_and_theta_mass_on_both_steps(self):
        cfg = set_config('toy_continuous_additive')
        reset = ContinuousFilter.from_config(cfg).pi
        parameters = build_stationary_linear_model(
            reset, cfg.M_net, cfg.delta, cfg.Lambda, cfg.p0, cfg.C[0, 0, :, 1],
        )
        A, B, H, R, mean0, covariance0 = parameters
        rng = np.random.default_rng(20260906)
        fine_dt = 0.01
        increments = rng.standard_normal((round(cfg.T / fine_dt), 2)) * np.sqrt(np.diag(R) * fine_dt)
        for stride in (1, 2):
            with self.subTest(dt=fine_dt * stride):
                mean, covariance = mean0.copy(), covariance0.copy()
                grouped = increments.reshape(-1, stride, 2).sum(axis=1)
                for observation in grouped:
                    mean, covariance = linear_filter_step(
                        mean, covariance, observation, fine_dt * stride, A, B, H, R,
                    )
                    self.assertTrue(np.all(np.isfinite(mean)))
                    self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
                    assert_allclose(covariance, covariance.T, atol=1e-14)
                    assert_allclose(mean[:cfg.N].sum(), 1, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
