"""Independent finite-state and static Bayes checks of continuous filtering."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm

from discretized_filter.config import _build
from discretized_filter.core.continuous_kernel import continuous_filter_step, predict_density
from discretized_filter.core.densities import NORMAL, POISSON, ObsChannel
from discretized_filter.core.filter import Filter
from discretized_filter.core.filter_continuous import ContinuousFilter


def drift(t, y, theta):
    return y[:, :1] + theta


def variance(t, y, theta):
    return np.ones((len(y), 1)) * 0.7


def intensity(t, y, theta):
    return np.ones((len(y), 1)) * (theta + 1)


NORMAL_CHANNEL = ObsChannel(NORMAL, drift=drift, var=variance)
COUNT_CHANNEL = ObsChannel(POISSON, intensity=intensity)


def model(channels=(NORMAL_CHANNEL,), generator=None):
    delta = np.array([0.5, 2.0])
    grid = np.array([[[-1.0], [0.0], [1.0]], [[0.0], [1.0], [2.0]]])
    reset = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    initial = np.array([[0.3, 0.0, 0.1], [0.04, 0.12, 0.08]])
    C = np.zeros((2, 3, len(channels), 2))
    for k, ch in enumerate(channels):
        if ch.kind == NORMAL:
            C[:, :, k, 0] = grid[:, :, 0]
            C[:, :, k, 1] = 0.7
        else:
            C[:, :, k, 0] = [[0, 1, 3], [0.5, 2, 4]]
    if generator is None:
        generator = np.array([[-1.3, 1.3], [0.4, -0.4]])
    return dict(pi_init=initial, pi=reset, M_net=grid, C=C, N=2,
                Lambda=generator, ht=0.2, delta=delta, channels=channels)


class ContinuousFilterTests(unittest.TestCase):
    def test_prediction_matches_dense_generator_with_returns(self):
        f = ContinuousFilter(**model(channels=()))
        dense = np.zeros((6, 6))
        for n in range(2):
            for j in range(3):
                dense[n * 3 + j, n * 3 + j] = f.Lambda[n, n]
                for m in range(2):
                    if n != m:
                        dense[n * 3 + j, m * 3:(m + 1) * 3] = f.Lambda[n, m] * f.pi[m] * f.delta[m]
        dt = 2.4
        initial_mass = f.psi * f.delta[:, None]
        expected = (initial_mass.ravel() @ expm(dense * dt)).reshape(2, 3)
        before = f.psi.copy()
        predicted = predict_density(f.psi, f.pi, f.delta, expm(f.Lambda * dt),
                                    np.exp(np.diag(f.Lambda) * dt))
        assert_allclose(predicted * f.delta[:, None], expected, atol=1e-14)
        assert_array_equal(f.psi, before)
        f.update(dt=dt)
        assert_allclose(f.psi * f.delta[:, None], expected, atol=1e-14)

    def test_invalid_transition_cannot_be_silently_clipped(self):
        f = ContinuousFilter(**model())
        with self.assertRaises(ValueError):
            predict_density(f.psi, f.pi, f.delta, np.eye(2) * 0.5, np.ones(2))

    def test_static_gaussian_matches_direct_bayes(self):
        f = ContinuousFilter(**model(generator=np.zeros((2, 2))))
        dt, dx = 0.37, 1.1
        prior = f.psi.copy()
        likelihood = np.exp(-((dx - f.drift[:, :, 0] * dt) ** 2) / (2 * 0.7 * dt))
        expected = prior * likelihood
        expected /= np.sum(expected * f.delta[:, None])
        f.update([dx], dt=dt)
        assert_allclose(f.psi, expected, atol=1e-14)
        self.assertEqual(f.psi[0, 1], 0)

    def test_static_poisson_zero_intensity_and_multiple_counts(self):
        for count in (0, 1, 4):
            with self.subTest(count=count):
                f = ContinuousFilter(**model((COUNT_CHANNEL,), np.zeros((2, 2))))
                h = f.intensity[:, :, 0]
                expected = f.psi * np.exp(-0.3 * h) * h ** count
                expected /= np.sum(expected * f.delta[:, None])
                f.update(counting_obs=[count], dt=0.3)
                assert_allclose(f.psi, expected, atol=1e-14)

    def test_mixed_split_inputs_match_merged_and_joint_bayes(self):
        kwargs = model((COUNT_CHANNEL, NORMAL_CHANNEL, COUNT_CHANNEL), np.zeros((2, 2)))
        a, b = ContinuousFilter(**kwargs), ContinuousFilter(**kwargs)
        expected = a.psi * np.exp(-0.2 * a.intensity.sum(axis=2))
        expected *= a.intensity[:, :, 0] ** 2 * a.intensity[:, :, 1]
        expected *= np.exp(-((0.4 - a.drift[:, :, 0] * 0.2) ** 2) / (2 * 0.7 * 0.2))
        expected /= np.sum(expected * a.delta[:, None])
        a.update([2, 0.4, 1])
        b.update(continuous_obs=[0.4], counting_obs=[2, 1])
        assert_allclose(a.psi, expected, atol=1e-14)
        assert_allclose(a.psi, b.psi)
        assert_array_equal(a.continuous_indices, [1])
        assert_array_equal(a.counting_indices, [0, 2])

    def test_impossible_observation_raises_without_mutation(self):
        kwargs = model((COUNT_CHANNEL,), np.zeros((2, 2)))
        kwargs['C'][:] = 0
        f = ContinuousFilter(**kwargs)
        old = f.psi.copy()
        with self.assertRaisesRegex(ValueError, 'impossible'):
            f.update([1])
        assert_array_equal(f.psi, old)
        f.update()
        assert_allclose(f.psi, old)

    def test_extreme_gaussian_increment_is_finite(self):
        f = ContinuousFilter(**model(generator=np.zeros((2, 2))))
        f.update([1e250])
        self.assertTrue(np.all(np.isfinite(f.psi)))
        assert_allclose(np.sum(f.psi * f.delta[:, None]), 1)
        self.assertGreater(f.psi[1, 2], 0)

    def test_extreme_tied_likelihood_preserves_prior_odds(self):
        posterior = continuous_filter_step(
            np.array([[0.2, 0.6, 0.2]]), np.array([[0.2, 0.6, 0.2]]),
            np.array([1.]), np.array([[1.]]), np.array([1.]),
            np.array([[[0.], [1e150], [1e150]]]), np.array([1.]),
            np.empty((1, 3, 0)), np.array([1e150]), np.empty(0), 1.,
        )
        assert_allclose(posterior, [[0., 0.75, 0.25]], atol=1e-15)

    def test_extreme_gaussian_keeps_smaller_counting_evidence(self):
        posterior = continuous_filter_step(
            np.array([[0.2, 0.6, 0.2]]), np.array([[0.2, 0.6, 0.2]]),
            np.array([1.]), np.array([[1.]]), np.array([1.]),
            np.array([[[0.], [1e150], [1e150]]]), np.array([1.]),
            np.array([[[1.], [1.], [2.]]]), np.array([1e150]), np.array([1.]), 1.,
        )
        expected = np.array([0., 0.6 * np.exp(-1), 0.2 * 2 * np.exp(-2)])
        expected /= expected.sum()
        assert_allclose(posterior[0], expected, atol=1e-15)

    def test_opposing_gaussian_channels_keep_counting_evidence(self):
        posterior = continuous_filter_step(
            np.array([[0.6, 0.4]]), np.array([[0.6, 0.4]]),
            np.array([1.]), np.array([[1.]]), np.array([1.]),
            np.array([[[0., 1e150], [1e150, 0.]]]), np.ones(2),
            np.array([[[1.], [2.]]]), np.array([1e150, 1e150]), np.array([1.]), 1.,
        )
        expected = np.array([0.6 * np.exp(-1), 0.4 * 2 * np.exp(-2)])
        expected /= expected.sum()
        assert_allclose(posterior[0], expected, atol=1e-15)

    def test_normalized_density_and_estimate_quadrature(self):
        kwargs = model()
        f = ContinuousFilter(**kwargs)
        assert_allclose(f.pi.sum(axis=1) * f.delta, 1)
        theta_est, y_est = f.estimate()
        assert_allclose(theta_est, f.psi.sum(axis=1) * f.delta)
        assert_allclose(theta_est.sum(), 1)
        assert_allclose(y_est, np.sum(f.psi[:, :, None] * f.delta[:, None, None] * f.M_net, axis=(0, 1)))
        discrete = Filter(f.psi, f.pi, f.M_net, f.C, f.N, f.Lambda,
                          f.ht, f.delta, obs_density=None)
        assert_allclose(discrete.estimate()[0], theta_est)

    def test_update_validation(self):
        f = ContinuousFilter(**model((NORMAL_CHANNEL, COUNT_CHANNEL)))
        for kwargs in ({}, {'obs': [0, 0], 'counting_obs': [0]}, {'obs': [0]},
                       {'obs': [0, -1]}, {'obs': [0, 0.5]}, {'obs': [np.nan, 0]},
                       {'obs': [0, 0], 'dt': 0}, {'obs': [0, 0], 'dt': np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                f.update(**kwargs)
        f.update(continuous_obs=[0])

    def test_constructor_validation(self):
        mutations = [
            ('delta', [1, 0]), ('delta', [1]), ('pi', np.zeros((2, 3))),
            ('pi_init', np.zeros((2, 3))), ('Lambda', [[-1, 1], [-1, 1]]),
            ('Lambda', [[-1, 0], [0, 0]]), ('M_net', np.zeros((2, 2, 1))),
            ('ht', -0.1),
        ]
        for key, value in mutations:
            kwargs = model()
            kwargs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ContinuousFilter(**kwargs)
        kwargs = model()
        kwargs['C'][0, 0, 0, 1] = 0.8
        with self.assertRaisesRegex(ValueError, 'constant'):
            ContinuousFilter(**kwargs)

    def test_unsupported_channel_generators(self):
        channels = [ObsChannel(NORMAL, drift=drift, var=variance, noise=lambda size: size),
                    ObsChannel(NORMAL, drift=drift, var=variance, intensity=intensity),
                    ObsChannel(POISSON, drift=drift, intensity=intensity),
                    ObsChannel(NORMAL, drift=drift, var=variance,
                               loc=drift, scale=variance, alpha=variance)]
        for ch in channels:
            with self.subTest(ch=ch), self.assertRaises(ValueError):
                ContinuousFilter(**model((ch,)))

    def test_config_passes_actual_theta_and_wires_observations(self):
        module = SimpleNamespace(
            exp_id='test', T=1, ht=0.1, seed=42, N=2,
            Lambda=np.array([[0., 0.3], [0.4, 0.]]),
            y_intervals=[np.array([[0., 1.], [0., 1.]])], num1=3,
            pi_family='uniform', channels=[NORMAL_CHANNEL, COUNT_CHANNEL],
            n_points=2, two_jumps=False,
        )
        cfg = _build(module, Path('test_config.py'))
        assert_allclose(cfg.C[1, :, 0, 0], cfg.M_net[1, :, 0] + 1)
        assert_allclose(cfg.C[0, :, 1, 0], 1)
        assert_allclose(cfg.C[1, :, 1, 0], 2)
        assert_array_equal(cfg.continuous_indices, [0])
        assert_array_equal(cfg.counting_indices, [1])
        f = ContinuousFilter.from_config(cfg, ht=0.25)
        self.assertEqual(f.ht, 0.25)
        observed = cfg.get_continuous_obs([0, 0.5, 1], [1], [[0.2]], [1], seed=12)
        self.assertEqual(observed.increments.shape[1], 2)


if __name__ == '__main__':
    unittest.main()
