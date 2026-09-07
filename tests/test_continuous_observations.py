"""Path-generation, event handling, and exact subset aggregation checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from discretized_filter.core.densities import NORMAL, POISSON, ObsChannel
from discretized_filter.core.filter_continuous import ContinuousFilter
from discretized_filter.core.observations import (
    ContinuousObservations, generate_continuous_observations,
)


def drift(t, y, theta):
    return y[:, :1] + theta


def variance(t, y, theta):
    return np.ones((len(y), 1)) * 0.3


def intensity(t, y, theta):
    return np.ones((len(y), 1)) * 12


NORMAL_CHANNEL = ObsChannel(NORMAL, drift=drift, var=variance)
COUNT_CHANNEL = ObsChannel(POISSON, intensity=intensity)


class ContinuousObservationsTests(unittest.TestCase):
    def generate(self, channels=(COUNT_CHANNEL, NORMAL_CHANNEL), seed=2026):
        return generate_continuous_observations(
            np.linspace(0, 1, 101), [0, 1, 0], [[0.2], [1.0], [-0.4]],
            [0.13, 0.17, 1.0], channels, seed=seed,
        )

    def test_reproducible_mixed_path_and_exact_events(self):
        a, b = self.generate(), self.generate()
        for name in ('times', 'continuous', 'counting', 'event_times', 'event_channels'):
            assert_array_equal(getattr(a, name), getattr(b, name))
        assert_array_equal(a.continuous_indices, [1])
        assert_array_equal(a.counting_indices, [0])
        self.assertGreater(len(a.event_times), 1)
        self.assertTrue(np.all(np.isin(a.event_times, a.times)))
        self.assertEqual(a.counting[-1, 0], len(a.event_times))
        assert_array_equal(a.counting[0], [0])
        assert_array_equal(a.continuous[0], [0])

    def test_sampling_telescopes_and_retains_absolute_values(self):
        path = self.generate()
        sampled = path.sample([0.2, 0.5, 1])
        assert_allclose(sampled.increments.sum(axis=0),
                        [sampled.counting[-1, 0] - sampled.counting[0, 0],
                         sampled.continuous[-1, 0] - sampled.continuous[0, 0]])
        assert_allclose(sampled.continuous[0], path.continuous[np.searchsorted(path.times, 0.2)])
        nested = path.sample([0, 0.5, 1], include_events=True)
        self.assertTrue(np.all(np.isin(path.event_times, nested.times)))
        assert_allclose(nested.increments.sum(axis=0), path.increments.sum(axis=0), atol=1e-14)
        self.assertTrue(np.all(np.diff(nested.counting[:, 0]) <= 1))

    def test_subset_lookup_tolerance_and_no_interpolation(self):
        path = self.generate((NORMAL_CHANNEL,))
        selected = path.sample([0.1 + 1e-16, 0.3 + 1e-16])
        assert_array_equal(selected.times, path.times[[10, 30]])
        for times in ([0, 0.12345], [-0.1, 1], [0, 1.1], [0.2, 0.1], [0, 0], [np.nan]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                path.sample(times)

    def test_more_hidden_segments_than_observations_integrates_exactly(self):
        ends = np.array([0.1, 0.2, 0.4, 0.8, 1.0])
        states = np.array([0, 1, 0, 1, 0])
        y = np.arange(5.)[:, None]
        path = generate_continuous_observations([0, 1], states, y, ends,
                                                [NORMAL_CHANNEL], seed=731)
        exact_mean = np.dot(np.diff(np.r_[0., ends]), y[:, 0] + states)
        exact_draw = exact_mean + np.sqrt(0.3) * np.random.default_rng(731).standard_normal()
        assert_allclose(path.continuous[-1, 0], exact_draw, atol=1e-14)

    def test_no_count_cap_and_state_dependent_rates(self):
        def rate(t, y, theta):
            return np.ones((len(y), 1)) * (50 if theta else 0)
        path = generate_continuous_observations(
            [0, 1], [0, 1, 0], [[0], [0], [0]], [0.2, 0.8, 1],
            [ObsChannel(POISSON, intensity=rate)], seed=27,
        )
        self.assertGreater(path.counting[-1, 0], 1)
        self.assertTrue(np.all((path.event_times >= 0.2) & (path.event_times <= 0.8)))

    def test_empty_channel_families(self):
        for channels, n_continuous, n_counting in (((NORMAL_CHANNEL,), 1, 0),
                                                  ((COUNT_CHANNEL,), 0, 1), ((), 0, 0)):
            path = self.generate(channels)
            self.assertEqual(path.continuous.shape, (len(path.times), n_continuous))
            self.assertEqual(path.counting.shape, (len(path.times), n_counting))
            self.assertEqual(path.increments.shape, (len(path.times) - 1, len(channels)))

    def test_simultaneous_and_endpoint_events_accumulate(self):
        path = ContinuousObservations(
            np.array([0., 0.25, 0.5, 1.]), np.empty((4, 0)),
            np.array([[0, 0], [2, 1], [2, 1], [3, 3]]),
            np.array([], dtype=int), np.array([0, 1]),
            np.array([0.25, 0.25, 0.25, 1., 1., 1.]), np.array([0, 0, 1, 0, 1, 1]),
        )
        sampled = path.sample([0, 1], include_events=True)
        assert_array_equal(sampled.times, [0, 0.25, 1])
        assert_array_equal(sampled.increments, [[2, 1], [1, 2]])
        rates = np.array([[[[1.], [2.]], [[3.], [1.]]]])
        f = ContinuousFilter([[0.5, 0.5]], [[0.5, 0.5]], [[[0.], [1.]]], rates,
                             1, [[0.]], 1., [1.], [COUNT_CHANNEL, COUNT_CHANNEL])
        for increment, dt in zip(sampled.increments, np.diff(sampled.times)):
            f.update(increment, dt=dt)
        likelihood = np.exp(-rates[0, :, :, 0].sum(axis=1)) * np.prod(rates[0, :, :, 0] ** [3, 3], axis=1)
        assert_allclose(f.psi[0], likelihood / likelihood.sum())

    def test_invalid_grids_and_latent_paths(self):
        arguments = dict(t_grid=[0, 1], theta=[0], y=[[0]], jump_ends=[1], channels=[NORMAL_CHANNEL])
        for key, value in [('t_grid', [0]), ('t_grid', [0, 0.5, 0.4]),
                           ('t_grid', [0.1, 1]), ('t_grid', [0, 2]),
                           ('jump_ends', [0]), ('jump_ends', [0.5, 0.5]),
                           ('theta', [0, 1]), ('theta', [0.5]), ('y', [[np.nan]])]:
            kwargs = dict(arguments)
            kwargs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generate_continuous_observations(**kwargs)

    def test_invalid_coefficients_and_generators(self):
        def variable_variance(t, y, theta):
            return np.ones((len(y), 1)) * (1 + theta)
        def negative_rate(t, y, theta):
            return -np.ones((len(y), 1))
        for channel in (ObsChannel(NORMAL, drift=drift, var=variable_variance),
                        ObsChannel(POISSON, intensity=negative_rate),
                        ObsChannel(NORMAL, drift=drift, var=variance, noise=lambda size: size)):
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                self.generate([channel])

    def test_local_rng_does_not_change_global_state(self):
        np.random.seed(721)
        before = np.random.get_state()
        self.generate()
        after = np.random.get_state()
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    unittest.main()
