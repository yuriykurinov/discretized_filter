"""Checks for the HALF_NORMAL channel kind and the odd/|alternating even| aggregate."""

import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from discretized_filter.core.densities import (
    HALF_NORMAL, NORMAL, ObsChannel, half_normal_pdf, make_obs_density,
    normal_pdf,
)

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class HalfNormalDensityTest(unittest.TestCase):
    def test_integrates_to_one(self):
        var = 2.5
        value, _ = quad(lambda x: half_normal_pdf(x, var), 0.0, 60.0)
        self.assertAlmostEqual(value, 1.0, places=8)

    def test_matches_twice_normal(self):
        for x in (0.0, 0.3, 1.7, 4.0):
            assert_allclose(half_normal_pdf(x, 1.3), 2 * normal_pdf(x, 0.0, 1.3))

    def test_negative_is_zero(self):
        self.assertEqual(half_normal_pdf(-1e-9, 1.0), 0.0)

    def test_obs_density_linear_in_time(self):
        dens = make_obs_density((NORMAL, HALF_NORMAL))
        u = np.array([0.4, 0.6])
        comp = np.zeros((2, 2, 2))
        comp[:, 0, 0] = [1.0, 3.0]
        comp[:, 0, 1] = [0.5, 1.5]
        comp[:, 1, 1] = [2.0, 4.0]
        obs = np.array([1.2, 0.8])
        mu = 0.4 * 1.0 + 0.6 * 3.0
        v0 = 0.4 * 0.5 + 0.6 * 1.5
        v1 = 0.4 * 2.0 + 0.6 * 4.0
        expected = normal_pdf(1.2, mu, v0) * 2 * normal_pdf(0.8, 0.0, v1)
        assert_allclose(dens(obs, u, comp, 2), expected)
        obs[1] = -0.8
        self.assertEqual(dens(obs, u, comp, 2), 0.0)

    def test_channel_validation(self):
        f = lambda t, y, th: y
        ObsChannel(kind=HALF_NORMAL, var=f)
        with self.assertRaises(ValueError):
            ObsChannel(kind=HALF_NORMAL)
        with self.assertRaises(ValueError):
            ObsChannel(kind=HALF_NORMAL, var=f, drift=f)


class AggregateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import run_channel_clt_split_comparison as mod
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f'script not importable: {exc}')
        finally:
            sys.path.remove(str(SCRIPTS))
        cls.aggregate = staticmethod(mod.aggregate)

    def test_b4(self):
        obs = np.array([[1.0, 5.0, 2.0, 3.0]]).T
        out = self.aggregate(obs, 4)['odd_abs_2ch']
        # O = X1 + X3 = 3; W = X2 - X4 = 2
        assert_allclose(out, [[3.0, 2.0]])
        obs = np.array([[1.0, 2.0, 2.0, 7.0]]).T
        assert_allclose(self.aggregate(obs, 4)['odd_abs_2ch'], [[3.0, 5.0]])

    def test_b8(self):
        x = np.arange(1.0, 9.0)
        obs = np.concatenate([x, 2 * x])[:, None]
        out = self.aggregate(obs, 8)['odd_abs_2ch']
        # block 1: O = 1+3+5+7 = 16, W = 2-4+6-8 = -4 -> 4; block 2 doubles it
        assert_allclose(out, [[16.0, 4.0], [32.0, 8.0]])


class OddAbsConfigTest(unittest.TestCase):
    """Checks of the odd_abs_2ch configs: channel 2 is HALF_NORMAL with zero drift."""

    CONFIG_NAMES = (
        'pareto_obs_clt_odd_abs_2ch',
        'exponential_obs_clt_odd_abs_2ch',
        'uniform_obs_clt_odd_abs_2ch',
    )

    def test_channel_parameters(self):
        import discretized_filter.config
        for name in self.CONFIG_NAMES:
            with self.subTest(config=name):
                cfg = discretized_filter.config.set_config(name)
                self.assertEqual(cfg.K, 2)
                # M_net has shape (N, n_grid, M); column 1 holds y2 on each node
                self.assertEqual(cfg.M_net.shape[-1], 2)
                y2 = cfg.M_net[..., 1]
                # C has shape (N, n_grid, K, P); channel 2 (index 1) has zero drift
                self.assertTrue(np.allclose(cfg.C[..., 1, 0], 0))
                self.assertTrue(np.allclose(cfg.C[..., 1, 1], cfg.C[..., 0, 1]))
                # var = v/2 = y2**2/24 for both channels
                self.assertTrue(np.allclose(cfg.C[..., 0, 1], y2 ** 2 / 24.0))
                self.assertTrue(np.allclose(cfg.C[..., 1, 1], y2 ** 2 / 24.0))

    def test_get_obs_raises(self):
        import discretized_filter.config
        for name in self.CONFIG_NAMES:
            with self.subTest(config=name):
                cfg = discretized_filter.config.set_config(name)
                with self.assertRaises(ValueError):
                    cfg.get_obs(cfg.t_net_filtering, None, None, 0.0)


if __name__ == '__main__':
    unittest.main()
