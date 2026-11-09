"""Tests for the constant-signal channel CLT script: marginals and CLI checks."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import _bootstrap  # noqa: E402,F401
import run_channel_clt_constant as script  # noqa: E402
from discretized_filter.config import set_config  # noqa: E402
from discretized_filter.core.filter_discrete import DiscreteFilter  # noqa: E402


class MarginalsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg = set_config('const2s_uniform_exact')
        pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
        filt = DiscreteFilter(cfg.p0[:, None] * pi, pi, cfg.M_net, cfg.C, cfg.N,
                              cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)
        rng = np.random.default_rng(1)
        for x in 7.5 + 12.5 * rng.uniform(0.5, 1.5, 3):
            filt.predict()
            filt.update(np.array([x]))
        cls.filt = filt
        cls.marg = script.compute_marginals(filt.psi, cfg.delta, cfg.nets)

    def test_integrates_to_one(self):
        cfg, m = self.cfg, self.marg
        self.assertAlmostEqual(m['theta'].sum(), 1.0, places=10)
        h1 = np.array([n[0][1] - n[0][0] for n in cfg.nets])
        h2 = np.array([n[1][1] - n[1][0] for n in cfg.nets])
        self.assertAlmostEqual((m['y1'] * h1[:, None]).sum(), 1.0, places=10)
        self.assertAlmostEqual((m['y2'] * h2[:, None]).sum(), 1.0, places=10)
        self.assertAlmostEqual(m['joint'].sum(), 1.0, places=10)

    def test_theta_matches_estimate(self):
        theta, y = self.filt.estimate()
        np.testing.assert_allclose(self.marg['theta'], theta, rtol=1e-12)
        n_y = self.marg['y1'] * np.array([n[0][1] - n[0][0] for n in self.cfg.nets])[:, None]
        np.testing.assert_allclose((n_y * self.marg['nodes_y1']).sum(), y[0], rtol=1e-10)

    def test_node_ordering(self):
        cfg, m = self.cfg, self.marg
        for n in range(cfg.N):
            grid = cfg.M_net[n].reshape(len(cfg.nets[n][0]), len(cfg.nets[n][1]), 2)
            np.testing.assert_array_equal(grid[:, 0, 0], m['nodes_y1'][n])
            np.testing.assert_array_equal(grid[0, :, 1], m['nodes_y2'][n])
            np.testing.assert_array_equal(grid[..., 0], m['nodes_y1'][n][:, None]
                                          * np.ones(len(m['nodes_y2'][n])))


class CliTest(unittest.TestCase):
    def parse(self, *argv):
        with mock.patch('sys.stderr'):
            return script.parse_args(list(argv))

    def test_default_y_is_midpoint(self):
        args = self.parse('--theta', '1')
        np.testing.assert_allclose(args.y_true, [12.5, 17.5])
        self.assertEqual(args.snap_windows, [args.n_blocks])

    def test_y_outside_theta_intervals_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse('--theta', '0', '--y', '12', '12')
        with self.assertRaises(SystemExit):
            self.parse('--theta', '1', '--y', '12', '12')
        self.assertEqual(self.parse('--theta', '1', '--y', '12', '17').theta, 1)

    def test_density_hours_validated(self):
        with self.assertRaises(SystemExit):
            self.parse('--hours', '1', '--density-hours', '2')
        args = self.parse('--hours', '0.0112', '--density-hours', '0.0056')
        self.assertEqual(args.snap_windows, [1])


if __name__ == '__main__':
    unittest.main()
