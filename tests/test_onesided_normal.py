"""Tests for the ONESIDED_NORMAL observation density and its configs."""

import math

import unittest

import numpy as np
from scipy import integrate, stats

from discretized_filter.config import set_config
from discretized_filter.core.densities import (
    ONESIDED_NORMAL, ObsChannel, make_obs_density, onesided_normal_obs_pdf,
)

M = math.sqrt(2.0 / math.pi)
CC = math.sqrt(12.0) * math.sqrt(1.0 - 2.0 / math.pi)
LOW = 1.0 - M / CC


def _pdf(x, loc=0.0, scale=1.0):
    return onesided_normal_obs_pdf(float(x), loc, scale)


class OnesidedNormalTest(unittest.TestCase):
    def test_integrates_to_one_with_moments(self):
        kw = dict(limit=200, epsabs=1e-12, epsrel=1e-12)
        total = integrate.quad(_pdf, LOW, 6.0, **kw)[0]
        mean = integrate.quad(lambda x: x * _pdf(x), LOW, 6.0, **kw)[0]
        var = integrate.quad(lambda x: (x - 1.0) ** 2 * _pdf(x), LOW, 6.0, **kw)[0]
        self.assertAlmostEqual(total, 1.0, delta=1e-8)
        self.assertAlmostEqual(mean, 1.0, delta=1e-8)
        self.assertAlmostEqual(var, 1.0 / 12.0, delta=1e-8)
    def test_zero_below_support(self):
        self.assertAlmostEqual(LOW, 0.618, delta=1e-3)
        assert _pdf(LOW - 1e-6) == 0.0
        assert _pdf(-5.0) == 0.0
        assert _pdf(LOW + 1e-6) > 0.0
        assert _pdf(1.0, 0.0, 0.0) == 0.0
        assert _pdf(1.0, 0.0, -1.0) == 0.0


    def test_point_value_hand_formula(self):
        loc, scale, x = 3.0, 2.0, 4.5
        a = M + CC * ((x - loc) / scale - 1.0)
        expected = CC / scale * 2.0 * math.exp(-a * a / 2.0) / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(_pdf(x, loc, scale), expected, delta=1e-12 * expected)
        # |Z| = a, so the density equals 2*phi(a) * |da/dx|
        self.assertAlmostEqual(
            _pdf(x, loc, scale), 2.0 * stats.norm.pdf(a) * CC / scale,
            delta=1e-12 * expected)


    def test_sampler_matches_pdf(self):
        rng = np.random.default_rng(20260803)
        z = rng.standard_normal(200_000)
        tau = 1.0 + (np.abs(z) - M) / CC
        self.assertAlmostEqual(tau.mean(), 1.0, delta=5e-3)
        self.assertAlmostEqual(tau.var(), 1.0 / 12.0, delta=2e-3)
        grid = np.linspace(LOW, 3.0, 4001)
        pdf = np.array([_pdf(g) for g in grid])
        cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0.0)
        ks = stats.kstest(tau, lambda q: np.interp(q, grid, cdf))
        assert ks.statistic < 0.005

        edges = np.linspace(LOW, 2.0, 31)
        hist, _ = np.histogram(tau, bins=edges)
        probs = np.diff(np.interp(edges, grid, cdf))
        expected = probs * tau.size
        assert np.all(np.abs(hist - expected) < 5 * np.sqrt(expected))


    def test_obs_density_matches_pdf(self):
        od = make_obs_density((ONESIDED_NORMAL,))
        comp = np.array([[[2.0, 1.5]]])
        u = np.array([0.7])
        for x in (2.5, 3.0, 3.4, 1.0, 10.0):
            got = od(np.array([x]), u, comp, 1)
            self.assertAlmostEqual(got, _pdf(x, 2.0, 1.5), delta=1e-12 * _pdf(x, 2.0, 1.5))
    def test_obs_density_mixture(self):
        od = make_obs_density((ONESIDED_NORMAL,))
        comp = np.array([[[0.0, 1.0]], [[0.5, 2.0]]])
        u = np.array([1.0, 3.0])
        x = 1.2
        expected = 0.25 * _pdf(x, 0.0, 1.0) + 0.75 * _pdf(x, 0.5, 2.0)
        self.assertAlmostEqual(od(np.array([x]), u, comp, 2), expected, delta=1e-12 * expected)
    def test_obschannel_validation(self):
        f = lambda t, y, th: y
        ObsChannel(kind=ONESIDED_NORMAL, loc=f, scale=f)
        with self.assertRaises(ValueError):
            ObsChannel(kind=ONESIDED_NORMAL, loc=f)
        with self.assertRaises(ValueError):
            ObsChannel(kind=ONESIDED_NORMAL, loc=f, scale=f, alpha=f)


    def test_halfnormal_config(self):
        cfg = set_config('const2s_halfnormal_exact')
        C = cfg.C
        assert C.shape[2:] == (1, 2)
        assert np.allclose(C[0, :, 0, 0], cfg.M_net[0][:, 0])
        assert np.allclose(C[0, :, 0, 1], cfg.M_net[0][:, 1])
        assert np.all(C[:, :, 0, 1] > 0)


    def test_pareto45_config(self):
        cfg = set_config('const2s_pareto45_exact')
        C = cfg.C
        assert C.shape[2:] == (1, 3)
        assert np.allclose(C[:, :, 0, 2], 4.5)
        assert np.allclose(C[0, :, 0, 0], cfg.M_net[0][:, 0])
        assert np.allclose(C[0, :, 0, 1], cfg.M_net[0][:, 1])


if __name__ == '__main__':
    unittest.main()
