"""Gaussian approximation for an upper-winsorized Pareto observation.

The channel uses ``W = min(X, U)``, where the fixed threshold ``U`` is
computed from the widest conditional Pareto distribution on the state grid.
The drift and variance below are the first two moments of one winsorized
observation.  A caller aggregating blocks of size ``n`` must instantiate the
filter with ``ht=float(n)``.
"""
import os

import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL


exp_id = 'pareto_obs_approx_winsorized'

ALPHA = 2.5
QUANTILE = float(os.environ.get('DFILTER_PARETO_QUANTILE', '0.999'))

if not 0.0 < QUANTILE < 1.0:
    raise ValueError(f'QUANTILE must satisfy 0 < QUANTILE < 1, got {QUANTILE}')

two_jumps = False
n_points = 2

T = 24 * 3600
ht = 1.0
seed = 20260803
N = 4

Lambda = np.full((4, 4), 1.0 / (3 * 3600))
np.fill_diagonal(Lambda, 0.0)

y1_intervals = np.array([[1., 5.], [5., 10.], [10., 15.], [15., 20.]])
y2_intervals = np.array([[10., 20.], [20., 30.], [30., 40.], [40., 50.]])
y_intervals = [y1_intervals, y2_intervals]
num1 = [50, 50]
pi_family = 'uniform'

_PARETO_MEAN = ALPHA / (ALPHA - 1.0)
_PARETO_SD = np.sqrt(ALPHA / (ALPHA - 2.0)) / (ALPHA - 1.0)
_C = 1.0 - _PARETO_MEAN / (_PARETO_SD * np.sqrt(12.0))
_D = 1.0 / (_PARETO_SD * np.sqrt(12.0))

LOC_MAX = float(np.max(y1_intervals))
SCALE_MAX = float(np.max(y2_intervals))
THRESHOLD = LOC_MAX + SCALE_MAX * (
    _C + _D * (1.0 - QUANTILE)**(-1.0 / ALPHA)
)


@njit(nogil=True)
def drift(t, y, theta):
    loc = y[:, 0:1]
    scale = y[:, 1:2]
    a = loc + scale * _C
    b = scale * _D
    p_u = (THRESHOLD - a) / b
    p_safe = np.maximum(p_u, 1.0)
    m1 = (ALPHA - p_safe**(1.0 - ALPHA)) / (ALPHA - 1.0)
    mean = a + b * m1
    return np.where(p_u <= 1.0, THRESHOLD, mean)


@njit(nogil=True)
def var(t, y, theta):
    loc = y[:, 0:1]
    scale = y[:, 1:2]
    a = loc + scale * _C
    b = scale * _D
    p_u = (THRESHOLD - a) / b
    p_safe = np.maximum(p_u, 1.0)
    m1 = (ALPHA - p_safe**(1.0 - ALPHA)) / (ALPHA - 1.0)
    m2 = (ALPHA - 2.0 * p_safe**(2.0 - ALPHA)) / (ALPHA - 2.0)
    variance = b**2 * np.maximum(m2 - m1**2, 0.0)
    return np.where(p_u <= 1.0, 0.0, variance)


channels = [
    ObsChannel(kind=NORMAL, drift=drift, var=var),
]
