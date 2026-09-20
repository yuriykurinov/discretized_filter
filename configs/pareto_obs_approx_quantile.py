"""Gaussian approximation for a quantile-selected Pareto observation.

The channel uses ``W = X * 1{X <= U}``, where the fixed threshold ``U`` is
computed from the widest conditional Pareto distribution on the state grid.
The drift and variance below are unconditional moments of ``W`` and therefore
include the atom at zero created by rejected observations.
"""
import os

import numpy as np
from numba import njit

from discretized_filter.core.densities import ObsChannel, NORMAL

exp_id = 'pareto_obs_approx_quantile'

ALPHA = 2.5
# The environment override lets a notebook set the experiment parameter before
# loading the config; direct runs can edit this default instead.
QUANTILE = float(os.environ.get('DFILTER_PARETO_QUANTILE', '0.999'))

if not 0.0 < QUANTILE < 1.0:
    raise ValueError(f'QUANTILE must satisfy 0 < QUANTILE < 1, got {QUANTILE}')

two_jumps = False

n_points = 2

T = 24 * 3600

# Moments in C are for one original observation. A caller that aggregates a
# block of size n must instantiate Filter with ht=n.
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
    f = 1.0 - p_u**(-ALPHA)
    m1 = ALPHA / (ALPHA - 1.0) * (1.0 - p_u**(1.0 - ALPHA))
    return a * f + b * m1


@njit(nogil=True)
def var(t, y, theta):
    loc = y[:, 0:1]
    scale = y[:, 1:2]
    a = loc + scale * _C
    b = scale * _D
    p_u = (THRESHOLD - a) / b
    f = 1.0 - p_u**(-ALPHA)
    m1 = ALPHA / (ALPHA - 1.0) * (1.0 - p_u**(1.0 - ALPHA))
    m2 = ALPHA / (ALPHA - 2.0) * (1.0 - p_u**(2.0 - ALPHA))
    mean = a * f + b * m1
    second = a**2 * f + 2.0 * a * b * m1 + b**2 * m2
    return second - mean**2


channels = [
    ObsChannel(kind=NORMAL, drift=drift, var=var),
]
