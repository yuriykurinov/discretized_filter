"""Compare exact Pareto points, Gaussian sums, and sum/alternating-sum pairs.

CLT assumes a constant signal within each block. At jumps the alternating
sum's zero mean and its zero covariance with the direct sum are approximations.
"""

import _bootstrap  # noqa: F401

import argparse
from math import lcm

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from run_clt_averaging import point_observations, report, run_filter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', default='20,100', help='Comma-separated even block sizes')
    parser.add_argument('--hours', type=float, help='Signal duration in hours (default: Pareto T)')
    parser.add_argument('--exp-id', default='pareto_clt_two_channels_discrete',
                        help='Output experiment name')
    args = parser.parse_args()
    try:
        args.ns = [int(n) for n in args.ns.split(',')]
        if (not args.ns or any(n <= 0 or n % 2 for n in args.ns)
                or len(set(args.ns)) != len(args.ns)):
            raise ValueError
    except ValueError:
        parser.error('--ns requires distinct positive even integers')
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours must be positive and finite')
    return args


def main():
    args = parse_args()
    cfg = set_config('pareto_obs')
    horizon = cfg.T if args.hours is None else args.hours * 3600
    multiple = lcm(*args.ns)
    steps = int(horizon / cfg.ht) // multiple * multiple
    if steps == 0:
        raise ValueError('Horizon must contain at least lcm(ns) observation steps')
    t = np.arange(steps + 1) * cfg.ht
    regimes, states, jump_ends = sparse_mc(
        cfg.p0, cfg.Lambda, cfg.lam, horizon, cfg.get_y, cfg.y_intervals,
    )
    indices = np.minimum(np.searchsorted(jump_ends, t, side='right'), len(regimes) - 1)
    theta, y = regimes[indices], states[indices]
    # Match the first (Pareto) SeedSequence stream in run_clt_averaging.
    stream = np.random.SeedSequence(cfg.seed).spawn(1)[0]
    obs = point_observations('pareto', cfg, y[1:], np.random.default_rng(stream))
    archive = dict(t=t, theta=theta, y=y, obs=obs, ns=np.array(args.ns),
                   seed=cfg.seed, Lambda=cfg.Lambda, block_statistic='sum')
    th, yy = run_filter(cfg, obs, cfg.ht)
    archive['theta_exact'], archive['y_exact'] = th, yy
    print(f'{steps} point observations; horizon={t[-1]:g}s', flush=True)
    print('CLT assumes a constant block signal; zero alternating mean/covariance are approximate at jumps.')
    report('Pareto exact', th, yy, theta, y, cfg.N)
    clt_cfg = set_config('pareto_obs_clt')
    two_cfg = set_config('pareto_obs_approx_2ch')
    for n in args.ns:
        blocks = obs.reshape(-1, n, 1)
        sums = blocks.sum(axis=1)
        signs = (-1.0) ** np.arange(1, n + 1)
        alt = (blocks[:, :, 0] * signs).sum(axis=1)[:, None]
        archive[f't_n{n}'] = t[::n]
        archive[f'obs_sum_n{n}'], archive[f'obs_alt_n{n}'] = sums, alt
        th, yy = run_filter(clt_cfg, sums, n * cfg.ht, n=n)
        archive[f'theta_clt_n{n}'], archive[f'y_clt_n{n}'] = th, yy
        report(f'CLT sum n={n}', th, yy, theta[::n], y[::n], cfg.N)
        # Scale unit moments once in the shared helper; alt is already a sum.
        th, yy = run_filter(two_cfg, np.column_stack((sums, alt)), n * cfg.ht, n=n)
        archive[f'theta_two_n{n}'], archive[f'y_two_n{n}'] = th, yy
        report(f'CLT two channels n={n}', th, yy, theta[::n], y[::n], cfg.N)
    out_dir = saved_path_dir(args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'comparison.npz', **archive)
    print(f'Saved {out_dir / "comparison.npz"}')


if __name__ == '__main__':
    main()
