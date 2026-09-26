"""Compare three point laws with reset prediction and Gaussian block means.

CLT treats the signal as constant within a block, approximating blocks with jumps.
"""

import _bootstrap  # noqa: F401

import argparse
from math import lcm

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir


LAWS = ('pareto', 'exponential', 'uniform')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', default='1,10,20,50', help='Comma-separated block sizes')
    parser.add_argument('--hours', type=float, help='Signal duration in hours (default: Pareto T)')
    parser.add_argument('--exp-id', default='clt_averaging_discrete', help='Output experiment name')
    args = parser.parse_args()
    try:
        args.ns = [int(n) for n in args.ns.split(',')]
        if not args.ns or min(args.ns) <= 0 or len(set(args.ns)) != len(args.ns):
            raise ValueError
    except ValueError:
        parser.error('--ns requires distinct positive integers')
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours must be positive and finite')
    return args


def point_observations(law, cfg, y, rng):
    """Draw independent X=p+q*tau with E[tau]=1 and Var[tau]=1/12."""
    size = len(y)
    if law == 'pareto':
        alpha = cfg.C[0, 0, 0, 2]
        mean = alpha / (alpha - 1)
        std = np.sqrt(alpha / (alpha - 2)) / (alpha - 1)
        tau = 1 + (rng.pareto(alpha, size) + 1 - mean) / (std * np.sqrt(12))
    elif law == 'exponential':
        tau = 1 + (rng.exponential(size=size) - 1) / np.sqrt(12)
    else:
        tau = rng.uniform(0.5, 1.5, size)
    return (y[:, 0] + y[:, 1] * tau)[:, None]


def run_filter(cfg, observations, dt, n=None):
    """Predict then correct, retaining the normalized initial prior at t=0."""
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    C = cfg.C.copy()
    if n is not None:
        # Point likelihood has unit exposure: only block-mean variance changes.
        C[..., 1] /= n
    filt = DiscreteFilter(
        cfg.p0[:, None] * pi, pi, cfg.M_net, C, cfg.N,
        cfg.Lambda, dt, cfg.delta, cfg.obs_density,
    )
    theta = np.empty((len(observations) + 1, cfg.N))
    y = np.empty((len(observations) + 1, cfg.M))
    theta[0], y[0] = filt.estimate()
    for i, obs in enumerate(observations, 1):
        filt.predict()
        filt.update(obs)
        theta[i], y[i] = filt.estimate()
    return theta, y


def report(label, theta, y, true_theta, true_y, N):
    truth = np.eye(N)[true_theta[1:]]
    theta_rmse = np.sqrt(np.mean((theta[1:] - truth) ** 2))
    y_rmse = np.sqrt(np.mean((y[1:] - true_y[1:]) ** 2, axis=0))
    print(f'{label:24s} RMSE theta={theta_rmse:.4f} p={y_rmse[0]:.4f} q={y_rmse[1]:.4f}', flush=True)


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
    archive = dict(t=t, theta=theta, y=y, laws=np.array(LAWS),
                   ns=np.array(args.ns), seed=cfg.seed, Lambda=cfg.Lambda)
    streams = np.random.SeedSequence(cfg.seed).spawn(len(LAWS))
    print(f'{steps} point observations per law; horizon={t[-1]:g}s', flush=True)
    print('CLT assumes a constant signal within each block, including blocks crossing jumps.')
    for law, stream in zip(LAWS, streams):
        exact_cfg = set_config(f'{law}_obs')
        clt_cfg = set_config(f'{law}_obs_approx')
        obs = point_observations(law, exact_cfg, y[1:], np.random.default_rng(stream))
        archive[f'obs_{law}'] = obs
        th, yy = run_filter(exact_cfg, obs, cfg.ht)
        archive[f'theta_{law}_exact'], archive[f'y_{law}_exact'] = th, yy
        report(f'{law} exact', th, yy, theta, y, cfg.N)
        for n in args.ns:
            means = obs.reshape(-1, n, 1).mean(axis=1)
            th, yy = run_filter(clt_cfg, means, n * cfg.ht, n=n)
            archive[f't_n{n}'] = t[::n]
            archive[f'obs_{law}_n{n}'] = means
            archive[f'theta_{law}_clt_n{n}'] = th
            archive[f'y_{law}_clt_n{n}'] = yy
            report(f'{law} CLT n={n}', th, yy, theta[::n], y[::n], cfg.N)
    out_dir = saved_path_dir(args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'comparison.npz', **archive)
    print(f'Saved {out_dir / "comparison.npz"}')


if __name__ == '__main__':
    main()
