"""Compare Gaussian point filters with and without external outlier removal."""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ht', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--hours', type=float)
    parser.add_argument('--exp-id')
    args = parser.parse_args()
    if not np.isfinite(args.ht) or args.ht <= 0 or not args.ht.is_integer():
        parser.error('--ht must be a finite positive integer number of seconds')
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error('--threshold must be finite and nonnegative')
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours must be finite and positive')
    return args


def removal_mask(cfg, observations, threshold):
    """Compare the nearest grid mean with the largest grid standard deviation."""
    means = cfg.C[:, :, 0, 0]
    std = np.sqrt(cfg.C[:, :, 0, 1].max())
    return np.array([
        np.abs(value - means).min() > threshold * std
        for value in observations[:, 0]
    ])[:, None]


def run_filter(cfg, observations, removed):
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    filt = DiscreteFilter(
        cfg.p0[:, None] * pi, pi, cfg.M_net, cfg.C,
        cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density,
    )
    theta_est, y_est = [], []
    theta, y = filt.estimate()
    theta_est.append(theta)
    y_est.append(y)
    status = 'complete'
    try:
        for observation, skip in zip(observations, removed[:, 0]):
            filt.predict()
            if not skip:
                filt.update(observation)
            theta, y = filt.estimate()
            theta_est.append(theta)
            y_est.append(y)
    except (ValueError, FloatingPointError) as exc:
        status = f'failed: {exc}'
    done = len(theta_est) - 1
    return dict(
        status=np.array(status), completed_steps=done,
        theta_est=np.asarray(theta_est), y_est=np.asarray(y_est),
        observation_removed=removed[:done], forecast_only=removed[:done, 0],
    )


def main():
    args = parse_args()
    cfg = set_config('pareto_obs')
    cfg_clt = set_config('pareto_obs_clt')
    h = int(args.ht)
    horizon = cfg.T if args.hours is None else args.hours * 3600
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('The horizon must be finite and positive')
    n_blocks = int(horizon) // h
    if n_blocks == 0:
        raise ValueError('The horizon must contain at least one complete observation block')
    used = n_blocks * h
    cfg_clt.ht = h
    cfg_clt.C = np.array(cfg_clt.C, copy=True)
    # Sum moments assume the state is approximately constant within each block.
    cfg_clt.C[..., 0] *= h
    cfg_clt.C[..., 1] *= h
    raw_time_grid = np.arange(int(horizon) + 1, dtype=float)
    regimes, states, jump_ends = sparse_mc(
        cfg.p0, cfg.Lambda, cfg.lam, raw_time_grid[-1], cfg.get_y, cfg.y_intervals,
    )
    indices = np.minimum(
        np.searchsorted(jump_ends, raw_time_grid, side='right'), len(regimes) - 1,
    )
    y = states[indices[1:]]
    alpha = cfg.C[0, 0, 0, 2]
    mean = alpha / (alpha - 1)
    std = np.sqrt(alpha / (alpha - 2)) / (alpha - 1)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(3)[0])
    tau = 1 + (rng.pareto(alpha, len(y)) + 1 - mean) / (std * np.sqrt(12))
    source_observations = (y[:, 0] + y[:, 1] * tau)[:, None]
    observations = source_observations[:used].reshape(n_blocks, h, 1).sum(axis=1)
    time_grid = raw_time_grid[:used + 1:h]
    block_indices = indices[:used + 1:h]
    removed = removal_mask(cfg_clt, observations, args.threshold)
    archive = dict(
        threshold=args.threshold, h=h, ratio=h, seed=cfg.seed,
        observations=observations, source_observations=source_observations,
        source_time_grid=raw_time_grid,
        time_grid=time_grid, theta_true=np.eye(cfg.N)[regimes[block_indices]],
        y_true=states[block_indices],
    )
    for label, mask in (('baseline', np.zeros_like(removed)), ('thresholded', removed)):
        result = run_filter(cfg_clt, observations, mask)
        archive.update({f'{label}_{key}': value for key, value in result.items()})
        print(f'{label}: {result["status"]}; {result["completed_steps"]}/{len(observations)} steps; '
              f'{result["observation_removed"].sum()} removed', flush=True)
    exp_id = args.exp_id if args.exp_id is not None else f'pareto_observation_removal_h={h:g}'
    out_dir = saved_path_dir(exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'comparison.npz', **archive)
    print(f'Saved {out_dir / "comparison.npz"}')


if __name__ == '__main__':
    main()
