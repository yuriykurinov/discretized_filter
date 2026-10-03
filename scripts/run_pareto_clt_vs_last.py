"""Compare Gaussian CLT block sums with exact Pareto block-end observations.

The CLT sum moments assume an approximately constant state within a block.
Both branches forecast over the same block interval and update once per block.
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.densities import NORMAL, PARETO, make_obs_density
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='pareto_clt_vs_last')
    parser.add_argument('--ht', type=float, help='Block interval in integer seconds')
    parser.add_argument('--hours', type=float, help='Requested source horizon in hours')
    parser.add_argument('--exp-id')
    args = parser.parse_args()
    if args.ht is not None and (
            not np.isfinite(args.ht) or args.ht <= 0 or not args.ht.is_integer()):
        parser.error('--ht must be a finite positive integer number of seconds')
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours must be finite and positive')
    return args


def validate_parameters(h, horizon):
    """Validate the interval and horizon before generating a source path."""
    if not np.isfinite(h) or h <= 0 or not float(h).is_integer():
        raise ValueError('The block interval must be a finite positive integer')
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('The horizon must be finite and positive')
    if horizon < h:
        raise ValueError('The horizon must contain at least one complete observation block')
    return int(h), int(horizon)


def build_clt_parameters(M_net, h):
    """Build Gaussian sum means and variances from the (p, q) grid."""
    C = np.empty((*M_net.shape[:2], 1, 2), dtype=float)
    C[:, :, 0, 0] = h * (M_net[:, :, 0] + M_net[:, :, 1])
    C[:, :, 0, 1] = h * M_net[:, :, 1] ** 2 / 12
    return C


def prepare_blocks(source_observations, h):
    """Retain complete blocks, returning sums, last points, and endpoint indices."""
    n_blocks = len(source_observations) // h
    used = n_blocks * h
    observations = source_observations[:used]
    return (
        observations.reshape(n_blocks, h, 1).sum(axis=1),
        observations[h - 1::h].copy(),
        np.arange(0, used + 1, h),
    )


def generate_source(cfg, horizon):
    """Generate one seeded reset-jump trajectory and one-second Pareto stream."""
    time_grid = np.arange(horizon + 1, dtype=float)
    regimes, states, jump_ends = sparse_mc(
        cfg.p0, cfg.Lambda, cfg.lam, horizon, cfg.get_y, cfg.y_intervals,
    )
    indices = np.minimum(
        np.searchsorted(jump_ends, time_grid, side='right'), len(regimes) - 1,
    )
    y_true = states[indices]
    alpha = cfg.C[0, 0, 0, 2]
    mean = alpha / (alpha - 1)
    std = np.sqrt(alpha / (alpha - 2)) / (alpha - 1)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(3)[0])
    tau = 1 + (rng.pareto(alpha, horizon) + 1 - mean) / (std * np.sqrt(12))
    observations = (y_true[1:, 0] + y_true[1:, 1] * tau)[:, None]
    return dict(
        source_time_grid=time_grid, source_observations=observations,
        source_theta_true=np.eye(cfg.N)[regimes[indices]], source_y_true=y_true,
    )


def run_filter(cfg, observations, h, C, obs_density):
    """Return completed posterior estimates, preserving a prefix on branch failure."""
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    filt = DiscreteFilter(
        cfg.p0[:, None] * pi, pi, cfg.M_net, C,
        cfg.N, cfg.Lambda, h, cfg.delta, obs_density,
    )
    theta, y = filt.estimate()
    theta_est, y_est = [theta], [y]
    status = 'complete'
    try:
        for observation in observations:
            filt.predict()
            filt.update(observation)
            theta, y = filt.estimate()
            if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(y)):
                raise FloatingPointError('Posterior estimate is not finite')
            theta_est.append(theta)
            y_est.append(y)
    except (ValueError, FloatingPointError) as exc:
        status = f'failed: {exc}'
    return dict(
        status=np.array(status), completed_steps=len(theta_est) - 1,
        theta_est=np.asarray(theta_est), y_est=np.asarray(y_est),
    )


def main():
    args = parse_args()
    # Load once so both branches use the same seeded source path and model.
    cfg = set_config(args.config)
    h, horizon = validate_parameters(
        cfg.ht if args.ht is None else args.ht,
        cfg.T if args.hours is None else args.hours * 3600,
    )
    if cfg.M != 2 or cfg.K != 1 or cfg.channels[0].kind != PARETO:
        raise ValueError('The source config must have a (p, q) grid and one Pareto channel')
    alpha = cfg.C[0, 0, 0, 2]
    if not np.all(cfg.C[..., 2] == alpha):
        raise ValueError('The source config must use one constant Pareto alpha')
    if (not np.array_equal(cfg.C[:, :, 0, 0], cfg.M_net[:, :, 0])
            or not np.array_equal(cfg.C[:, :, 0, 1], cfg.M_net[:, :, 1])):
        raise ValueError('The source config must use loc=p and scale=q')

    source = generate_source(cfg, horizon)
    clt_observations, last_observations, endpoints = prepare_blocks(
        source['source_observations'], h,
    )
    C_clt = build_clt_parameters(cfg.M_net, h)
    archive = dict(
        h=h, seed=cfg.seed, last_model='pareto', alpha=alpha,
        grid_shape=np.asarray(cfg.num_nodes), M_net=cfg.M_net, delta=cfg.delta,
        source_C=cfg.C, clt_C=C_clt, requested_horizon=(
            cfg.T if args.hours is None else args.hours * 3600),
        **source,
        time_grid=source['source_time_grid'][endpoints],
        theta_true=source['source_theta_true'][endpoints],
        y_true=source['source_y_true'][endpoints],
        clt_observations=clt_observations, last_observations=last_observations,
    )
    for label, observations, C, obs_density in (
            ('clt', clt_observations, C_clt, make_obs_density([NORMAL])),
            ('last', last_observations, cfg.C, cfg.obs_density)):
        result = run_filter(cfg, observations, h, C, obs_density)
        archive.update({f'{label}_{key}': value for key, value in result.items()})
        print(f'{label}: {result["status"]}; '
              f'{result["completed_steps"]}/{len(observations)} steps', flush=True)

    exp_id = args.exp_id if args.exp_id is not None else f'pareto_clt_vs_last_h={h:g}'
    out_dir = saved_path_dir(exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy without reloading the config and resetting the random generators.
    (out_dir / 'config_source.py').write_text(
        Path(cfg._config_path).read_text(encoding='utf-8'), encoding='utf-8',
    )
    np.savez(out_dir / 'comparison.npz', **archive)
    print(f'Saved {out_dir / "comparison.npz"}', flush=True)


if __name__ == '__main__':
    main()
