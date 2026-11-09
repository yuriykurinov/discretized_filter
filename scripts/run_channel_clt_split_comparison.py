"""Четыре ЦПТ-фильтра на одной траектории и общих окнах исходных наблюдений.

Методы: full_1ch, full_2ch, split_2ch и odd_abs_2ch (сумма нечётных и модуль
знакопеременной суммы чётных; требует B % 4 == 0).
Планы: plans/20261029-channel-clt-split-comparison.md,
plans/20261109-channel-clt-odd-abs.md.
"""
import _bootstrap  # noqa: F401

import argparse
from time import monotonic

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from discretized_filter.utils.grids import set_seed
from run_channel_clt_comparison import (
    CHANNELS, _SHARED_SCALARS, _SHARED_ARRAYS, onehot_theta,
)

from run_clt_averaging import point_observations

METHODS = ('full_1ch', 'full_2ch', 'split_2ch', 'odd_abs_2ch')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', default=','.join(CHANNELS))
    parser.add_argument('--hours', type=float, default=None)
    parser.add_argument('--block-size', type=int, default=20)
    parser.add_argument('--progress', type=int, default=5000)
    parser.add_argument('--checkpoint', type=int, default=20000)
    parser.add_argument('--exp-id', default='channel_clt_split_abs_comparison')
    args = parser.parse_args()
    args.order = [c.strip() for c in args.channels.split(',') if c.strip()]
    if (not args.order or len(set(args.order)) != len(args.order)
            or any(c not in CHANNELS for c in args.order)):
        parser.error(f'--channels: нужны неповторяющиеся имена из {CHANNELS}')
    if args.block_size < 4 or args.block_size % 4:
        parser.error('--block-size должен быть положительным числом, кратным 4')
    if args.hours is not None and (not np.isfinite(args.hours) or args.hours <= 0):
        parser.error('--hours должен быть конечным положительным числом')
    if args.progress < 0 or args.checkpoint < 0:
        parser.error('--progress и --checkpoint должны быть неотрицательными')
    return args


def check_configs(named_cfgs):
    """Проверяет сигнал и сетки, разрешая разное число каналов K."""
    ref_name, ref = named_cfgs[0]
    for name, cfg in named_cfgs[1:]:
        for attr in _SHARED_SCALARS:
            if attr != 'K' and getattr(ref, attr) != getattr(cfg, attr):
                raise ValueError(f'{ref_name}, {name}: различается {attr}')
        for attr in _SHARED_ARRAYS + ('num_nodes', 'y_intervals'):
            if not np.array_equal(getattr(ref, attr), getattr(cfg, attr)):
                raise ValueError(f'{ref_name}, {name}: различается {attr}')


def check_moments(raw, filters, block_size):
    """Проверяет моменты после явного масштабирования C размером блока."""
    if raw.ht != 1 or raw.K != 1:
        raise ValueError('Ожидается один исходный канал с raw_ht=1')
    mu = raw.C[..., 0, 0] + raw.C[..., 0, 1]
    var = raw.C[..., 0, 1] ** 2 / 12
    expected = {
        'full_1ch': [(mu, var)],
        'full_2ch': [(mu, var), (np.zeros_like(mu), var)],
        'split_2ch': [(mu / 2, var / 2), (mu / 2, var / 2)],
        'odd_abs_2ch': [(mu / 2, var / 2), (np.zeros_like(mu), var / 2)],
    }
    for method, cfg in filters.items():
        if cfg.K != len(expected[method]) or cfg.ht != block_size * raw.ht:
            raise ValueError(f'{method}: неверное число каналов или шаг')
        for k, moments in enumerate(expected[method]):
            for p, moment in enumerate(moments):
                np.testing.assert_allclose(block_size * cfg.C[..., k, p],
                                           block_size * moment, rtol=1e-10)


def aggregate(obs, block_size):
    """Нечётность нумеруется с 1 внутри каждого неперекрывающегося блока."""
    blocks = obs[:, 0].reshape(-1, block_size)
    odd, even = blocks[:, ::2].sum(axis=1), blocks[:, 1::2].sum(axis=1)
    # W = X_2 - X_4 + X_6 - ... (чётные позиции с 1), знаки +,-,+,...
    signs = np.where(np.arange(blocks.shape[1] // 2) % 2 == 0, 1.0, -1.0)
    abs_w = np.abs(blocks[:, 1::2] @ signs)
    return {'full_1ch': (odd + even)[:, None],
            'full_2ch': np.column_stack((odd + even, even - odd)),
            'split_2ch': np.column_stack((odd, even)),
            'odd_abs_2ch': np.column_stack((odd, abs_w))}


def common_rmse(results, theta_true, y_true):
    """Общие завершённые концы окон; начальная априорная точка исключена."""
    count = min(len(r[0]) - 1 for r in results.values())
    sl = slice(1, count + 1)
    scores = {}
    for method, (theta_est, y_est) in results.items():
        if count:
            scores[method] = np.r_[
                np.sqrt(np.mean((theta_est[sl] - theta_true[sl]) ** 2)),
                np.sqrt(np.mean((y_est[sl] - y_true[sl]) ** 2, axis=0)),
            ]
        else:
            scores[method] = np.full(1 + y_true.shape[1], np.nan)
    return count, scores


def run_filter(label, cfg, observations, block_size, progress, checkpoint, out_dir):
    """Прогноз и коррекция DiscreteFilter с единичной экспозицией плотности."""
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    coefficients = cfg.C.copy()
    coefficients[..., :2] *= block_size
    filt = DiscreteFilter(cfg.p0[:, None] * pi, pi, cfg.M_net, coefficients,
                          cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)
    th0, y0 = filt.estimate()
    theta_est, y_est = [th0], [y0]
    failure = ''
    start = monotonic()
    total = len(observations)
    print(f'[{label}] DiscreteFilter: {total} окон', flush=True)
    for step, obs in enumerate(observations, 1):
        try:
            filt.predict()
            filt.update(obs)
            th, yy = filt.estimate()
            if not np.all(np.isfinite(th)) or not np.all(np.isfinite(yy)):
                raise FloatingPointError('Оценки содержат nan/inf')
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
            print(f'[{label}] ОСТАНОВ на окне {step}: {failure}', flush=True)
            break
        theta_est.append(th)
        y_est.append(yy)
        if progress and step % progress == 0:
            print(f'[{label}] {step}/{total}; {monotonic() - start:.1f} с', flush=True)
        if checkpoint and step % checkpoint == 0:
            np.savez(out_dir / f'checkpoint_{label}.npz', filter_engine='DiscreteFilter',
                     step=step, requested_windows=total, complete=step == total,
                     theta_est=np.array(theta_est), y_est=np.array(y_est))
    print(f'[{label}] завершено {len(theta_est) - 1}/{total} окон', flush=True)
    return np.array(theta_est), np.array(y_est), failure


def main():
    args = parse_args()
    cfgs, named = {}, []
    for law in args.order:
        names = {
            'raw': f'{law}_obs', 'full_1ch': f'{law}_obs_clt',
            'full_2ch': ('pareto_obs_approx_2ch' if law == 'pareto'
                         else f'{law}_obs_clt_2ch'),
            'split_2ch': f'{law}_obs_clt_split_2ch',
            'odd_abs_2ch': f'{law}_obs_clt_odd_abs_2ch',
        }
        cfgs[law] = {method: set_config(name) for method, name in names.items()}
        for method, cfg in cfgs[law].items():
            if method != 'raw':
                cfg.ht = args.block_size * cfgs[law]['raw'].ht
            named.append((names[method], cfg))
        check_moments(cfgs[law]['raw'],
                      {m: cfgs[law][m] for m in METHODS}, args.block_size)
    check_configs(named)
    ref = cfgs[args.order[0]]['raw']
    horizon = ref.T if args.hours is None else min(ref.T, args.hours * 3600)
    n_blocks = int(np.floor(horizon / (args.block_size * ref.ht) + 1e-12))
    if n_blocks < 1:
        raise ValueError('Горизонт слишком мал для одного полного окна')
    n_raw = n_blocks * args.block_size
    raw_times = np.arange(n_raw + 1) * ref.ht
    times = raw_times[::args.block_size]
    out_dir = saved_path_dir(args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(ref.seed)
    theta, y, t = sparse_mc(ref.p0, ref.Lambda, ref.lam, ref.T,
                            ref.get_y, ref.y_intervals)
    indices = np.minimum(np.searchsorted(t, times, side='right'), len(theta) - 1)
    theta_true, y_true = onehot_theta(theta[indices], ref.N), y[indices]
    raw_indices = np.minimum(np.searchsorted(t, raw_times[1:], side='right'), len(theta) - 1)
    streams = dict(zip(CHANNELS, np.random.SeedSequence(ref.seed).spawn(len(CHANNELS))))
    payload = dict(theta=theta, y=y, t=t, theta_true=theta_true, y_true=y_true,
                   times=times, raw_times=raw_times, N=ref.N, T=ref.T,
                   seed=ref.seed, delta=ref.delta, filter_engine='DiscreteFilter',
                   observation_generator='run_clt_averaging.point_observations',
                   channels=np.array(args.order),
                   methods=np.array(METHODS), block_size=args.block_size,
                   raw_ht=ref.ht, filter_ht=args.block_size * ref.ht,
                   requested_hours=np.nan if args.hours is None else args.hours,
                   effective_horizon=times[-1], n_raw=n_raw, n_blocks=n_blocks)
    lines = ['Фильтр: DiscreteFilter (predict, update); C умножены на B один раз.',
             'RMSE на общих завершённых концах окон; начальная точка исключена.',
             f'Запрошено окон: {n_blocks}; горизонт: {times[-1]:g} с.',
             f'Эффективные B={args.block_size}, ht={args.block_size * ref.ht:g}.',
             'Копии конфигов содержат исходные значения; B и ht переопределены CLI.',
             'канал метод точек RMSE_theta RMSE_Y1 RMSE_Y2']
    complete = True
    for law in args.order:
        for method, cfg in cfgs[law].items():
            source = cfg._config_path.read_text(encoding='utf-8')
            (out_dir / f'config_{law}_{method}.py').write_text(source, encoding='utf-8')
            payload[f'config_name_{law}_{method}'] = cfg._config_name
            payload[f'config_source_{law}_{method}'] = source
        raw = cfgs[law]['raw']
        stream = streams[law]
        obs = point_observations(law, raw, y[raw_indices], np.random.default_rng(stream))
        payload[f'seed_spawn_key_{law}'] = np.array(stream.spawn_key)
        if obs.shape != (n_raw, 1):
            raise ValueError(f'{law}: неверная форма исходных наблюдений {obs.shape}')
        payload[f'obs_{law}'] = obs
        aggregates = aggregate(obs, args.block_size)
        results = {}
        for method in METHODS:
            payload[f'obs_{law}_{method}'] = aggregates[method]
            th, yy, failure = run_filter(
                f'{law}_{method}', cfgs[law][method], aggregates[method],
                args.block_size, args.progress, args.checkpoint, out_dir)
            results[method] = th, yy
            payload[f'failure_{law}_{method}'] = failure
            done = len(th) - 1
            payload[f'theta_est_{law}_{method}'] = th
            payload[f'y_est_{law}_{method}'] = yy
            payload[f'completed_windows_{law}_{method}'] = done
            payload[f'complete_{law}_{method}'] = done == n_blocks
            complete = complete and done == n_blocks
            lines.append(f'Статус {law}/{method}: {done}/{n_blocks} окон')
        count, scores = common_rmse(results, theta_true, y_true)
        payload[f'rmse_count_{law}'] = count
        for method, score in scores.items():
            payload[f'rmse_{law}_{method}'] = score
            lines.append(f'{law} {method} {count} ' + ' '.join(f'{v:.8g}' for v in score))
    payload['complete'] = complete
    lines.insert(0, 'ЗАВЕРШЕНО' if complete else 'НЕПОЛНЫЙ ПРОГОН')
    np.savez(out_dir / 'comparison.npz', **payload)
    summary = '\n'.join(lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print(f'Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
