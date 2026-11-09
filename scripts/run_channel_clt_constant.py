"""Постоянный сигнал, 2 состояния, Lambda = 0: точный фильтр и четыре ЦПТ-фильтра.

Методы: exact (по каждому исходному наблюдению), full_1ch, full_2ch, split_2ch,
odd_abs_2ch (агрегаты окон по B наблюдений). Theta и Y фиксируются аргументами.
Снимки маргинальных апостериорных плотностей p(theta), p(Y1), p(Y2) сохраняются
в заданные моменты (по умолчанию -- конец траектории).
План: plans/20261109-channel-clt-constant.md.
"""
import _bootstrap  # noqa: F401

import argparse
from time import monotonic

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.paths import saved_path_dir
from run_channel_clt_comparison import CHANNELS
from run_channel_clt_split_comparison import (
    aggregate, check_moments, common_rmse, check_configs,
)
from run_clt_averaging import point_observations

CLT_METHODS = ('full_1ch', 'full_2ch', 'split_2ch', 'odd_abs_2ch')
METHODS = ('exact',) + CLT_METHODS

# интервалы Y по состояниям (совпадают с configs/const2s_*.py; сверяются в main)
Y1_INTERVALS = np.array([[5., 10.], [10., 15.]])
Y2_INTERVALS = np.array([[10., 15.], [15., 20.]])


def resolve_truth(theta, y, y1_intervals=Y1_INTERVALS, y2_intervals=Y2_INTERVALS):
    """Возвращает Y (по умолчанию середины интервалов theta); Y обязан лежать в них."""
    if theta not in range(len(y1_intervals)):
        raise ValueError(f'theta должен быть из 0..{len(y1_intervals) - 1}')
    lo = np.array([y1_intervals[theta, 0], y2_intervals[theta, 0]])
    hi = np.array([y1_intervals[theta, 1], y2_intervals[theta, 1]])
    if y is None:
        return (lo + hi) / 2
    y = np.asarray(y, dtype=float)
    if y.shape != (2,) or not np.all(np.isfinite(y)) or np.any(y < lo) or np.any(y > hi):
        raise ValueError(f'Y={y.tolist()} вне интервалов состояния theta={theta}: '
                         f'Y1 in [{lo[0]:g}, {hi[0]:g}], Y2 in [{lo[1]:g}, {hi[1]:g}]')
    return y


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', default=','.join(CHANNELS))
    parser.add_argument('--hours', type=float, default=24.0)
    parser.add_argument('--theta', type=int, choices=(0, 1), default=0)
    parser.add_argument('--y', type=float, nargs=2, metavar=('Y1', 'Y2'), default=None,
                        help='по умолчанию середины интервалов выбранного theta')
    parser.add_argument('--block-size', type=int, default=20)
    parser.add_argument('--density-hours', type=float, nargs='+', default=None,
                        metavar='H', help='моменты снимков плотностей, часы '
                        '(округляются вниз до конца окна; по умолчанию --hours)')
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument('--progress', type=int, default=5000)
    parser.add_argument('--checkpoint', type=int, default=20000)
    parser.add_argument('--exp-id', default='channel_clt_constant')
    args = parser.parse_args(argv)
    args.order = [c.strip() for c in args.channels.split(',') if c.strip()]
    if (not args.order or len(set(args.order)) != len(args.order)
            or any(c not in CHANNELS for c in args.order)):
        parser.error(f'--channels: нужны неповторяющиеся имена из {CHANNELS}')
    if args.block_size < 4 or args.block_size % 4:
        parser.error('--block-size должен быть положительным числом, кратным 4')
    if not np.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours должен быть конечным положительным числом')
    if args.progress < 0 or args.checkpoint < 0:
        parser.error('--progress и --checkpoint должны быть неотрицательными')
    try:
        args.y_true = resolve_truth(args.theta, args.y)
    except ValueError as exc:
        parser.error(str(exc))
    hours = [args.hours] if args.density_hours is None else args.density_hours
    if any(not np.isfinite(h) or h <= 0 or h > args.hours for h in hours):
        parser.error('--density-hours: каждое значение должно быть в (0, --hours]')
    n_blocks = n_windows(args.hours, args.block_size)
    if n_blocks < 1:
        parser.error('--hours слишком мал для одного полного окна')
    args.n_blocks = n_blocks
    args.snap_windows = sorted({n_windows(h, args.block_size) for h in hours})
    if args.snap_windows[0] < 1:
        parser.error('--density-hours: момент раньше конца первого окна')
    return args


def n_windows(hours, block_size, raw_ht=1.0):
    return int(np.floor(hours * 3600 / (block_size * raw_ht) + 1e-12))


def compute_marginals(psi, delta, nets):
    """Маргиналы по плотности psi (N, G) на сетке cartesian_product(nets[n]).

    Узлы M_net[n] идут в C-порядке: Y1 -- медленная координата, Y2 -- быстрая,
    поэтому psi[n] приводится к (n1, n2). Возвращает theta (N,), joint
    (N, n1, n2) -- масса, y1 (N, n1), y2 (N, n2) -- плотности, nodes_y1/nodes_y2.
    """
    N = psi.shape[0]
    n1, n2 = len(nets[0][0]), len(nets[0][1])
    mass = psi * delta[:, None]
    joint = mass.reshape(N, n1, n2)
    h1 = np.array([nets[n][0][1] - nets[n][0][0] for n in range(N)])
    h2 = np.array([nets[n][1][1] - nets[n][1][0] for n in range(N)])
    return dict(
        theta=mass.sum(axis=1), joint=joint,
        y1=joint.sum(axis=2) / h1[:, None], y2=joint.sum(axis=1) / h2[:, None],
        nodes_y1=np.array([nets[n][0] for n in range(N)]),
        nodes_y2=np.array([nets[n][1] for n in range(N)]),
    )


def run_filter(label, cfg, observations, block_size, window_step, snap_windows,
               progress, checkpoint, out_dir):
    """Прогноз и коррекция DiscreteFilter.

    window_step -- число наблюдений в окне (1 для ЦПТ-агрегатов, B для exact);
    оценки и снимки psi берутся только на концах окон. C умножается на block_size
    один раз для ЦПТ-методов (window_step == 1) и не масштабируется для exact.
    """
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    coefficients = cfg.C.copy()
    if window_step == 1:
        coefficients[..., :2] *= block_size
    filt = DiscreteFilter(cfg.p0[:, None] * pi, pi, cfg.M_net, coefficients,
                          cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)
    th0, y0 = filt.estimate()
    theta_est, y_est = [th0], [y0]
    snaps = {}
    failure = ''
    start = monotonic()
    total = len(observations)
    print(f'[{label}] DiscreteFilter: {total} наблюдений', flush=True)
    for step, obs in enumerate(observations, 1):
        try:
            filt.predict()
            filt.update(obs)
            if step % window_step == 0:
                th, yy = filt.estimate()
                if not np.all(np.isfinite(th)) or not np.all(np.isfinite(yy)):
                    raise FloatingPointError('Оценки содержат nan/inf')
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
            print(f'[{label}] ОСТАНОВ на шаге {step}: {failure}', flush=True)
            break
        if step % window_step == 0:
            window = step // window_step
            theta_est.append(th)
            y_est.append(yy)
            if window in snap_windows:
                snaps[window] = filt.psi.copy()
        if progress and step % progress == 0:
            print(f'[{label}] {step}/{total}; {monotonic() - start:.1f} с', flush=True)
        if checkpoint and step % checkpoint == 0:
            np.savez(out_dir / f'checkpoint_{label}.npz', filter_engine='DiscreteFilter',
                     step=step, requested_steps=total, complete=step == total,
                     theta_est=np.array(theta_est), y_est=np.array(y_est))
    print(f'[{label}] завершено {len(theta_est) - 1} окон', flush=True)
    return np.array(theta_est), np.array(y_est), failure, snaps


def main():
    args = parse_args()
    B = args.block_size
    cfgs, named = {}, []
    for law in args.order:
        names = {'raw': f'const2s_{law}_exact',
                 **{m: f'const2s_clt_{m}' for m in CLT_METHODS}}
        cfgs[law] = {m: set_config(name) for m, name in names.items()}
        for method, cfg in cfgs[law].items():
            if method != 'raw':
                cfg.ht = B * cfgs[law]['raw'].ht
            named.append((names[method], cfg))
        check_moments(cfgs[law]['raw'], {m: cfgs[law][m] for m in CLT_METHODS}, B)
    check_configs(named)
    ref = cfgs[args.order[0]]['raw']
    if (not np.array_equal(ref.y_intervals[0], Y1_INTERVALS)
            or not np.array_equal(ref.y_intervals[1], Y2_INTERVALS)):
        raise ValueError('Y1_INTERVALS/Y2_INTERVALS не совпадают с конфигами')
    n_blocks, n_raw = args.n_blocks, args.n_blocks * B
    snap_windows = args.snap_windows
    raw_times = np.arange(n_raw + 1) * ref.ht
    times = raw_times[::B]
    theta_true = np.tile(np.eye(ref.N)[args.theta], (n_blocks + 1, 1))
    y_true = np.tile(args.y_true, (n_blocks + 1, 1))
    y_raw = np.tile(args.y_true, (n_raw, 1))
    out_dir = saved_path_dir(args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = dict(zip(CHANNELS, np.random.SeedSequence(args.seed).spawn(len(CHANNELS))))
    payload = dict(theta_const=args.theta, y_const=args.y_true,
                   theta_true=theta_true, y_true=y_true,
                   times=times, raw_times=raw_times, N=ref.N, T=ref.T,
                   seed=args.seed, delta=ref.delta, filter_engine='DiscreteFilter',
                   observation_generator='run_clt_averaging.point_observations',
                   channels=np.array(args.order), methods=np.array(METHODS),
                   block_size=B, raw_ht=ref.ht, filter_ht=B * ref.ht,
                   requested_hours=args.hours, effective_horizon=times[-1],
                   n_raw=n_raw, n_blocks=n_blocks,
                   density_time=times[snap_windows],
                   density_windows=np.array(snap_windows))
    lines = ['Фильтр: DiscreteFilter (predict, update); C ЦПТ-методов умножены на B '
             'один раз, exact без масштабирования.',
             'RMSE на общих завершённых концах окон; начальная точка исключена.',
             f'Постоянный сигнал: theta={args.theta}, Y={args.y_true.tolist()}.',
             f'Окон: {n_blocks}; горизонт: {times[-1]:g} с; снимки на окнах {snap_windows}.',
             f'Эффективные B={B}, ht={B * ref.ht:g}; seed={args.seed}.',
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
        obs = point_observations(law, raw, y_raw, np.random.default_rng(stream))
        payload[f'seed_spawn_key_{law}'] = np.array(stream.spawn_key)
        if obs.shape != (n_raw, 1):
            raise ValueError(f'{law}: неверная форма исходных наблюдений {obs.shape}')
        payload[f'obs_{law}'] = obs
        inputs = {'exact': obs, **aggregate(obs, B)}
        results = {}
        for method in METHODS:
            cfg = raw if method == 'exact' else cfgs[law][method]
            step = B if method == 'exact' else 1
            if method != 'exact':
                payload[f'obs_{law}_{method}'] = inputs[method]
            th, yy, failure, snaps = run_filter(
                f'{law}_{method}', cfg, inputs[method], B, step, set(snap_windows),
                args.progress, args.checkpoint, out_dir)
            results[method] = th, yy
            done = len(th) - 1
            payload.update({f'failure_{law}_{method}': failure,
                            f'theta_est_{law}_{method}': th,
                            f'y_est_{law}_{method}': yy,
                            f'completed_windows_{law}_{method}': done,
                            f'complete_{law}_{method}': done == n_blocks})
            complete = complete and done == n_blocks
            lines.append(f'Статус {law}/{method}: {done}/{n_blocks} окон')
            # снимки: NaN там, где фильтр не дошёл до момента снимка
            shapes = compute_marginals(cfg.pi, cfg.delta, cfg.nets)
            keys = ('theta', 'y1', 'y2', 'joint')
            stacks = {k: np.full((len(snap_windows),) + shapes[k].shape, np.nan)
                      for k in keys}
            for i, w in enumerate(snap_windows):
                if w in snaps:
                    marg = compute_marginals(snaps[w], cfg.delta, cfg.nets)
                    for k in keys:
                        stacks[k][i] = marg[k]
            for k in keys:
                name = {'theta': 'marg_theta', 'y1': 'marg_y1', 'y2': 'marg_y2',
                        'joint': 'joint'}[k]
                payload[f'{name}_{law}_{method}'] = stacks[k]
            payload['nodes_y1'] = shapes['nodes_y1']
            payload['nodes_y2'] = shapes['nodes_y2']
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
