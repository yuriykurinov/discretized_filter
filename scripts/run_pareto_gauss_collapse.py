"""Pareto, постоянный сигнал: точный фильтр и точный фильтр с гауссовым размазыванием.

Методы: exact (без изменений) и exact_gauss (после каждой коррекции, если апостериорная
масса схлопнулась в одну ячейку сетки, psi заменяется дискретизованным гауссианом).
Оба метода работают по исходным наблюдениям, окно = B наблюдений, без масштабирования C.
План: plans/20261113-pareto-gauss-collapse.md.
"""
import _bootstrap  # noqa: F401

import argparse
from time import monotonic

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.paths import saved_path_dir
from run_channel_clt_comparison import CHANNELS
from run_channel_clt_constant import (
    Y1_INTERVALS, Y2_INTERVALS, resolve_truth, n_windows, compute_marginals,
)
from run_channel_clt_split_comparison import common_rmse, check_configs
from run_clt_averaging import point_observations

LAW = 'pareto'
METHODS = ('exact', 'exact_gauss')
SIGMA_FACTOR = 2 / 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument('--exp-id', default='pareto_gauss_collapse')
    parser.add_argument('--collapse-tol', type=float, default=1e-10,
                        help='схлопывание: максимальная масса ячейки >= 1 - tol')
    args = parser.parse_args(argv)
    if args.block_size < 4 or args.block_size % 4:
        parser.error('--block-size должен быть положительным числом, кратным 4')
    if not np.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours должен быть конечным положительным числом')
    if args.progress < 0 or args.checkpoint < 0:
        parser.error('--progress и --checkpoint должны быть неотрицательными')
    if not np.isfinite(args.collapse_tol) or not 0 < args.collapse_tol < 1:
        parser.error('--collapse-tol должен быть конечным числом из (0, 1)')
    try:
        args.y_true = resolve_truth(args.theta, args.y)
    except ValueError as exc:
        parser.error(str(exc))
    hours = [args.hours] if args.density_hours is None else args.density_hours
    if any(not np.isfinite(h) or h <= 0 or h > args.hours for h in hours):
        parser.error('--density-hours: каждое значение должно быть в (0, --hours]')
    args.n_blocks = n_windows(args.hours, args.block_size)
    if args.n_blocks < 1:
        parser.error('--hours слишком мал для одного полного окна')
    args.snap_windows = sorted({n_windows(h, args.block_size) for h in hours})
    if args.snap_windows[0] < 1:
        parser.error('--density-hours: момент раньше конца первого окна')
    return args


def collapse_cell(psi, delta, tol):
    """Ячейка с максимальной массой; (n, j, max_mass, collapsed)."""
    mass = psi * delta[:, None]
    n, j = np.unravel_index(np.argmax(mass), mass.shape)
    top = mass[n, j]
    return int(n), int(j), float(top), bool(top >= 1 - tol)


def gaussian_respread(psi, delta, M_net, nets, tol, sigma_factor=SIGMA_FACTOR):
    """Замена схлопнувшегося psi дискретизованным гауссианом.

    Схлопывание: max_{n,j} psi[n, j] * delta[n] >= 1 - tol. Тогда узел c = M_net[n*, j*],
    sigma_k = sigma_factor * h_k (шаг сетки состояния n*), координаты независимы,
    psi_new[n*, j] = g[j] / (delta[n*] * sum g), в остальных состояниях 0.
    Возвращает (new_psi или None, info); info -- словарь с n, j, max_mass, c, sigma.
    """
    n, j, top, collapsed = collapse_cell(psi, delta, tol)
    info = dict(collapsed=collapsed, n=n, j=j, max_mass=top)
    if not collapsed:
        return None, info
    c = M_net[n, j]
    sigma = np.array([sigma_factor * (nets[n][k][1] - nets[n][k][0]) for k in range(2)])
    g = np.exp(-(M_net[n, :, 0] - c[0]) ** 2 / (2 * sigma[0] ** 2)
               - (M_net[n, :, 1] - c[1]) ** 2 / (2 * sigma[1] ** 2))
    new_psi = np.zeros_like(psi)
    new_psi[n] = g / (delta[n] * g.sum())
    info.update(c=c.copy(), sigma=sigma)
    return new_psi, info


def run_filter(label, cfg, observations, block_size, snap_windows, progress,
               checkpoint, out_dir, gauss, tol):
    """Прогноз, коррекция и (для gauss) замена при схлопывании.

    Оценки и снимки psi берутся на концах окон (B наблюдений) после замены.
    Возвращает theta_est, y_est, failure, snaps, max_mass (n_windows+1, NaN после
    останова), reset_steps, first_collapse_step (первый шаг, на котором выполнен
    критерий; для gauss совпадает с первой заменой).
    """
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    filt = DiscreteFilter(cfg.p0[:, None] * pi, pi, cfg.M_net, cfg.C,
                          cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)
    th0, y0 = filt.estimate()
    theta_est, y_est = [th0], [y0]
    max_mass = [float((filt.psi * filt.delta[:, None]).max())]
    snaps, resets = {}, []
    first_collapse = -1
    failure = ''
    start = monotonic()
    total = len(observations)
    print(f'[{label}] DiscreteFilter: {total} наблюдений', flush=True)
    for step, obs in enumerate(observations, 1):
        try:
            filt.predict()
            filt.update(obs)
            if gauss:
                new_psi, info = gaussian_respread(filt.psi, filt.delta, cfg.M_net,
                                                  cfg.nets, tol)
                if new_psi is not None:
                    filt.psi = new_psi
                    resets.append(step)
            else:
                info = dict(collapsed=collapse_cell(filt.psi, filt.delta, tol)[3])
            if info['collapsed'] and first_collapse < 0:
                first_collapse = step
            if step % block_size == 0:
                th, yy = filt.estimate()
                if not np.all(np.isfinite(th)) or not np.all(np.isfinite(yy)):
                    raise FloatingPointError('Оценки содержат nan/inf')
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
            print(f'[{label}] ОСТАНОВ на шаге {step}: {failure}', flush=True)
            break
        if step % block_size == 0:
            window = step // block_size
            theta_est.append(th)
            y_est.append(yy)
            max_mass.append(float((filt.psi * filt.delta[:, None]).max()))
            if window in snap_windows:
                snaps[window] = filt.psi.copy()
        if progress and step % progress == 0:
            print(f'[{label}] {step}/{total}; {monotonic() - start:.1f} с; '
                  f'замен: {len(resets)}', flush=True)
        if checkpoint and step % checkpoint == 0:
            np.savez(out_dir / f'checkpoint_{label}.npz', filter_engine='DiscreteFilter',
                     step=step, requested_steps=total, complete=step == total,
                     theta_est=np.array(theta_est), y_est=np.array(y_est),
                     max_mass=np.array(max_mass), reset_steps=np.array(resets, dtype=int))
    print(f'[{label}] завершено {len(theta_est) - 1} окон; замен: {len(resets)}', flush=True)
    return (np.array(theta_est), np.array(y_est), failure, snaps,
            np.array(max_mass), np.array(resets, dtype=int), first_collapse)


def main():
    args = parse_args()
    B = args.block_size
    name = f'const2s_{LAW}_exact'
    cfg = set_config(name)
    check_configs([(name, cfg)])
    if (not np.array_equal(cfg.y_intervals[0], Y1_INTERVALS)
            or not np.array_equal(cfg.y_intervals[1], Y2_INTERVALS)):
        raise ValueError('Y1_INTERVALS/Y2_INTERVALS не совпадают с конфигом')
    n_blocks, n_raw = args.n_blocks, args.n_blocks * B
    snap_windows = args.snap_windows
    raw_times = np.arange(n_raw + 1) * cfg.ht
    times = raw_times[::B]
    theta_true = np.tile(np.eye(cfg.N)[args.theta], (n_blocks + 1, 1))
    y_true = np.tile(args.y_true, (n_blocks + 1, 1))
    y_raw = np.tile(args.y_true, (n_raw, 1))
    out_dir = saved_path_dir(args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = dict(zip(CHANNELS, np.random.SeedSequence(args.seed).spawn(len(CHANNELS))))
    stream = streams[LAW]
    source = cfg._config_path.read_text(encoding='utf-8')
    (out_dir / f'config_{LAW}_raw.py').write_text(source, encoding='utf-8')
    obs = point_observations(LAW, cfg, y_raw, np.random.default_rng(stream))
    if obs.shape != (n_raw, 1):
        raise ValueError(f'{LAW}: неверная форма исходных наблюдений {obs.shape}')
    payload = dict(theta_const=args.theta, y_const=args.y_true,
                   theta_true=theta_true, y_true=y_true,
                   times=times, raw_times=raw_times, N=cfg.N, T=cfg.T,
                   seed=args.seed, delta=cfg.delta, filter_engine='DiscreteFilter',
                   observation_generator='run_clt_averaging.point_observations',
                   channels=np.array([LAW]), methods=np.array(METHODS),
                   block_size=B, raw_ht=cfg.ht, filter_ht=B * cfg.ht,
                   requested_hours=args.hours, effective_horizon=times[-1],
                   n_raw=n_raw, n_blocks=n_blocks,
                   density_time=times[snap_windows],
                   density_windows=np.array(snap_windows),
                   collapse_tol=args.collapse_tol, sigma_factor=SIGMA_FACTOR,
                   config_name_pareto_raw=cfg._config_name,
                   config_source_pareto_raw=source,
                   seed_spawn_key_pareto=np.array(stream.spawn_key),
                   obs_pareto=obs)
    lines = ['Фильтр: DiscreteFilter (predict, update) по исходным наблюдениям, '
             'окно = B наблюдений, C без масштабирования.',
             f'exact_gauss: замена на гауссиан при max массе ячейки >= 1 - '
             f'{args.collapse_tol:g}, sigma = {SIGMA_FACTOR:.4f} * h.',
             'RMSE на общих завершённых концах окон; начальная точка исключена.',
             f'Постоянный сигнал: theta={args.theta}, Y={args.y_true.tolist()}.',
             f'Окон: {n_blocks}; горизонт: {times[-1]:g} с; снимки на окнах {snap_windows}.',
             f'B={B}, seed={args.seed}.',
             'канал метод точек RMSE_theta RMSE_Y1 RMSE_Y2']
    complete = True
    results = {}
    for method in METHODS:
        th, yy, failure, snaps, mm, resets, first = run_filter(
            f'{LAW}_{method}', cfg, obs, B, set(snap_windows), args.progress,
            args.checkpoint, out_dir, method == 'exact_gauss', args.collapse_tol)
        results[method] = th, yy
        done = len(th) - 1
        mm_full = np.full(n_blocks + 1, np.nan)
        mm_full[:len(mm)] = mm
        payload.update({f'failure_{LAW}_{method}': failure,
                        f'theta_est_{LAW}_{method}': th,
                        f'y_est_{LAW}_{method}': yy,
                        f'completed_windows_{LAW}_{method}': done,
                        f'complete_{LAW}_{method}': done == n_blocks,
                        f'max_mass_{method}': mm_full})
        if method == 'exact':
            payload['first_collapse_step_exact'] = first
        else:
            payload['reset_steps_exact_gauss'] = resets
            payload['n_resets_exact_gauss'] = len(resets)
        complete = complete and done == n_blocks
        lines.append(f'Статус {LAW}/{method}: {done}/{n_blocks} окон')
        shapes = compute_marginals(cfg.pi, cfg.delta, cfg.nets)
        keys = {'theta': 'marg_theta', 'y1': 'marg_y1', 'y2': 'marg_y2', 'joint': 'joint'}
        stacks = {k: np.full((len(snap_windows),) + shapes[k].shape, np.nan) for k in keys}
        for i, w in enumerate(snap_windows):
            if w in snaps:
                marg = compute_marginals(snaps[w], cfg.delta, cfg.nets)
                for k in keys:
                    stacks[k][i] = marg[k]
        for k, nm in keys.items():
            payload[f'{nm}_{LAW}_{method}'] = stacks[k]
        payload['nodes_y1'] = shapes['nodes_y1']
        payload['nodes_y2'] = shapes['nodes_y2']
    lines.append(f'exact: первое схлопывание на шаге {payload["first_collapse_step_exact"]}; '
                 f'exact_gauss: замен {payload["n_resets_exact_gauss"]}')
    count, scores = common_rmse(results, theta_true, y_true)
    payload[f'rmse_count_{LAW}'] = count
    for method, score in scores.items():
        payload[f'rmse_{LAW}_{method}'] = score
        lines.append(f'{LAW} {method} {count} ' + ' '.join(f'{v:.8g}' for v in score))
    payload['complete'] = complete
    lines.insert(0, 'ЗАВЕРШЕНО' if complete else 'НЕПОЛНЫЙ ПРОГОН')
    np.savez(out_dir / 'comparison.npz', **payload)
    summary = '\n'.join(lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print(f'Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
