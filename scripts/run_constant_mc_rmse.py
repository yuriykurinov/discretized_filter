"""Постоянный сигнал, Lambda = 0: RMSE(t) по R траекториям, 5 законов, 4 фильтра.

Законы шума tau (E tau = 1, D tau = 1/12): uniform, halfnormal, exponential,
pareto45, pareto25. Фильтры: exact (по каждому наблюдению) и ЦПТ-фильтры
full_1ch, full_2ch, odd_abs_2ch (агрегаты окон по B наблюдений) на одной и той же
реализации. Theta и Y фиксированы аргументами. Параллельно по задачам (закон, путь).
План: plans/20261125-constant-mc-rmse.md.
"""
import os

# до импорта numba (воркеры 'spawn' заново импортируют этот модуль)
os.environ['NUMBA_NUM_THREADS'] = '1'

import _bootstrap  # noqa: F401

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from time import monotonic

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter_discrete import DiscreteFilter
from discretized_filter.paths import saved_path_dir
from run_channel_clt_constant import (
    Y1_INTERVALS, Y2_INTERVALS, n_windows, resolve_truth,
)
from run_channel_clt_split_comparison import aggregate, check_configs, check_moments

LAWS = ('uniform', 'halfnormal', 'exponential', 'pareto45', 'pareto25')
CLT_METHODS = ('full_1ch', 'full_2ch', 'odd_abs_2ch')
METHODS = ('exact',) + CLT_METHODS
RAW_CONFIGS = {'uniform': 'const2s_uniform_exact',
               'halfnormal': 'const2s_halfnormal_exact',
               'exponential': 'const2s_exponential_exact',
               'pareto45': 'const2s_pareto45_exact',
               'pareto25': 'const2s_pareto_exact'}
PARETO_ALPHA = {'pareto45': 4.5, 'pareto25': 2.5}


def sample_tau(law, alpha, size, rng):
    """tau с E tau = 1, D tau = 1/12 (вызовы rng как в run_clt_averaging)."""
    if law == 'uniform':
        return rng.uniform(0.5, 1.5, size)
    if law == 'halfnormal':
        m = np.sqrt(2 / np.pi)
        c = np.sqrt(12) * np.sqrt(1 - 2 / np.pi)
        return 1 + (np.abs(rng.standard_normal(size)) - m) / c
    if law == 'exponential':
        return 1 + (rng.exponential(size=size) - 1) / np.sqrt(12)
    if law in PARETO_ALPHA:
        mean = alpha / (alpha - 1)
        std = np.sqrt(alpha / (alpha - 2)) / (alpha - 1)
        return 1 + (rng.pareto(alpha, size) + 1 - mean) / (std * np.sqrt(12))
    raise ValueError(f'неизвестный закон {law}')


def sample_observations(law, alpha, y, n_raw, rng):
    """X_i = p + q*tau_i, форма (n_raw, 1)."""
    tau = sample_tau(law, alpha, n_raw, rng)
    return (y[0] + y[1] * tau)[:, None]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--laws', default=','.join(LAWS))
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--n-paths', type=int, default=50)
    parser.add_argument('--hours', type=float, default=24.0)
    parser.add_argument('--theta', type=int, choices=(0, 1), default=0)
    parser.add_argument('--y', type=float, nargs=2, metavar=('Y1', 'Y2'), default=None,
                        help='по умолчанию середины интервалов выбранного theta')
    parser.add_argument('--block-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument('--workers', type=int, default=None,
                        help='по умолчанию SLURM_CPUS_PER_TASK или os.cpu_count(); '
                        '1 -- последовательно в этом процессе')
    parser.add_argument('--exp-id', default='constant_mc_rmse')
    parser.add_argument('--overwrite', action='store_true',
                        help='разрешить перезапись существующего comparison.npz')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='частичное сохранение после каждых K задач (0 -- выкл.)')
    args = parser.parse_args(argv)

    def pick(text, allowed, flag):
        items = [s.strip() for s in text.split(',') if s.strip()]
        if not items or len(set(items)) != len(items) or any(s not in allowed for s in items):
            parser.error(f'{flag}: нужны неповторяющиеся имена из {allowed}')
        return [s for s in allowed if s in items]  # канонический порядок

    args.law_list = pick(args.laws, LAWS, '--laws')
    args.method_list = pick(args.methods, METHODS, '--methods')
    if args.block_size < 4 or args.block_size % 4:
        parser.error('--block-size должен быть положительным числом, кратным 4')
    if not np.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours должен быть конечным положительным числом')
    if args.n_paths < 1:
        parser.error('--n-paths должен быть >= 1')
    if args.checkpoint_every < 0:
        parser.error('--checkpoint-every должен быть неотрицательным')
    if args.workers is None:
        args.workers = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
    if args.workers < 1:
        parser.error('--workers должен быть >= 1')
    try:
        args.y_true = resolve_truth(args.theta, args.y)
    except ValueError as exc:
        parser.error(str(exc))
    args.n_blocks = n_windows(args.hours, args.block_size)
    if args.n_blocks < 1:
        parser.error('--hours слишком мал для одного полного окна')
    return args


# --- воркер -----------------------------------------------------------------

_CFG_CACHE = {}


def _cfg(name, block_size, scale_ht):
    """Конфиг с кешем на процесс; для ЦПТ-конфигов ht = B * raw_ht (raw_ht = 1)."""
    key = (name, block_size, scale_ht)
    if key not in _CFG_CACHE:
        cfg = set_config(name)
        if scale_ht:
            cfg.ht = block_size * 1.0  # B * raw_ht, raw_ht = 1 (проверяется check_moments)
        _CFG_CACHE[key] = cfg
    return _CFG_CACHE[key]


def run_filter(cfg, observations, block_size, window_step):
    """predict() затем update(); оценки на концах окон (window_step наблюдений).

    C слотов 0,1 умножается на B один раз для ЦПТ-методов (window_step == 1),
    для exact не масштабируется. Возвращает theta [W+1, N], y [W+1, 2], failure;
    после отказа остаток NaN.
    """
    pi = cfg.pi / np.sum(cfg.pi * cfg.delta[:, None], axis=1)[:, None]
    coefficients = cfg.C.copy()
    if window_step == 1:
        coefficients[..., :2] *= block_size
    filt = DiscreteFilter(cfg.p0[:, None] * pi, pi, cfg.M_net, coefficients,
                          cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density)
    n_win = len(observations) // window_step
    th0, y0 = filt.estimate()
    theta_est = np.full((n_win + 1, len(th0)), np.nan)
    y_est = np.full((n_win + 1, len(y0)), np.nan)
    theta_est[0], y_est[0] = th0, y0
    failure = ''
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
            break
        if step % window_step == 0:
            theta_est[step // window_step] = th
            y_est[step // window_step] = yy
    return theta_est, y_est, failure


def run_task(task):
    """Одна задача (закон, путь): генерирует X и прогоняет выбранные методы."""
    law, r, seed_seq, y_true, n_blocks, block_size, methods = task
    start = monotonic()
    B = block_size
    raw = _cfg(RAW_CONFIGS[law], B, False)
    alpha = float(raw.C[0, 0, 0, 2]) if law in PARETO_ALPHA else None
    obs = sample_observations(law, alpha, y_true, n_blocks * B, np.random.default_rng(seed_seq))
    aggs = aggregate(obs, B)
    out = {}
    for method in methods:
        if method == 'exact':
            out[method] = run_filter(raw, obs, B, B)
        else:
            cfg = _cfg(f'const2s_clt_{method}', B, True)
            out[method] = run_filter(cfg, aggs[method], B, 1)
    elapsed = monotonic() - start
    print(f'[{law} path {r}] {elapsed:.1f} с; ' + ', '.join(
        f'{m}: {int(np.isfinite(v[0][:, 0]).sum()) - 1}/{n_blocks}' for m, v in out.items()),
        flush=True)
    return law, r, out, elapsed


# --- агрегация и сохранение -------------------------------------------------

def rmse_curves(theta_est, y_est, methods, theta, y_true):
    """RMSE(t) на общем наборе живых путей; ключи (метод, величина) и n_alive."""
    R, W1, N = next(iter(theta_est.values())).shape
    alive = np.ones((R, W1), dtype=bool)
    for m in methods:
        alive &= (np.isfinite(theta_est[m]).all(axis=2) & np.isfinite(y_est[m]).all(axis=2))
    n_alive = alive.sum(axis=0)
    onehot = np.eye(N)[theta]
    curves = {}
    for m in methods:
        with np.errstate(invalid='ignore', divide='ignore'):
            e_th = np.where(alive, ((theta_est[m] - onehot) ** 2).mean(axis=2), np.nan)
            e1 = np.where(alive, (y_est[m][..., 0] - y_true[0]) ** 2, np.nan)
            e2 = np.where(alive, (y_est[m][..., 1] - y_true[1]) ** 2, np.nan)
            for key, e in (('theta', e_th), ('y1', e1), ('y2', e2)):
                # окна, где все пути NaN, дают NaN без предупреждений
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mean_e = np.nanmean(e, axis=0)
                curves[(m, key)] = np.where(n_alive > 0, np.sqrt(mean_e), np.nan)
    return curves, n_alive


def build_payload(args, state, cfg_info, times, complete):
    payload = dict(cfg_info)
    payload.update(
        times=times, theta_const=args.theta, y_const=args.y_true, seed=args.seed,
        laws=np.array(args.law_list), methods=np.array(args.method_list),
        block_size=args.block_size, n_paths=args.n_paths, n_blocks=args.n_blocks,
        requested_hours=args.hours, workers=args.workers, filter_engine='DiscreteFilter',
        complete=complete)
    lines = []
    for law in args.law_list:
        st = state[law]
        for m in args.method_list:
            payload[f'theta_est_{law}_{m}'] = st['theta'][m]
            payload[f'y_est_{law}_{m}'] = st['y'][m]
            payload[f'failure_{law}_{m}'] = np.array(st['failure'][m])
            payload[f'completed_{law}_{m}'] = st['completed'][m]
        payload[f'task_done_{law}'] = st['done']
        payload[f'task_seconds_{law}'] = st['seconds']
        curves, n_alive = rmse_curves(st['theta'], st['y'], args.method_list,
                                      args.theta, args.y_true)
        payload[f'n_alive_{law}'] = n_alive
        lines.append(f'{law}: путей завершено {int(st["done"].sum())}/{args.n_paths}; '
                     f'живых на последнем окне {n_alive[-1]}')
        for m in args.method_list:
            for key in ('theta', 'y1', 'y2'):
                payload[f'rmse_{key}_{law}_{m}'] = curves[(m, key)]
            payload[f'n_reached_{law}_{m}'] = np.isfinite(st['theta'][m][..., 0]).sum(axis=0)
            nfail = sum(1 for f in st['failure'][m] if f)
            with np.errstate(all='ignore'):
                mean_tail = [np.nanmean(curves[(m, k)][1:]) if args.n_blocks >= 1
                             and np.isfinite(curves[(m, k)][1:]).any() else np.nan
                             for k in ('theta', 'y1', 'y2')]
            last = [curves[(m, k)][-1] for k in ('theta', 'y1', 'y2')]
            lines.append(f'  {m:12s} отказов {nfail}; последнее окно RMSE '
                         f'theta/Y1/Y2 = ' + ' '.join(f'{v:.6g}' for v in last)
                         + '; среднее по w>=1 = ' + ' '.join(f'{v:.6g}' for v in mean_tail))
    return payload, lines


def save_archive(out_dir, payload, lines, complete, args):
    tmp = out_dir / 'comparison.tmp.npz'
    np.savez(tmp, **payload)
    os.replace(tmp, out_dir / 'comparison.npz')
    head = ['ЗАВЕРШЕНО' if complete else 'НЕПОЛНЫЙ ПРОГОН',
            f'Постоянный сигнал: theta={args.theta}, Y={args.y_true.tolist()}.',
            f'Окон: {args.n_blocks}; B={args.block_size}; путей на закон: {args.n_paths}; '
            f'seed={args.seed}.',
            'RMSE(t) на общем наборе путей, где все выбранные методы дошли до окна w.']
    summary = '\n'.join(head + lines) + '\n'
    (out_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    return summary


def main():
    args = parse_args()
    B, R = args.block_size, args.n_paths
    out_dir = saved_path_dir(args.exp_id)
    if (out_dir / 'comparison.npz').exists() and not args.overwrite:
        raise SystemExit(f'{out_dir / "comparison.npz"} уже существует; '
                         'используйте другой --exp-id или --overwrite')

    # проверки конфигов в главном процессе
    cfg_info, named = {}, []
    raws = {}
    clts = {m: _cfg(f'const2s_clt_{m}', B, True) for m in CLT_METHODS}
    for law in args.law_list:
        raws[law] = raw = _cfg(RAW_CONFIGS[law], B, False)
        if law in PARETO_ALPHA:
            alphas = np.unique(raw.C[..., 2])
            if not np.all(alphas == PARETO_ALPHA[law]):
                raise ValueError(f'{law}: alpha в конфиге {alphas} != {PARETO_ALPHA[law]}')
        check_moments(raw, clts, B)
        named.append((RAW_CONFIGS[law], raw))
        cfg_info[f'config_name_{law}'] = RAW_CONFIGS[law]
        cfg_info[f'config_source_{law}'] = raw._config_path.read_text(encoding='utf-8')
    for m, cfg in clts.items():
        named.append((f'const2s_clt_{m}', cfg))
        cfg_info[f'config_name_{m}'] = f'const2s_clt_{m}'
        cfg_info[f'config_source_{m}'] = cfg._config_path.read_text(encoding='utf-8')
    check_configs(named)
    ref = raws[args.law_list[0]]
    if (not np.array_equal(ref.y_intervals[0], Y1_INTERVALS)
            or not np.array_equal(ref.y_intervals[1], Y2_INTERVALS)):
        raise ValueError('Y1_INTERVALS/Y2_INTERVALS не совпадают с конфигами')
    N, W = ref.N, args.n_blocks
    times = np.arange(W + 1) * B * ref.ht

    # сиды: порядок законов фиксирован (LAWS), путь r не зависит от R и набора законов
    law_ss = dict(zip(LAWS, np.random.SeedSequence(args.seed).spawn(len(LAWS))))
    tasks, spawn_keys = [], {}
    for law in args.law_list:
        path_ss = law_ss[law].spawn(R)
        spawn_keys[f'seed_spawn_keys_{law}'] = np.array([s.spawn_key for s in path_ss])
        tasks += [(law, r, path_ss[r], args.y_true, W, B, tuple(args.method_list))
                  for r in range(R)]
    cfg_info.update(spawn_keys)

    state = {law: dict(
        theta={m: np.full((R, W + 1, N), np.nan) for m in args.method_list},
        y={m: np.full((R, W + 1, 2), np.nan) for m in args.method_list},
        failure={m: [''] * R for m in args.method_list},
        completed={m: np.full(R, -1, dtype=int) for m in args.method_list},
        done=np.zeros(R, dtype=bool), seconds=np.full(R, np.nan))
        for law in args.law_list}

    out_dir.mkdir(parents=True, exist_ok=True)
    for law in args.law_list:
        (out_dir / f'config_{law}.py').write_text(cfg_info[f'config_source_{law}'],
                                                  encoding='utf-8')
    for m in CLT_METHODS:
        (out_dir / f'config_{m}.py').write_text(cfg_info[f'config_source_{m}'],
                                                encoding='utf-8')

    def record(result):
        law, r, out, elapsed = result
        st = state[law]
        for m, (th, yy, failure) in out.items():
            st['theta'][m][r], st['y'][m][r] = th, yy
            st['failure'][m][r] = failure
            st['completed'][m][r] = int(np.isfinite(th[:, 0]).sum()) - 1
        st['done'][r] = True
        st['seconds'][r] = elapsed

    def save(complete):
        payload, lines = build_payload(args, state, cfg_info, times, complete)
        return save_archive(out_dir, payload, lines, complete, args)

    print(f'Задач: {len(tasks)}; воркеров: {args.workers}; окон: {W}', flush=True)
    finished = 0

    def after():
        nonlocal finished
        finished += 1
        if args.checkpoint_every and finished % args.checkpoint_every == 0 \
                and finished < len(tasks):
            save(False)

    if args.workers == 1:
        for task in tasks:
            record(run_task(task))
            after()
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(run_task, t) for t in tasks]
            try:
                for fut in as_completed(futures):
                    record(fut.result())
                    after()
            except Exception as exc:
                # при любом сбое (например, BrokenProcessPool) сохраняем частичный результат и пробрасываем ошибку
                print(f'Ошибка: {type(exc).__name__}: {exc}', flush=True)
                save(False)
                raise
    summary = save(True)
    print(summary)
    print(f'Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
