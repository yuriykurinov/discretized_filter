"""Постоянный сигнал, одно состояние (N = 1): MSE(t) и квантили по R траекториям, 5 законов, 4 фильтра.

Законы шума tau (E tau = 1, D tau = 1/12): uniform, halfnormal, exponential,
pareto45, pareto25. Фильтры: exact (по каждому наблюдению) и ЦПТ-фильтры
full_1ch, full_2ch, odd_abs_2ch (агрегаты окон по B наблюдений) на одной и той же
реализации. Конфиги const1s_*; сетка --grid N1 [N2] общая для всех фильтров.
Истинные (p, q): фиксированы аргументом --y P Q либо (по умолчанию) случайны для
каждого пути: p ~ U[5,10], q ~ U[10,15] (интервалы из конфига), генерируются ПОСЛЕ
tau из того же ГПСЧ пути, поэтому поток tau одинаков в обоих режимах. RMSE каждого
пути считается относительно его собственной истины. Параллельно по задачам (закон, путь).
Архив компактный: theta не хранится; для каждого (закон, метод) и окна w -- MSE и
квантили 0.05/0.95 квадрата ошибки p и q по общему набору живых путей; оценки y_est
сохраняются только для первых --save-paths путей.
План: plans/20261125-constant-mc-rmse.md (дополнения 2026-11-26, 2026-11-27).
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
from run_channel_clt_constant import n_windows
from run_channel_clt_split_comparison import aggregate, check_configs, check_moments

LAWS = ('uniform', 'halfnormal', 'exponential', 'pareto45', 'pareto25')
CLT_METHODS = ('full_1ch', 'full_2ch', 'odd_abs_2ch')
METHODS = ('exact',) + CLT_METHODS
RAW_CONFIGS = {'uniform': 'const1s_uniform_exact',
               'halfnormal': 'const1s_halfnormal_exact',
               'exponential': 'const1s_exponential_exact',
               'pareto45': 'const1s_pareto45_exact',
               'pareto25': 'const1s_pareto25_exact'}
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


def sample_path(law, alpha, y_fixed, intervals, n_raw, rng):
    """Один путь: сначала tau (n_raw значений), затем (p, q) из того же rng.

    y_fixed -- None (случайные p ~ U[lo1, hi1], q ~ U[lo2, hi2]) или пара (p, q);
    при фиксированном Y rng после tau не используется. Возвращает X формы
    (n_raw, 1) и истину y_true формы (2,).
    """
    tau = sample_tau(law, alpha, n_raw, rng)
    if y_fixed is None:
        p = rng.uniform(*intervals[0])
        q = rng.uniform(*intervals[1])
        y_true = np.array([p, q])
    else:
        y_true = np.asarray(y_fixed, dtype=float)
    return (y_true[0] + y_true[1] * tau)[:, None], y_true


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--laws', default=','.join(LAWS))
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--n-paths', type=int, default=50)
    parser.add_argument('--hours', type=float, default=24.0)
    parser.add_argument('--grid', type=int, nargs='+', metavar='N', default=[50, 50],
                        help='число узлов по p и q для ВСЕХ фильтров: N1 [N2] '
                        '(одно число -- на обе координаты), каждое >= 2')
    parser.add_argument('--y', type=float, nargs=2, metavar=('P', 'Q'), default=None,
                        help='фиксированные истинные (p, q) для всех путей; '
                        'по умолчанию случайные для каждого пути из интервалов конфига')
    parser.add_argument('--block-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument('--workers', type=int, default=None,
                        help='по умолчанию SLURM_CPUS_PER_TASK или os.cpu_count(); '
                        '1 -- последовательно в этом процессе')
    parser.add_argument('--save-paths', type=int, default=3, metavar='K',
                        help='сколько первых путей (r = 0..K-1) сохранить с оценками '
                        'y_est в архиве, 0 <= K <= --n-paths')
    parser.add_argument('--exp-id', default='constant_mc_rmse_1s')
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
    if not 0 <= args.save_paths <= args.n_paths:
        parser.error('--save-paths должен быть в диапазоне 0..--n-paths')
    if args.checkpoint_every < 0:
        parser.error('--checkpoint-every должен быть неотрицательным')
    if args.workers is None:
        args.workers = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
    if args.workers < 1:
        parser.error('--workers должен быть >= 1')
    if len(args.grid) not in (1, 2) or any(g < 2 for g in args.grid):
        parser.error('--grid: одно или два целых числа, каждое >= 2')
    args.grid = [args.grid[0], args.grid[-1]]
    args.n_blocks = n_windows(args.hours, args.block_size)
    if args.n_blocks < 1:
        parser.error('--hours слишком мал для одного полного окна')
    return args


# --- воркер -----------------------------------------------------------------

_CFG_CACHE = {}


def _cfg(name, block_size, scale_ht, grid):
    """Конфиг с кешем на процесс; для ЦПТ-конфигов ht = B * raw_ht (raw_ht = 1).

    grid -- число узлов (N1, N2) по p и q, подставляется в num1 конфига."""
    grid = tuple(int(g) for g in grid)
    key = (name, block_size, scale_ht, grid)
    if key not in _CFG_CACHE:
        cfg = set_config(name, num1=list(grid))
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
    law, r, seed_seq, y_fixed, n_blocks, block_size, methods, grid = task
    start = monotonic()
    B = block_size
    raw = _cfg(RAW_CONFIGS[law], B, False, grid)
    alpha = float(raw.C[0, 0, 0, 2]) if law in PARETO_ALPHA else None
    intervals = [raw.y_intervals[0][0], raw.y_intervals[1][0]]
    obs, y_true = sample_path(law, alpha, y_fixed, intervals, n_blocks * B,
                              np.random.default_rng(seed_seq))
    aggs = aggregate(obs, B)
    out = {}
    for method in methods:
        if method == 'exact':
            out[method] = run_filter(raw, obs, B, B)
        else:
            cfg = _cfg(f'const1s_clt_{method}', B, True, grid)
            out[method] = run_filter(cfg, aggs[method], B, 1)
    elapsed = monotonic() - start
    print(f'[{law} path {r}] {elapsed:.1f} с; ' + ', '.join(
        f'{m}: {int(np.isfinite(v[0][:, 0]).sum()) - 1}/{n_blocks}' for m, v in out.items()),
        flush=True)
    return law, r, out, elapsed, y_true


# --- агрегация и сохранение -------------------------------------------------

QUANTILE_LEVELS = (('q05', 0.05), ('q95', 0.95))


def error_stats(y_est, methods, y_true):
    """MSE и квантили e^2 на общем наборе живых путей.

    y_est[m] [R, W+1, 2], y_true [R, 2]. alive_w -- пути, где ВСЕ методы дошли до w.
    Возвращает (stats, n_alive, n_reached): stats[(m, coord, key)] [W+1], coord in
    {'p', 'q'}, key in {'mse', 'q05', 'q95'}; NaN там, где alive_w пуст.
    """
    R, W1, _ = y_est[methods[0]].shape
    reached = {m: np.isfinite(y_est[m]).all(axis=2) for m in methods}
    alive = np.ones((R, W1), dtype=bool)
    for m in methods:
        alive &= reached[m]
    n_alive = alive.sum(axis=0)
    stats = {}
    for m in methods:
        for i, coord in enumerate(('p', 'q')):
            e2 = np.where(alive, (y_est[m][..., i] - y_true[:, None, i]) ** 2, np.nan)
            with warnings.catch_warnings():
                # окна, где все пути NaN, дают NaN без предупреждений
                warnings.simplefilter('ignore', RuntimeWarning)
                stats[(m, coord, 'mse')] = np.nanmean(e2, axis=0)
                for key, level in QUANTILE_LEVELS:
                    stats[(m, coord, key)] = np.nanquantile(e2, level, axis=0)
    n_reached = {m: reached[m].sum(axis=0) for m in methods}
    return stats, n_alive, n_reached


def build_payload(args, state, cfg_info, times, complete):
    K = args.save_paths
    payload = dict(cfg_info)
    payload.update(
        times=times,
        y_const=np.full(2, np.nan) if args.y is None else np.array(args.y),
        y_mode='random' if args.y is None else 'fixed',
        grid=np.array(args.grid), seed=args.seed,
        laws=np.array(args.law_list), methods=np.array(args.method_list),
        block_size=args.block_size, n_paths=args.n_paths, n_blocks=args.n_blocks,
        saved_paths=K,
        requested_hours=args.hours, workers=args.workers, filter_engine='DiscreteFilter',
        complete=complete)
    lines = []
    for law in args.law_list:
        st = state[law]
        stats, n_alive, n_reached = error_stats(st['y'], args.method_list, st['y_true'])
        for m in args.method_list:
            payload[f'y_est_{law}_{m}'] = st['y'][m][:K]
            payload[f'failure_{law}_{m}'] = np.array(st['failure'][m])
            payload[f'completed_{law}_{m}'] = st['completed'][m]
            payload[f'n_reached_{law}_{m}'] = n_reached[m]
            for coord in ('p', 'q'):
                payload[f'mse_{coord}_{law}_{m}'] = stats[(m, coord, 'mse')]
                for key, _ in QUANTILE_LEVELS:
                    payload[f'sqerr_{key}_{coord}_{law}_{m}'] = stats[(m, coord, key)]
        payload[f'y_true_{law}'] = st['y_true']
        payload[f'task_done_{law}'] = st['done']
        payload[f'task_seconds_{law}'] = st['seconds']
        payload[f'n_alive_{law}'] = n_alive
        lines.append(f'{law}: путей завершено {int(st["done"].sum())}/{args.n_paths}; '
                     f'живых на последнем окне {n_alive[-1]}')
        for m in args.method_list:
            nfail = sum(1 for f in st['failure'][m] if f)
            parts = []
            for coord in ('p', 'q'):
                mse = stats[(m, coord, 'mse')]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tail = np.nanmean(mse[1:]) if np.isfinite(mse[1:]).any() else np.nan
                parts.append(f'{coord}: MSE {mse[-1]:.6g} (sqrt {np.sqrt(mse[-1]):.6g}), '
                             f'среднее по w>=1 {tail:.6g} (sqrt {np.sqrt(tail):.6g})')
            lines.append(f'  {m:12s} отказов {nfail}; последнее окно / ' + '; '.join(parts))
    return payload, lines


def save_archive(out_dir, payload, lines, complete, args):
    tmp = out_dir / 'comparison.tmp.npz'
    np.savez(tmp, **payload)
    os.replace(tmp, out_dir / 'comparison.npz')
    head = ['ЗАВЕРШЕНО' if complete else 'НЕПОЛНЫЙ ПРОГОН',
            'Постоянный сигнал, одно состояние (N = 1).',
            f'Сетка (p, q): {args.grid[0]}x{args.grid[1]}; истинные (p, q): '
            + (f'фиксированы {list(args.y)}' if args.y is not None
               else 'случайные для каждого пути, p ~ U[5,10], q ~ U[10,15]') + '.',
            f'Окон: {args.n_blocks}; B={args.block_size}; путей на закон: {args.n_paths}; '
            f'seed={args.seed}.',
            'MSE и квантили (0.05, 0.95) квадрата ошибки на общем наборе путей, где все '
            'выбранные методы дошли до окна w; оценки хранятся для первых '
            f'{args.save_paths} путей.']
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
    grid = tuple(args.grid)
    clts = {m: _cfg(f'const1s_clt_{m}', B, True, grid) for m in CLT_METHODS}
    for law in args.law_list:
        raws[law] = raw = _cfg(RAW_CONFIGS[law], B, False, grid)
        if law in PARETO_ALPHA:
            alphas = np.unique(raw.C[..., 2])
            if not np.all(alphas == PARETO_ALPHA[law]):
                raise ValueError(f'{law}: alpha в конфиге {alphas} != {PARETO_ALPHA[law]}')
        check_moments(raw, clts, B)
        named.append((RAW_CONFIGS[law], raw))
        cfg_info[f'config_name_{law}'] = RAW_CONFIGS[law]
        cfg_info[f'config_source_{law}'] = raw._config_path.read_text(encoding='utf-8')
    for m, cfg in clts.items():
        named.append((f'const1s_clt_{m}', cfg))
        cfg_info[f'config_name_{m}'] = f'const1s_clt_{m}'
        cfg_info[f'config_source_{m}'] = cfg._config_path.read_text(encoding='utf-8')
    check_configs(named)
    ref = raws[args.law_list[0]]
    if ref.N != 1:
        raise ValueError('ожидается конфиг с одним состоянием (N = 1)')
    lo1, hi1 = ref.y_intervals[0][0]
    lo2, hi2 = ref.y_intervals[1][0]
    if args.y is not None and not (lo1 <= args.y[0] <= hi1 and lo2 <= args.y[1] <= hi2):
        raise SystemExit(f'--y {args.y} вне интервалов p in [{lo1}, {hi1}], '
                         f'q in [{lo2}, {hi2}]')
    W = args.n_blocks
    times = np.arange(W + 1) * B * ref.ht

    # сиды: порядок законов фиксирован (LAWS), путь r не зависит от R и набора законов
    law_ss = dict(zip(LAWS, np.random.SeedSequence(args.seed).spawn(len(LAWS))))
    tasks, spawn_keys = [], {}
    for law in args.law_list:
        path_ss = law_ss[law].spawn(R)
        spawn_keys[f'seed_spawn_keys_{law}'] = np.array([s.spawn_key for s in path_ss])
        tasks += [(law, r, path_ss[r], args.y, W, B, tuple(args.method_list), grid)
                  for r in range(R)]
    cfg_info.update(spawn_keys)

    state = {law: dict(
        y={m: np.full((R, W + 1, 2), np.nan) for m in args.method_list},
        failure={m: [''] * R for m in args.method_list},
        completed={m: np.full(R, -1, dtype=int) for m in args.method_list},
        y_true=np.full((R, 2), np.nan),
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
        law, r, out, elapsed, y_true = result
        st = state[law]
        st['y_true'][r] = y_true
        for m, (th, yy, failure) in out.items():
            st['y'][m][r] = yy  # theta (N = 1) не сохраняется
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
