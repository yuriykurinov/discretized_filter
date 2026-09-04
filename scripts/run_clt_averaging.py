"""
ЦПТ-фильтры Парето при разных объёмах осреднения n = 1, 10, 20, 50 против
точного фильтра -- на ОДНОЙ траектории (theta, Y) и ОДНОМ потоке наблюдений
с шагом ht=1. Обобщение ``scripts/run_pareto_comparison.py`` (там объём
блока фиксирован и равен ``ht`` конфига approx/clt) на набор объёмов
осреднения ``--ns``. Ноутбучный прототип приёма -- см.
``notebooks/pareto_three_way_comparison.ipynb``, ячейки 5-9.

  конфиг              ht   правдоподобие шага                наблюдения
  ------------------- ---- --------------------------------- ---------------------
  pareto_obs           1   точная плотность Y1 + Y2*eps      исходные, шаг 1
  pareto_obs_approx   10   гауссовская по двум моментам      блоки по n, сумма

Ключевой приём: один конфиг ``pareto_obs_approx`` на все n
-------------------------------------------------------------
``configs/pareto_obs_approx.py`` задаёт условные моменты ОДНОГО исходного
наблюдения: ``C_mu = Y1 + Y2``, ``C_v = Y2^2 / 12``. Ядро фильтра
(``core/densities.make_obs_density``, канал NORMAL) накапливает параметры
линейно по временам пребывания: ``mu = sum_r u_r * C_mu``,
``v = sum_r u_r * C_v``, ``sum_r u_r = ht``. Поэтому конструктор ``Filter``
вызывается с ``ht = float(n)`` вместо ``cfg.ht`` (== 10 у этого конфига) --
это даёт ровно ЦПТ-правдоподобие суммы блока из n наблюдений:

    S_n = sum_{k=1}^{n} xi_k  ~  N( n (Y1+Y2), n Y2^2 / 12 ),

а при скачке состояния внутри блока -- те же формулы с фактическими
временами пребывания. Заводить отдельные конфиги под каждое n не нужно;
``cfg_clt.ht`` при построении ``Filter`` для ЦПТ-фильтров игнорируется
(см. ``run_filter``, где ``ht`` -- явный параметр, отдельный от ``cfg.ht``).

Наблюдения для всех n получаются суммированием непересекающихся блоков
одного и того же потока исходных наблюдений ``pareto_obs`` (ht=1); отдельно
не генерируются.

Оценка theta считается в двух вариантах:
  * theta_est       -- как в Filter.estimate(): sum_y psi[n, y], нормированная;
  * theta_est_delta -- delta[n] * sum_y psi[n, y] (формула (3.7) статьи).

Запуск:
    .venv/bin/python scripts/run_clt_averaging.py --hours 0.2 --ns 1,10,20,50
    .venv/bin/python scripts/run_clt_averaging.py                # весь T, n=1,10,20,50
    .venv/bin/python scripts/run_clt_averaging.py --no-exact --ns 5,20,100
"""
import _bootstrap  # noqa: F401

import argparse
import datetime
import os
import pathlib
import pickle
from math import lcm
from time import time

import numpy as np
import numba as nb

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from discretized_filter.utils.grids import to_discrete


EXACT_CONFIG = 'pareto_obs'
CLT_CONFIG = 'pareto_obs_approx'

# параметры, которые обязаны совпадать у обоих конфигов (кроме ht -- он
# заведомо разный: 1 у точного, 10 у approx, и в это сравнение не входит)
_SHARED_SCALARS = ('T', 'seed', 'N', 'M', 'K', 'pi_family', 'n_points',
                   'two_jumps', 'shared_grid')
_SHARED_ARRAYS = ('Lambda', 'p0', 'delta', 'M_net', 'pi', 'pi_init')


def parse_args():
    parser = argparse.ArgumentParser(
        description='ЦПТ-фильтры Парето при разных объёмах осреднения '
                     'против точного фильтра на одной траектории.'
    )
    parser.add_argument(
        '--ns', default='1,10,20,50',
        help='объёмы осреднения блока -- список положительных целых через '
             'запятую, без повторов (по умолчанию "1,10,20,50")',
    )
    parser.add_argument(
        '--hours', type=float, default=None,
        help='сколько часов траектории пропустить через фильтры '
             '(по умолчанию -- весь T конфига)',
    )
    parser.add_argument(
        '--no-exact', action='store_true',
        help='не считать точный фильтр (только набор ЦПТ-фильтров)',
    )
    parser.add_argument(
        '--load-exact', default=None, metavar='DIR',
        help='взять готовый прогон точного фильтра из каталога с '
             'pickle-артефактами theta/y/t/theta_est/y_est/observations '
             '(см. utils.io.save_path) вместо генерации траектории заново '
             'и пересчёта точного фильтра; несовместимо с --no-exact',
    )
    parser.add_argument(
        '--progress', type=int, default=5000,
        help='печатать прогресс каждые N шагов (0 -- не печатать)',
    )
    parser.add_argument(
        '--checkpoint', type=int, default=20000,
        help='сохранять промежуточный npz каждые N шагов (0 -- не сохранять)',
    )
    parser.add_argument(
        '--exp-id', default='clt_averaging',
        help='имя каталога результата: saved_path_<exp-id>/',
    )
    args = parser.parse_args()

    if args.no_exact and args.load_exact:
        parser.error('--no-exact и --load-exact несовместимы')

    try:
        ns = [int(s.strip()) for s in args.ns.split(',') if s.strip()]
    except ValueError:
        parser.error(f'--ns: не удалось разобрать список целых: {args.ns!r}')
    if not ns:
        parser.error('--ns: пустой список')
    if any(n <= 0 for n in ns):
        parser.error(f'--ns: все значения должны быть положительными: {ns}')
    if len(set(ns)) != len(ns):
        parser.error(f'--ns: значения не должны повторяться: {ns}')
    args.ns = ns
    return args


def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m {s:02d}s'


def check_configs(named_cfgs):
    """Сверяет параметры, общие для сравниваемых конфигов (перенесено из
    run_pareto_comparison.py)."""
    (ref_name, ref) = named_cfgs[0]
    for name, cfg in named_cfgs[1:]:
        for attr in _SHARED_SCALARS:
            a, b = getattr(ref, attr), getattr(cfg, attr)
            if a != b:
                raise ValueError(
                    f'конфиги {ref_name} и {name} расходятся по {attr}: '
                    f'{a!r} против {b!r}; сравнение на одной траектории '
                    'невозможно'
                )
        if [int(v) for v in ref.num_nodes] != [int(v) for v in cfg.num_nodes]:
            raise ValueError(
                f'конфиги {ref_name} и {name} расходятся по num1: '
                f'{ref.num_nodes} против {cfg.num_nodes}'
            )
        for attr in _SHARED_ARRAYS:
            if not np.array_equal(getattr(ref, attr), getattr(cfg, attr)):
                raise ValueError(
                    f'конфиги {ref_name} и {name} расходятся по массиву {attr}'
                )
        for i, (u, v) in enumerate(zip(ref.y_intervals, cfg.y_intervals)):
            if not np.array_equal(u, v):
                raise ValueError(
                    f'конфиги {ref_name} и {name} расходятся по '
                    f'y_intervals[{i}]'
                )


def estimate3(filt):
    """(theta как в Filter.estimate, theta по формуле (3.7) с delta, Y)."""
    theta_lib, y_est = filt.estimate()
    theta_delta = (filt.delta[:, np.newaxis] * filt.psi).sum(axis=1)
    theta_delta = theta_delta / theta_delta.sum()
    return theta_lib, theta_delta, y_est


def run_filter(key, label, cfg, ht, observations, progress, checkpoint, out_dir):
    """Прогон одного фильтра; возвращает (theta_est, theta_est_delta, y_est).

    ``ht`` -- ЯВНЫЙ параметр, отдельный от ``cfg.ht``: для точного фильтра
    он равен ``cfg.ht`` (== 1), а для ЦПТ-фильтра -- объёму осреднения n,
    а не ``cfg.ht`` (== 10) конфига ``pareto_obs_approx``. ``cfg.C`` этого
    конфига задаёт условные моменты ОДНОГО исходного наблюдения; ядро
    (core/densities.make_obs_density, канал NORMAL) накапливает параметры
    линейно по временам пребывания с суммой, равной переданному ``ht``.
    Подстановка ``ht=n`` даёт ровно ЦПТ-правдоподобие суммы блока из n
    исходных наблюдений -- см. докстринг модуля.
    """
    filt = Filter(
        cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
        cfg.N, cfg.Lambda, ht, cfg.delta, cfg.obs_density,
        n_points=cfg.n_points, two_jumps=cfg.two_jumps,
        filter_step=cfg.filter_step,
    )

    n_obs = observations.shape[0]
    # начальная точка -- оценка по pi_init, до первого наблюдения (см.
    # run_pareto_comparison.py про артефакт нормировки pi_init на сетке)
    est0 = estimate3(filt)
    th, thd, yy = [est0[0]], [est0[1]], [est0[2]]

    kernel = getattr(cfg.filter_step, 'py_func', cfg.filter_step)
    print(f'  [{label}] {n_obs} шагов, ht(Filter)={ht}, ядро '
          f'{getattr(kernel, "__name__", kernel)}', flush=True)
    start = time()
    done = n_obs

    for i, obs_ in enumerate(observations, start=1):
        try:
            filt.update(obs_)
        except (ZeroDivisionError, FloatingPointError) as exc:
            print(f'  [{label}] ОСТАНОВ на шаге {i}/{n_obs}: {exc}', flush=True)
            done = i - 1
            break

        est = estimate3(filt)
        if any(np.any(~np.isfinite(v)) for v in est):
            print(f'  [{label}] ОСТАНОВ на шаге {i}/{n_obs}: nan/inf в оценке',
                  flush=True)
            done = i - 1
            break
        th.append(est[0])
        thd.append(est[1])
        yy.append(est[2])

        if progress and (i % progress == 0):
            elapsed = time() - start
            eta = elapsed * (n_obs - i) / i
            print(f'  [{label}] {i}/{n_obs}  прошло {fmt_dur(elapsed)}  '
                  f'осталось ~{fmt_dur(eta)}', flush=True)
        if checkpoint and (i % checkpoint == 0):
            np.savez(
                os.path.join(out_dir, f'checkpoint_{key}.npz'),
                step=i, theta_est=np.array(th),
                theta_est_delta=np.array(thd), y_est=np.array(yy),
            )

    elapsed = time() - start
    print(f'  [{label}] готово: {done}/{n_obs} шагов за {fmt_dur(elapsed)}',
          flush=True)
    return np.array(th), np.array(thd), np.array(yy)


def load_exact_archive(directory):
    """Прямое чтение шести pickle-артефактов из произвольного каталога --
    та же раскладка файлов, что и у ``utils.io.load_saved_path``/
    ``save_path``, но без привязки к ``saved_path_dir(exp_id)``."""
    d = pathlib.Path(directory)
    names = ('theta', 'y', 't', 'theta_est', 'y_est', 'observations')
    out = {}
    for name in names:
        with open(d / f'{name}.pkl', 'rb') as f:
            out[name] = pickle.load(f)
    return (out['theta'], out['y'], out['t'],
            out['theta_est'], out['y_est'], out['observations'])


def onehot_theta(theta, N):
    return np.vstack([np.int64(theta == i) for i in range(N)]).T


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def main():
    args = parse_args()
    ns = args.ns
    L = lcm(*ns)

    # порядок set_config -- как в run_pareto_comparison.py: точный конфиг
    # активен последним, поэтому траектория порождается его ГПСЧ
    cfg_clt = set_config(CLT_CONFIG)
    cfg = set_config(EXACT_CONFIG)
    check_configs([(EXACT_CONFIG, cfg), (CLT_CONFIG, cfg_clt)])

    exact_mode = 'skip' if args.no_exact else ('load' if args.load_exact else 'run')

    out_dir = saved_path_dir(args.exp_id)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for key, c in (('exact', cfg), ('clt', cfg_clt)):
        # копия файла конфига напрямую: повторный set_config переопределил бы
        # sys.modules и джит-диспетчеры уже построенных cfg
        src = pathlib.Path(c._config_path).read_text(encoding='utf-8')
        header = (f'# Копия конфига для эксперимента {args.exp_id} ({key})\n'
                  f'# Имя конфига: {c._config_name}\n'
                  f'# Сохранено: {stamp}\n# {"=" * 60}\n')
        (out_dir / f'config_{key}.py').write_text(header + src, encoding='utf-8')

    n_grid = cfg.M_net.shape[1]
    print(f'потоков numba: {nb.get_num_threads()}')
    print(f'N={cfg.N}, узлов сетки={n_grid}, n_points={cfg.n_points}, '
          f'two_jumps={cfg.two_jumps}, ns={ns}, lcm(ns)={L}')
    print(f'delta по состояниям: {np.array2string(cfg.delta, precision=6)}')

    if exact_mode == 'load':
        theta, y, t, theta_est_loaded, y_est_loaded, obs_full = load_exact_archive(
            args.load_exact
        )
        theta = np.asarray(theta)
        y = np.asarray(y)
        t = np.asarray(t)
        theta_est_loaded = np.asarray(theta_est_loaded)
        y_est_loaded = np.asarray(y_est_loaded)
        obs_full = np.asarray(obs_full)
        if obs_full.ndim == 1:
            obs_full = obs_full[:, np.newaxis]
        if obs_full.shape[1] != cfg.K:
            raise ValueError(
                f'--load-exact {args.load_exact}: observations.shape[1]='
                f'{obs_full.shape[1]} != cfg.K={cfg.K}'
            )
        if theta_est_loaded.shape != (len(obs_full) + 1, cfg.N):
            raise ValueError(
                f'--load-exact {args.load_exact}: theta_est.shape='
                f'{theta_est_loaded.shape}, ожидалось '
                f'{(len(obs_full) + 1, cfg.N)}'
            )
        print(f'точный фильтр загружен из {args.load_exact}: '
              f'{obs_full.shape[0]} наблюдений (не пересчитывается)')
    else:
        theta, y, t = sparse_mc(
            cfg.p0, cfg.Lambda, cfg.lam, cfg.T, cfg.get_y, cfg.y_intervals
        )
        obs_full = cfg.get_obs(cfg.t_net_filtering, theta, y, t)
        print(f'скачков theta: {len(t)}, наблюдений (ht={cfg.ht}): '
              f'{obs_full.shape[0]}')

    # длина прогона: выравниваем по границе НОК(ns), чтобы все разбиения на
    # блоки покрывали ровно один и тот же отрезок времени
    n_obs = obs_full.shape[0]
    n_exact_max = n_obs
    if args.hours is not None:
        n_exact_max = min(n_exact_max, int(round(args.hours * 3600 / cfg.ht)))
    n_blocks_L = n_exact_max // L
    if n_blocks_L < 1:
        raise ValueError(
            f'--hours слишком мал: {n_exact_max} шагов не хватает даже на '
            f'один блок из lcm(ns)={L}'
        )
    n_exact = n_blocks_L * L
    obs = obs_full[:n_exact]

    horizon = n_exact * cfg.ht
    print(f'горизонт прогона: {horizon:.0f} ед. времени '
          f'({horizon / 3600:.2f} ч из {cfg.T / 3600:.2f} ч конфига); '
          f'{n_exact} исходных наблюдений (ht={cfg.ht})')

    obs_by_n = {
        n: obs.reshape(n_exact // n, n, cfg.K).sum(axis=1) for n in ns
    }

    if exact_mode == 'run':
        calls = n_exact * cfg.N * n_grid * (cfg.N - 1) * n_grid * cfg.n_points
        print(f'вызовов integrand у точного фильтра: ~{calls:.2e}')

    print('выборка            ht     n        mean        std      max|x|')
    if exact_mode == 'run':
        print(f'{"exact":<18} {cfg.ht:5.0f} {obs.shape[0]:7d} '
              f'{obs.mean():11.3f} {obs.std():10.3f} {np.abs(obs).max():11.3f}')
    for n in ns:
        o = obs_by_n[n]
        print(f'{"n=" + str(n):<18} {float(n):5.0f} {o.shape[0]:7d} '
              f'{o.mean():11.3f} {o.std():10.3f} {np.abs(o).max():11.3f}')

    # results: key -> (label, ht_grid, theta_est, theta_est_delta, y_est)
    results = {}

    if exact_mode == 'run':
        label = f'точный, ht={cfg.ht:.0f}'
        print(f'\n=== {label} ({EXACT_CONFIG})', flush=True)
        th, thd, yy = run_filter(
            'exact', label, cfg, cfg.ht, obs,
            args.progress, args.checkpoint, out_dir,
        )
        results['exact'] = (label, cfg.ht, th, thd, yy)
    elif exact_mode == 'load':
        # обрезаем до общего горизонта n_exact -- архив мог считаться на
        # большем горизонте, а сравнение с ЦПТ-фильтрами должно быть честным
        th = theta_est_loaded[:n_exact + 1]
        yy = y_est_loaded[:n_exact + 1]
        # theta_est_delta недоступна: save_path не сохраняет psi/delta,
        # только theta_est и y_est -- заполняем NaN и не участвуем в RMSE th_d
        thd = np.full_like(th, np.nan)
        results['exact'] = (f'точный (загружен), ht={cfg.ht:.0f}', cfg.ht, th, thd, yy)

    for n in ns:
        label = 'ЦПТ (Gaussian, h=1)' if n == 1 else f'ЦПТ (сумма блока n={n})'
        print(f'\n=== {label} ({CLT_CONFIG}, ht(Filter)=n)', flush=True)
        th, thd, yy = run_filter(
            f'n{n}', label, cfg_clt, float(n), obs_by_n[n],
            args.progress, args.checkpoint, out_dir,
        )
        results[f'n{n}'] = (label, float(n), th, thd, yy)

    # --- RMSE на собственной сетке каждого фильтра
    theta_onehot = onehot_theta(theta, cfg.N)
    grid_cache = {}

    def grids_for(ht_grid):
        if ht_grid not in grid_cache:
            dtheta = to_discrete(theta_onehot, t, cfg.T, ht_grid)
            dY = to_discrete(y, t, cfg.T, ht_grid)
            grid_cache[ht_grid] = (dtheta, dY)
        return grid_cache[ht_grid]

    order = (['exact'] if 'exact' in results else []) + [f'n{n}' for n in ns]
    rows = []
    for key in order:
        label, ht_grid, th, thd, yy = results[key]
        dtheta, dY = grids_for(ht_grid)
        m = min(dtheta.shape[0], th.shape[0])
        rows.append((
            label, m,
            rmse(dtheta[:m], th[:m]),
            rmse(dtheta[:m], thd[:m]),
            rmse(dY[:m, 0], yy[:m, 0]),
            rmse(dY[:m, 1], yy[:m, 1]),
        ))

    head = (f'{"фильтр":<32}{"точек":>7}{"RMSE th":>10}{"RMSE th_d":>11}'
            f'{"RMSE Y1":>10}{"RMSE Y2":>10}')
    lines = ['', 'RMSE на собственной сетке каждого фильтра '
                 '(th_d -- оценка theta с delta, формула (3.7))',
             head, '-' * len(head)]
    for label, n_pts, rt, rtd, r1, r2 in rows:
        lines.append(f'{label:<32}{n_pts:>7}{rt:>10.4f}{rtd:>11.4f}'
                     f'{r1:>10.4f}{r2:>10.4f}')

    if 'exact' in results:
        # главная интересующая величина: деградация ЦПТ-приближения с
        # ростом объёма осреднения n относительно точного фильтра,
        # прореженного на сетку блоков (theta_est_exact[::n])
        _, _, th_e, thd_e, yy_e = results['exact']
        lines += ['', 'расхождение ЦПТ-оценки с точной, прореженной на '
                      'сетку блоков (theta_est_exact[::n]):']
        sub_head = f'{"n":>6}{"RMSE th":>10}{"RMSE Y1":>10}{"RMSE Y2":>10}'
        lines.append(sub_head)
        lines.append('-' * len(sub_head))
        for n in ns:
            _, _, th_n, thd_n, yy_n = results[f'n{n}']
            th_thin = th_e[::n]
            yy_thin = yy_e[::n]
            m = min(th_thin.shape[0], th_n.shape[0])
            lines.append(
                f'{n:>6}{rmse(th_thin[:m], th_n[:m]):>10.4f}'
                f'{rmse(yy_thin[:m, 0], yy_n[:m, 0]):>10.4f}'
                f'{rmse(yy_thin[:m, 1], yy_n[:m, 1]):>10.4f}'
            )

    summary = '\n'.join(lines)
    print(summary, flush=True)

    payload = {
        'theta': theta, 'y': y, 't': t, 'obs': obs,
        'ns': np.array(ns), 'n_exact': n_exact, 'ht_exact': cfg.ht,
        'delta': cfg.delta,
        't_net_exact': cfg.t_net_filtering[:n_exact + 1],
    }
    if 'exact' in results:
        _, _, th_e, thd_e, yy_e = results['exact']
        payload['theta_est_exact'] = th_e
        payload['theta_est_delta_exact'] = thd_e
        payload['y_est_exact'] = yy_e
    for n in ns:
        _, _, th_n, thd_n, yy_n = results[f'n{n}']
        payload[f'obs_n{n}'] = obs_by_n[n]
        payload[f'theta_est_clt_n{n}'] = th_n
        payload[f'theta_est_delta_clt_n{n}'] = thd_n
        payload[f'y_est_clt_n{n}'] = yy_n
        # времена границ блоков размера n на сетке исходных наблюдений (ht=1)
        payload[f't_net_n{n}'] = np.arange(obs_by_n[n].shape[0] + 1) * n * cfg.ht

    np.savez(os.path.join(out_dir, 'comparison.npz'), **payload)

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nсохранено в {out_dir}')


if __name__ == '__main__':
    main()
