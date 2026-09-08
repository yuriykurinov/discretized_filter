"""
Сравнение трёх фильтров на ОДНОЙ траектории (theta, Y) и ОДНОМ потоке
наблюдений -- скриптовый аналог notebooks/pareto_obs.ipynb для длинных
прогонов (в том числе на кластере).

  конфиг              ht   правдоподобие шага                наблюдения
  ------------------- ---- --------------------------------- ---------------------
  pareto_obs           1   точная плотность Y1 + Y2*eps      исходные, шаг 1
  pareto_obs_approx   10   гауссовская по двум моментам      блоки по 10, сумма
  pareto_obs_clt      10   то же, по среднему блока          блоки по 10, среднее

Наблюдения для фильтров с ht=10 НЕ генерируются заново: берутся блоки по
ratio = 10 из выборки точного конфига (наблюдение -- процесс с приращениями).
Порядок set_config -- тот же, что в ноутбуке (approx, clt, exact), поэтому
поток ГПСЧ и траектория совпадают с ноутбучными.

Оценка theta считается в двух вариантах:
  * theta_est       -- как в Filter.estimate(): sum_y psi[n, y], нормированная;
  * theta_est_delta -- delta[n] * sum_y psi[n, y] (формула (3.7) статьи).
Они различаются, когда delta[n] не одинакова по состояниям, -- а в конфигах
Парето это так (y1_intervals[0] шириной 4 против 5 у остальных), см. сводку
в конце прогона.

Запуск:
    .venv/bin/python scripts/run_pareto_comparison.py --hours 3      # пробный
    .venv/bin/python scripts/run_pareto_comparison.py                # весь T
    .venv/bin/python scripts/run_pareto_comparison.py --filters approx,clt
"""
import _bootstrap  # noqa: F401

import argparse
import datetime
import os
import pathlib
from time import time

import numpy as np
import numba as nb

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from discretized_filter.utils.grids import to_discrete
from discretized_filter.utils.io import save_path


EXACT = 'exact'
APPROX = 'approx'
CLT = 'clt'

_CONFIG_NAME = {
    EXACT: 'pareto_obs',
    APPROX: 'pareto_obs_approx',
    CLT: 'pareto_obs_clt',
}
_LABEL = {
    EXACT: 'точный, ht=1',
    APPROX: 'аппрокс. (сумма блока), ht=10',
    CLT: 'ЦПТ (среднее блока), ht=10',
}

# параметры, которые обязаны совпадать у всех трёх конфигов
_SHARED_SCALARS = ('T', 'seed', 'N', 'M', 'K', 'pi_family', 'n_points',
                   'two_jumps', 'shared_grid')
_SHARED_ARRAYS = ('Lambda', 'p0', 'delta', 'M_net', 'pi', 'pi_init')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Точный фильтр Парето против гауссовских аппроксимаций.'
    )
    parser.add_argument(
        '--hours', type=float, default=None,
        help='сколько часов траектории пропустить через фильтры '
             '(по умолчанию -- весь T конфига)',
    )
    parser.add_argument(
        '--filters', default=f'{APPROX},{CLT},{EXACT}',
        help=f'какие фильтры прогонять, через запятую из '
             f'{{{EXACT},{APPROX},{CLT}}}; порядок задаёт порядок прогона '
             '(по умолчанию сначала дешёвые)',
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
        '--exp-id', default='pareto_comparison',
        help='имя каталога результата: saved_path_<exp-id>/',
    )
    args = parser.parse_args()

    order = [s.strip() for s in args.filters.split(',') if s.strip()]
    unknown = [s for s in order if s not in _CONFIG_NAME]
    if unknown:
        parser.error(f'--filters: неизвестные имена {unknown}; '
                     f'доступны {sorted(_CONFIG_NAME)}')
    if not order:
        parser.error('--filters: не выбран ни один фильтр')
    args.order = order
    return args


def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m {s:02d}s'


def check_configs(named_cfgs):
    """Сверяет параметры, общие для сравниваемых конфигов."""
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


def run_filter(key, cfg, observations, progress, checkpoint, out_dir):
    """Прогон одного фильтра; возвращает (theta_est, theta_est_delta, y_est)."""
    label = _LABEL[key]
    filt = Filter(
        cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
        cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density,
        n_points=cfg.n_points, two_jumps=cfg.two_jumps,
        filter_step=cfg.filter_step,
    )

    n_obs = observations.shape[0]
    # начальная точка -- оценка по pi_init, до первого наблюдения; масса
    # pi_init на сетке равна (num/(num-1))^M, а не 1 (прямоугольники по
    # замкнутому интервалу), поэтому y_est в этой точке завышен на тот же
    # множитель; после первого update psi нормирована и артефакт уходит
    est0 = estimate3(filt)
    th, thd, yy = [est0[0]], [est0[1]], [est0[2]]

    kernel = getattr(cfg.filter_step, 'py_func', cfg.filter_step)
    print(f'  [{label}] {n_obs} шагов, ядро '
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


def rmse_rows(results, dtheta, dY, ratio):
    """Строки таблицы RMSE на общей сетке ht=10 (точный фильтр прореживается)."""
    rows = []
    for key, (th, thd, yy) in results.items():
        if key == EXACT:
            th, thd, yy = th[::ratio], thd[::ratio], yy[::ratio]
        n = min(dtheta.shape[0], th.shape[0])
        rows.append((
            _LABEL[key],
            n,
            float(np.sqrt(((dtheta[:n] - th[:n]) ** 2).mean())),
            float(np.sqrt(((dtheta[:n] - thd[:n]) ** 2).mean())),
            float(np.sqrt(((dY[:n, 0] - yy[:n, 0]) ** 2).mean())),
            float(np.sqrt(((dY[:n, 1] - yy[:n, 1]) ** 2).mean())),
        ))
    return rows


def main():
    args = parse_args()

    # порядок загрузки -- как в ноутбуке: точный конфиг активен последним,
    # поэтому траектория порождается его ГПСЧ
    cfg_approx = set_config(_CONFIG_NAME[APPROX])
    cfg_clt = set_config(_CONFIG_NAME[CLT])
    cfg = set_config(_CONFIG_NAME[EXACT])
    cfgs = {EXACT: cfg, APPROX: cfg_approx, CLT: cfg_clt}

    check_configs([(_CONFIG_NAME[k], c) for k, c in cfgs.items()])

    if cfg_clt.ht != cfg_approx.ht:
        raise ValueError(
            f'ht конфигов approx ({cfg_approx.ht}) и clt ({cfg_clt.ht}) '
            'должны совпадать: оба работают на блоках одной длины'
        )
    ratio_f = cfg_approx.ht / cfg.ht
    ratio = int(round(ratio_f))
    if abs(ratio_f - ratio) > 1e-12 or ratio < 1:
        raise ValueError(
            f'ht approx / ht exact = {ratio_f} не целое: блоки наблюдений '
            'не собираются'
        )

    out_dir = saved_path_dir(args.exp_id)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for key, c in cfgs.items():
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
          f'two_jumps={cfg.two_jumps}, ratio={ratio}')
    print(f'delta по состояниям: {np.array2string(cfg.delta, precision=6)}')

    theta, y, t = sparse_mc(
        cfg.p0, cfg.Lambda, cfg.lam, cfg.T, cfg.get_y, cfg.y_intervals
    )
    obs_full = cfg.get_obs(cfg.t_net_filtering, theta, y, t)
    print(f'скачков theta: {len(t)}, наблюдений (ht={cfg.ht}): '
          f'{obs_full.shape[0]}')

    # длина прогона: выравниваем по границе блока, чтобы все три фильтра
    # покрывали ровно один и тот же отрезок времени
    n_exact_max = obs_full.shape[0]
    if args.hours is not None:
        n_exact_max = min(n_exact_max, int(round(args.hours * 3600 / cfg.ht)))
    n_blocks = n_exact_max // ratio
    if n_blocks < 1:
        raise ValueError(
            f'--hours слишком мал: {n_exact_max} шагов не хватает даже на '
            f'один блок из {ratio}'
        )
    n_exact = n_blocks * ratio

    obs = obs_full[:n_exact]
    obs_agg = obs.reshape(n_blocks, ratio, cfg.K).sum(axis=1)
    obs_mean = obs_agg / ratio
    observations = {EXACT: obs, APPROX: obs_agg, CLT: obs_mean}

    horizon = n_exact * cfg.ht
    print(f'горизонт прогона: {horizon:.0f} ед. времени '
          f'({horizon / 3600:.2f} ч из {cfg.T / 3600:.2f} ч конфига); '
          f'{n_exact} шагов точного, {n_blocks} шагов приближённых')

    calls = n_exact * cfg.N * n_grid * (cfg.N - 1) * n_grid * cfg.n_points
    print(f'вызовов integrand у точного фильтра: ~{calls:.2e}')
    print('выборка            ht     n        mean        std      max|x|')
    for key in (EXACT, APPROX, CLT):
        o = observations[key]
        print(f'{key:<18} {cfgs[key].ht:5.0f} {o.shape[0]:7d} '
              f'{o.mean():11.3f} {o.std():10.3f} {np.abs(o).max():11.3f}')

    results = {}
    for key in args.order:
        print(f'\n=== {_LABEL[key]} ({_CONFIG_NAME[key]})', flush=True)
        results[key] = run_filter(
            key, cfgs[key], observations[key],
            args.progress, args.checkpoint, out_dir,
        )
        th, thd, yy = results[key]
        # совместимость с utils.io.load_saved_path и старыми ноутбуками
        save_path(cfgs[key].exp_id, theta, y, t, th, yy, observations[key])

    # --- RMSE на общей сетке ht = ratio * ht_exact
    dtheta = to_discrete(
        np.vstack([np.int64(theta == i) for i in range(cfg.N)]).T,
        t, cfg.T, cfg_approx.ht,
    )
    dY = to_discrete(y, t, cfg.T, cfg_approx.ht)
    rows = rmse_rows(results, dtheta, dY, ratio)

    head = (f'{"фильтр":<32}{"точек":>7}{"RMSE th":>10}{"RMSE th_d":>11}'
            f'{"RMSE Y1":>10}{"RMSE Y2":>10}')
    lines = ['', f'RMSE на общей сетке ht={cfg_approx.ht:.0f} '
                 '(точный прорежен); th_d -- оценка theta с delta, формула (3.7)',
             head, '-' * len(head)]
    for label, n, rt, rtd, r1, r2 in rows:
        lines.append(f'{label:<32}{n:>7}{rt:>10.4f}{rtd:>11.4f}'
                     f'{r1:>10.4f}{r2:>10.4f}')

    if EXACT in results:
        # справочно: точный фильтр на своей сетке ht=1, без прореживания
        dtheta1 = to_discrete(
            np.vstack([np.int64(theta == i) for i in range(cfg.N)]).T,
            t, cfg.T, cfg.ht,
        )
        dY1 = to_discrete(y, t, cfg.T, cfg.ht)
        th, thd, yy = results[EXACT]
        n1 = min(dtheta1.shape[0], th.shape[0])
        lines += [
            '',
            f'{"точный на своей сетке ht=1":<32}{n1:>7}'
            f'{np.sqrt(((dtheta1[:n1] - th[:n1]) ** 2).mean()):>10.4f}'
            f'{np.sqrt(((dtheta1[:n1] - thd[:n1]) ** 2).mean()):>11.4f}'
            f'{np.sqrt(((dY1[:n1, 0] - yy[:n1, 0]) ** 2).mean()):>10.4f}'
            f'{np.sqrt(((dY1[:n1, 1] - yy[:n1, 1]) ** 2).mean()):>10.4f}',
        ]

    if APPROX in results and CLT in results:
        # сумма и среднее блока обязаны совпасть: осреднение -- общий
        # множитель, сокращающийся при нормировке psi
        nc = min(results[APPROX][0].shape[0], results[CLT][0].shape[0])
        d_th = np.abs(results[APPROX][0][:nc] - results[CLT][0][:nc]).max()
        d_y = np.abs(results[APPROX][2][:nc] - results[CLT][2][:nc]).max()
        lines += [
            '',
            f'сумма против среднего блока: max|dtheta| = {d_th:.3e}, '
            f'max|dY| = {d_y:.3e}  (ожидается уровень ошибок округления)',
        ]

    summary = '\n'.join(lines)
    print(summary, flush=True)

    payload = {
        'theta': theta, 'y': y, 't': t,
        'obs': obs, 'obs_agg': obs_agg, 'obs_mean': obs_mean,
        'ratio': ratio, 'n_exact': n_exact, 'n_blocks': n_blocks,
        'ht_exact': cfg.ht, 'ht_agg': cfg_approx.ht,
        't_net_exact': cfg.t_net_filtering[:n_exact + 1],
        't_net_agg': cfg_approx.t_net_filtering[:n_blocks + 1],
        'delta': cfg.delta,
    }
    for key, (th, thd, yy) in results.items():
        payload[f'theta_est_{key}'] = th
        payload[f'theta_est_delta_{key}'] = thd
        payload[f'y_est_{key}'] = yy
    np.savez(os.path.join(out_dir, 'comparison.npz'), **payload)

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nсохранено в {out_dir}')


if __name__ == '__main__':
    main()
