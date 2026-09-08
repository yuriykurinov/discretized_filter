"""
Точный фильтр против ЦПТ-аппроксимации на трёх каналах наблюдений (Парето,
экспоненциальный, равномерный) на ОДНОЙ траектории скрытого сигнала
``(theta, Y, t)``. См. план ``plans/20260908-channel-clt-comparison.md``.

  канал         точный конфиг     ЦПТ-конфиг            ratio  ht ЦПТ
  ------------- ----------------- --------------------- -----  ------
  pareto        pareto_obs        pareto_obs_clt          10     10
  exponential   exponential_obs   exponential_obs_clt      10     10
  uniform       uniform_obs       uniform_obs_clt           1      1

Траектория генерируется один раз (``sparse_mc``) по конфигу первого из
выбранных каналов; наблюдения для ЦПТ-фильтра не генерируются заново --
берутся блоки по ``ratio`` подряд идущих наблюдений точного потока и
усредняются. Перед генерацией наблюдений каждого канала посев ГПСЧ
принудительно сбрасывается на общий ``seed`` (``set_seed``), поэтому все три
канала видят один и тот же поток равномерных величин (common random
numbers) -- выбор подмножества через ``--channels`` не меняет результат
остальных каналов.

Оценка ``theta``/``Y`` берётся только через ``Filter.estimate()`` (умножение
``psi`` на ``delta`` уже внутри) -- отдельная оценка с явным ``delta``, как
в ``run_pareto_comparison.py``, ей тождественна и здесь не нужна.

Запуск:
    .venv/bin/python scripts/run_channel_clt_comparison.py --hours 3
    .venv/bin/python scripts/run_channel_clt_comparison.py
    .venv/bin/python scripts/run_channel_clt_comparison.py --channels pareto,uniform
"""
import _bootstrap  # noqa: F401

import argparse
import datetime
import os
from time import time

import numpy as np
import numba as nb

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from discretized_filter.utils.grids import set_seed, to_discrete


CHANNELS = ('pareto', 'exponential', 'uniform')

_CONFIG_EXACT = {
    'pareto': 'pareto_obs',
    'exponential': 'exponential_obs',
    'uniform': 'uniform_obs',
}
_CONFIG_CLT = {
    'pareto': 'pareto_obs_clt',
    'exponential': 'exponential_obs_clt',
    'uniform': 'uniform_obs_clt',
}

# параметры, которые обязаны совпадать у всех задействованных конфигов
_SHARED_SCALARS = ('T', 'seed', 'N', 'M', 'K', 'pi_family', 'n_points',
                   'two_jumps', 'shared_grid')
_SHARED_ARRAYS = ('Lambda', 'p0', 'delta', 'M_net', 'pi', 'pi_init')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Точный фильтр против ЦПТ-аппроксимации на трёх '
                     'каналах наблюдений (Парето, экспоненциальный, '
                     'равномерный) на одной траектории.'
    )
    parser.add_argument(
        '--channels', default=','.join(CHANNELS),
        help=f'какие каналы прогонять, через запятую из '
             f'{{{",".join(CHANNELS)}}}; порядок задаёт порядок прогона '
             '(по умолчанию все три)',
    )
    parser.add_argument(
        '--hours', type=float, default=None,
        help='сколько часов траектории пропустить через фильтры '
             '(по умолчанию -- весь T конфига)',
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
        '--exp-id', default='channel_clt_comparison',
        help='имя каталога результата: saved_path_<exp-id>/',
    )
    args = parser.parse_args()

    order = [s.strip() for s in args.channels.split(',') if s.strip()]
    unknown = [s for s in order if s not in CHANNELS]
    if unknown:
        parser.error(f'--channels: неизвестные имена {unknown}; '
                     f'доступны {sorted(CHANNELS)}')
    if not order:
        parser.error('--channels: не выбран ни один канал')
    if len(set(order)) != len(order):
        parser.error(f'--channels: имена не должны повторяться: {order}')
    args.order = order
    return args


def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m {s:02d}s'


def check_configs(named_cfgs):
    """Сверяет параметры, общие для сравниваемых конфигов (см.
    run_pareto_comparison.py: та же логика, набор имён вынесен в модуль)."""
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


def check_clt_moments(channel, cfg_exact, cfg_clt):
    """Численно сверяет моменты ЦПТ-конфига с точным (ловит рассогласование
    ``ratio``/``ht`` внутри ``*_obs_clt.py``: ``ratio`` -- модульная
    константа, вмороженная в джитованные ``drift``/``var``, наружу через
    ``cfg`` не выходит). Возвращает вычисленный ``ratio``."""
    ratio_f = cfg_clt.ht / cfg_exact.ht
    ratio = int(round(ratio_f))
    if abs(ratio_f - ratio) > 1e-9 or ratio < 1:
        raise ValueError(
            f'{channel}: ht ЦПТ / ht точного = {ratio_f} не целое >= 1 '
            f'({cfg_clt.ht} / {cfg_exact.ht}); блоки наблюдений не '
            'собираются'
        )

    loc = cfg_exact.C[:, :, 0, 0]
    scale = cfg_exact.C[:, :, 0, 1]
    mu_expected = loc + scale
    v_expected = scale ** 2 / (12.0 * ratio)
    mu_actual = cfg_clt.ht * cfg_clt.C[:, :, 0, 0]
    v_actual = cfg_clt.ht * cfg_clt.C[:, :, 0, 1]

    if not np.allclose(mu_actual, mu_expected, rtol=1e-10):
        resid = float(np.max(np.abs(mu_actual - mu_expected)))
        raise ValueError(
            f'{channel}: снос конфига {_CONFIG_CLT[channel]!r} (ht='
            f'{cfg_clt.ht}) не согласован с точным {_CONFIG_EXACT[channel]!r}'
            f' (ht={cfg_exact.ht}) при ratio={ratio}: max|resid|={resid:.3e};'
            ' вероятно, в конфиге не совпадают ratio и ht'
        )
    if not np.allclose(v_actual, v_expected, rtol=1e-10):
        resid = float(np.max(np.abs(v_actual - v_expected)))
        raise ValueError(
            f'{channel}: дисперсия конфига {_CONFIG_CLT[channel]!r} (ht='
            f'{cfg_clt.ht}) не согласована с точным {_CONFIG_EXACT[channel]!r}'
            f' (ht={cfg_exact.ht}) при ratio={ratio}: max|resid|={resid:.3e};'
            ' вероятно, в конфиге не совпадают ratio и ht'
        )
    return ratio


def onehot_theta(theta, N):
    return np.vstack([np.int64(theta == i) for i in range(N)]).T


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def run_filter(label, ckpt_key, cfg, observations, progress, checkpoint, out_dir):
    """Прогон одного фильтра; возвращает (theta_est, y_est) -- ровно то, что
    отдаёт ``Filter.estimate()`` (``psi`` уже умножена на ``delta``)."""
    filt = Filter(
        cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
        cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density,
        n_points=cfg.n_points, two_jumps=cfg.two_jumps,
        filter_step=cfg.filter_step,
    )

    n_obs = observations.shape[0]
    # начальная точка -- оценка по pi_init, до первого наблюдения (см.
    # run_pareto_comparison.py про артефакт нормировки pi_init на сетке)
    th0, y0 = filt.estimate()
    th, yy = [th0], [y0]

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

        est_th, est_y = filt.estimate()
        if np.any(~np.isfinite(est_th)) or np.any(~np.isfinite(est_y)):
            print(f'  [{label}] ОСТАНОВ на шаге {i}/{n_obs}: nan/inf в оценке',
                  flush=True)
            done = i - 1
            break
        th.append(est_th)
        yy.append(est_y)

        if progress and (i % progress == 0):
            elapsed = time() - start
            eta = elapsed * (n_obs - i) / i
            print(f'  [{label}] {i}/{n_obs}  прошло {fmt_dur(elapsed)}  '
                  f'осталось ~{fmt_dur(eta)}', flush=True)
        if checkpoint and (i % checkpoint == 0):
            np.savez(
                os.path.join(out_dir, f'checkpoint_{ckpt_key}.npz'),
                step=i, theta_est=np.array(th), y_est=np.array(yy),
            )

    elapsed = time() - start
    print(f'  [{label}] готово: {done}/{n_obs} шагов за {fmt_dur(elapsed)}',
          flush=True)
    return np.array(th), np.array(yy)


def main():
    args = parse_args()

    # 1. конфиги выбранных каналов (точный + ЦПТ каждый)
    cfgs = {}
    for c in args.order:
        cfgs[c] = {
            'exact': set_config(_CONFIG_EXACT[c]),
            'clt': set_config(_CONFIG_CLT[c]),
        }

    # 2. сверка конфигов -- одна и та же траектория должна быть осмысленна
    # для всех выбранных каналов сразу
    named_cfgs = []
    for c in args.order:
        named_cfgs.append((_CONFIG_EXACT[c], cfgs[c]['exact']))
        named_cfgs.append((_CONFIG_CLT[c], cfgs[c]['clt']))
    check_configs(named_cfgs)

    # 3. сверка моментов ЦПТ-конфига (ловит протухший ratio/ht)
    ratios = {c: check_clt_moments(c, cfgs[c]['exact'], cfgs[c]['clt'])
              for c in args.order}

    cfg0 = cfgs[args.order[0]]['exact']

    out_dir = saved_path_dir(args.exp_id)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for c in args.order:
        for kind in ('exact', 'clt'):
            cc = cfgs[c][kind]
            # копия файла конфига напрямую: повторный set_config переопределил
            # бы sys.modules и джит-диспетчеры уже построенных cfg
            src = cc._config_path.read_text(encoding='utf-8')
            header = (f'# Копия конфига для эксперимента {args.exp_id} '
                      f'({c}, {kind})\n'
                      f'# Имя конфига: {cc._config_name}\n'
                      f'# Сохранено: {stamp}\n# {"=" * 60}\n')
            (out_dir / f'config_{c}_{kind}.py').write_text(
                header + src, encoding='utf-8'
            )

    n_grid = cfg0.M_net.shape[1]
    print(f'потоков numba: {nb.get_num_threads()}')
    print(f'N={cfg0.N}, узлов сетки={n_grid}, n_points={cfg0.n_points}, '
          f'two_jumps={cfg0.two_jumps}')
    print(f'каналы: {args.order}, ratio: '
          f'{ {c: ratios[c] for c in args.order} }')
    print(f'delta по состояниям: {np.array2string(cfg0.delta, precision=6)}')

    # 4. одна траектория на все каналы
    set_seed(cfg0.seed)
    theta, y, t = sparse_mc(
        cfg0.p0, cfg0.Lambda, cfg0.lam, cfg0.T, cfg0.get_y, cfg0.y_intervals
    )
    print(f'скачков theta: {len(t)}, горизонт конфига: {cfg0.T / 3600:.2f} ч')

    # 5-6. наблюдения и длина прогона по каждому каналу
    channel_data = {}
    print('\nвыборка                  ht     n        mean        std      max|x|')
    for c in args.order:
        cfg_exact = cfgs[c]['exact']
        cfg_clt = cfgs[c]['clt']
        ratio = ratios[c]

        # общий посев перед get_obs -- common random numbers для всех каналов
        set_seed(cfg_exact.seed)
        obs_full = cfg_exact.get_obs(cfg_exact.t_net_filtering, theta, y, t)
        K = cfg_exact.K

        n_avail = obs_full.shape[0]
        if args.hours is not None:
            n_avail = min(n_avail, int(round(args.hours * 3600 / cfg_exact.ht)))
        n_blocks = n_avail // ratio
        if n_blocks < 1:
            raise ValueError(
                f'{c}: --hours слишком мал: {n_avail} шагов не хватает даже '
                f'на один блок из ratio={ratio}'
            )
        n_exact = n_blocks * ratio

        obs = obs_full[:n_exact]
        obs_mean = obs.reshape(n_blocks, ratio, K).mean(axis=1)

        channel_data[c] = dict(
            cfg_exact=cfg_exact, cfg_clt=cfg_clt, ratio=ratio,
            obs=obs, obs_mean=obs_mean, n_exact=n_exact, n_blocks=n_blocks,
        )

        print(f'{c + " obs":<25} {cfg_exact.ht:5.0f} {obs.shape[0]:7d} '
              f'{obs.mean():11.3f} {obs.std():10.3f} {np.abs(obs).max():11.3f}')
        print(f'{c + " obs_mean":<25} {cfg_clt.ht:5.0f} {obs_mean.shape[0]:7d} '
              f'{obs_mean.mean():11.3f} {obs_mean.std():10.3f} '
              f'{np.abs(obs_mean).max():11.3f}')

    # 7. фильтры
    results = {}
    for c in args.order:
        d = channel_data[c]
        horizon = d['n_exact'] * d['cfg_exact'].ht
        print(f'\n=== канал {c}: ratio={d["ratio"]}, ht точного='
              f'{d["cfg_exact"].ht:.0f}, ht ЦПТ={d["cfg_clt"].ht:.0f}, '
              f'{d["n_exact"]} шагов точного / {d["n_blocks"]} шагов ЦПТ, '
              f'горизонт {horizon:.0f} ед. времени ({horizon / 3600:.2f} ч)',
              flush=True)
        th_e, y_e = run_filter(
            f'{c}: точный, ht=1', f'{c}_exact', d['cfg_exact'], d['obs'],
            args.progress, args.checkpoint, out_dir,
        )
        th_c, y_c = run_filter(
            f'{c}: ЦПТ, ht={d["cfg_clt"].ht:.0f}', f'{c}_clt', d['cfg_clt'],
            d['obs_mean'], args.progress, args.checkpoint, out_dir,
        )
        results[c] = dict(theta_est_exact=th_e, y_est_exact=y_e,
                           theta_est_clt=th_c, y_est_clt=y_c)

    # --- RMSE: истина на ЦПТ-сетке (точный фильтр прореживается [::ratio]),
    # плюс отдельная строка "точный на своей сетке ht=1" для каждого канала
    head = (f'{"канал":<14}{"фильтр":<24}{"точек":>7}{"RMSE th":>10}'
            f'{"RMSE Y1":>10}{"RMSE Y2":>10}')
    lines = ['', 'RMSE на ЦПТ-сетке каждого канала (точный фильтр прорежен '
                 '[::ratio]); RMSE th -- по всем N компонентам one-hot',
             head, '-' * len(head)]

    for c in args.order:
        d = channel_data[c]
        r = results[c]
        cfg_exact = d['cfg_exact']
        cfg_clt = d['cfg_clt']
        theta_onehot = onehot_theta(theta, cfg_exact.N)

        dtheta_clt = to_discrete(theta_onehot, t, cfg_exact.T, cfg_clt.ht)
        dY_clt = to_discrete(y, t, cfg_exact.T, cfg_clt.ht)

        th_thin = r['theta_est_exact'][::d['ratio']]
        y_thin = r['y_est_exact'][::d['ratio']]
        n_e = min(dtheta_clt.shape[0], th_thin.shape[0])
        lines.append(
            f'{c:<14}{"точный (прорежен)":<24}{n_e:>7}'
            f'{rmse(dtheta_clt[:n_e], th_thin[:n_e]):>10.4f}'
            f'{rmse(dY_clt[:n_e, 0], y_thin[:n_e, 0]):>10.4f}'
            f'{rmse(dY_clt[:n_e, 1], y_thin[:n_e, 1]):>10.4f}'
        )

        th_c = r['theta_est_clt']
        y_c = r['y_est_clt']
        n_c = min(dtheta_clt.shape[0], th_c.shape[0])
        lines.append(
            f'{c:<14}{"ЦПТ":<24}{n_c:>7}'
            f'{rmse(dtheta_clt[:n_c], th_c[:n_c]):>10.4f}'
            f'{rmse(dY_clt[:n_c, 0], y_c[:n_c, 0]):>10.4f}'
            f'{rmse(dY_clt[:n_c, 1], y_c[:n_c, 1]):>10.4f}'
        )

        dtheta1 = to_discrete(theta_onehot, t, cfg_exact.T, cfg_exact.ht)
        dY1 = to_discrete(y, t, cfg_exact.T, cfg_exact.ht)
        th_e = r['theta_est_exact']
        y_e = r['y_est_exact']
        n1 = min(dtheta1.shape[0], th_e.shape[0])
        lines.append(
            f'{c:<14}{"точный, своя сетка ht=1":<24}{n1:>7}'
            f'{rmse(dtheta1[:n1], th_e[:n1]):>10.4f}'
            f'{rmse(dY1[:n1, 0], y_e[:n1, 0]):>10.4f}'
            f'{rmse(dY1[:n1, 1], y_e[:n1, 1]):>10.4f}'
        )

    summary = '\n'.join(lines)
    print(summary, flush=True)

    # --- comparison.npz по схеме плана (раздел 3): ноутбук читает именно её
    payload = {
        'theta': theta, 'y': y, 't': t,
        'delta': cfg0.delta, 'T': cfg0.T, 'seed': cfg0.seed, 'N': cfg0.N,
        'channels': np.array(args.order),
    }
    for c in args.order:
        d = channel_data[c]
        r = results[c]
        cfg_exact = d['cfg_exact']
        cfg_clt = d['cfg_clt']
        payload[f'obs_{c}'] = d['obs']
        payload[f'obs_mean_{c}'] = d['obs_mean']
        payload[f'ratio_{c}'] = d['ratio']
        payload[f'ht_exact_{c}'] = cfg_exact.ht
        payload[f'ht_clt_{c}'] = cfg_clt.ht
        payload[f'n_exact_{c}'] = d['n_exact']
        payload[f'n_blocks_{c}'] = d['n_blocks']
        payload[f't_net_exact_{c}'] = np.arange(d['n_exact'] + 1) * cfg_exact.ht
        payload[f't_net_clt_{c}'] = np.arange(d['n_blocks'] + 1) * cfg_clt.ht
        payload[f'theta_est_exact_{c}'] = r['theta_est_exact']
        payload[f'y_est_exact_{c}'] = r['y_est_exact']
        payload[f'theta_est_clt_{c}'] = r['theta_est_clt']
        payload[f'y_est_clt_{c}'] = r['y_est_clt']

    np.savez(os.path.join(out_dir, 'comparison.npz'), **payload)

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nсохранено в {out_dir}')


if __name__ == '__main__':
    main()
