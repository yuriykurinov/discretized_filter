"""Прогон винзоризированных ЦПТ-фильтров по сохранённой траектории.

Исходные ЦПТ-ветви и точные оценки берутся из ``comparison.npz`` без
пересчёта. Для каждого исходного наблюдения применяется верхняя
винзоризация ``min(X, U)``, после чего наблюдения складываются в блоки.
"""
import _bootstrap  # noqa: F401

import argparse
import os
import pathlib
import shutil
import tempfile
from time import time

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.paths import saved_path_dir


WINSORIZED_CONFIG = 'pareto_obs_approx_winsorized'
DEFAULT_SOURCE_DIR = 'saved_path_clt_averaging_diag'
PARETO_ALPHA = 2.5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Винзоризированные ЦПТ-фильтры на сохранённой траектории.'
    )
    parser.add_argument(
        '--source-dir', default=DEFAULT_SOURCE_DIR,
        help='каталог исходного comparison.npz '
             f'(по умолчанию {DEFAULT_SOURCE_DIR})',
    )
    parser.add_argument(
        '--quantile', type=float, default=0.999,
        help='квантиль верхней винзоризации (по умолчанию 0.999)',
    )
    parser.add_argument(
        '--ns', default='1,10,20,50',
        help='размеры блоков через запятую (по умолчанию 1,10,20,50)',
    )
    parser.add_argument(
        '--exp-id', default='clt_averaging_winsorized',
        help='имя каталога результата saved_path_<exp-id>/',
    )
    parser.add_argument(
        '--progress', type=int, default=5000,
        help='печатать прогресс каждые N шагов (0 -- не печатать)',
    )
    parser.add_argument(
        '--checkpoint', type=int, default=20000,
        help='сохранять checkpoint ветви каждые N шагов (0 -- не сохранять)',
    )
    args = parser.parse_args()

    if not np.isfinite(args.quantile) or not 0.0 < args.quantile < 1.0:
        parser.error('--quantile должен удовлетворять 0 < q < 1')
    if args.progress < 0:
        parser.error('--progress не может быть отрицательным')
    if args.checkpoint < 0:
        parser.error('--checkpoint не может быть отрицательным')
    try:
        ns = [int(item.strip()) for item in args.ns.split(',') if item.strip()]
    except ValueError:
        parser.error(f'--ns: не удалось разобрать список {args.ns!r}')
    if not ns or any(n <= 0 for n in ns):
        parser.error('--ns должен содержать положительные целые')
    if len(ns) != len(set(ns)):
        parser.error('--ns не должен содержать повторов')
    args.ns = ns
    return args


def fmt_dur(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s'


def atomic_savez(path, payload):
    """Атомарно заменяет npz в пределах его каталога."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f'.{path.name}.', suffix='.npz',
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
        np.savez(tmp_name, **payload)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_source(source_dir, ns):
    archive_path = pathlib.Path(source_dir) / 'comparison.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}

    common = (
        'theta', 'y', 't', 'ns', 'n_exact', 'ht_exact', 'delta',
        't_net_exact', 'theta_est_exact', 'theta_est_delta_exact',
        'y_est_exact',
    )
    missing = [key for key in common if key not in payload]
    obs_key = 'obs_raw' if 'obs_raw' in payload else 'obs'
    if obs_key not in payload:
        missing.append('obs (или obs_raw)')
    for n in ns:
        for key in (
            f't_net_n{n}', f'obs_n{n}', f'theta_est_clt_n{n}',
            f'theta_est_delta_clt_n{n}', f'y_est_clt_n{n}',
        ):
            if key not in payload:
                missing.append(key)
    if missing:
        raise KeyError(f'{archive_path}: отсутствуют ключи: {", ".join(missing)}')

    source_ns = set(np.asarray(payload['ns'], dtype=int).tolist())
    absent_ns = [n for n in ns if n not in source_ns]
    if absent_ns:
        raise ValueError(f'в исходном архиве нет ветвей n={absent_ns}')

    obs_raw = np.asarray(payload[obs_key])
    if obs_raw.ndim == 1:
        obs_raw = obs_raw[:, np.newaxis]
    if obs_raw.ndim != 2:
        raise ValueError(f'{obs_key}.shape={obs_raw.shape}, ожидалась матрица')
    n_exact = int(np.asarray(payload['n_exact']).item())
    if len(obs_raw) != n_exact:
        raise ValueError(
            f'len({obs_key})={len(obs_raw)} не совпадает с n_exact={n_exact}'
        )

    for n in ns:
        if n_exact % n:
            raise ValueError(f'n_exact={n_exact} не делится на n={n}')
        n_blocks = n_exact // n
        obs_n = np.asarray(payload[f'obs_n{n}'])
        if obs_n.ndim == 1:
            obs_n = obs_n[:, np.newaxis]
        if obs_n.shape != (n_blocks, obs_raw.shape[1]):
            raise ValueError(
                f'obs_n{n}.shape={obs_n.shape}, ожидалось '
                f'{(n_blocks, obs_raw.shape[1])}'
            )
        if np.asarray(payload[f't_net_n{n}']).shape[0] != n_blocks + 1:
            raise ValueError(f't_net_n{n} имеет неверную длину')
        for stem in ('theta_est_clt', 'theta_est_delta_clt', 'y_est_clt'):
            arr = np.asarray(payload[f'{stem}_n{n}'])
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[0] > n_blocks + 1:
                raise ValueError(f'{stem}_n{n}.shape={arr.shape} несовместима с n={n}')
    return payload, obs_raw, archive_path


def estimate3(filt):
    theta_est, y_est = filt.estimate()
    theta_est_delta = (filt.delta[:, np.newaxis] * filt.psi).sum(axis=1)
    theta_est_delta = theta_est_delta / theta_est_delta.sum()
    return theta_est, theta_est_delta, y_est


def winsorization_metadata(cfg, quantile):
    """Вычисляет тот же общий порог, что и winsorized-конфиг.

    ``set_config`` возвращает собранный объект конфигурации и не обязан
    экспортировать модульные константы ``ALPHA``, ``QUANTILE`` и
    ``THRESHOLD``. Поэтому producer воспроизводит формулу порога из того же
    параметра CLI и сеточных интервалов уже загруженного конфига.
    """
    alpha = PARETO_ALPHA
    pareto_mean = alpha / (alpha - 1.0)
    pareto_sd = np.sqrt(alpha / (alpha - 2.0)) / (alpha - 1.0)
    c = 1.0 - pareto_mean / (pareto_sd * np.sqrt(12.0))
    d = 1.0 / (pareto_sd * np.sqrt(12.0))
    loc_max = float(np.max(cfg.y_intervals[0]))
    scale_max = float(np.max(cfg.y_intervals[1]))
    threshold = loc_max + scale_max * (
        c + d * (1.0 - quantile)**(-1.0 / alpha)
    )
    return alpha, threshold


def run_arm(n, cfg, observations, progress, checkpoint, out_dir):
    """Запускает одну ветвь и всегда возвращает явный статус."""
    requested = observations.shape[0]
    failure_step = -1
    failure_message = ''
    started = time()

    try:
        filt = Filter(
            cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
            cfg.N, cfg.Lambda, float(n), cfg.delta, cfg.obs_density,
            n_points=cfg.n_points, two_jumps=cfg.two_jumps,
            filter_step=cfg.filter_step,
        )
        initial = estimate3(filt)
        theta_est = [initial[0]]
        theta_est_delta = [initial[1]]
        y_est = [initial[2]]
    except Exception as exc:  # одна ветвь не должна блокировать следующие
        return {
            'theta_est': np.empty((0, cfg.N)),
            'theta_est_delta': np.empty((0, cfg.N)),
            'y_est': np.empty((0, len(cfg.y_intervals))),
            'status': 'failed', 'requested': requested, 'done': 0,
            'failure_step': 0,
            'failure_message': f'{type(exc).__name__}: {exc}',
        }

    for step, observation in enumerate(observations, start=1):
        try:
            filt.update(observation)
            estimate = estimate3(filt)
            if any(np.any(~np.isfinite(value)) for value in estimate):
                raise FloatingPointError('nan/inf в оценке')
        except Exception as exc:  # сохраняем префикс и продолжаем другие n
            failure_step = step
            failure_message = f'{type(exc).__name__}: {exc}'
            break

        theta_est.append(estimate[0])
        theta_est_delta.append(estimate[1])
        y_est.append(estimate[2])

        if progress and step % progress == 0:
            elapsed = time() - started
            eta = elapsed * (requested - step) / step
            print(
                f'  [winsorized n={n}] {step}/{requested}; '
                f'прошло {fmt_dur(elapsed)}, осталось ~{fmt_dur(eta)}',
                flush=True,
            )
        if checkpoint and step % checkpoint == 0:
            atomic_savez(
                pathlib.Path(out_dir) / f'checkpoint_winsorized_n{n}.npz',
                {
                    'step': np.int64(step),
                    'theta_est': np.asarray(theta_est),
                    'theta_est_delta': np.asarray(theta_est_delta),
                    'y_est': np.asarray(y_est),
                },
            )

    done = len(theta_est) - 1
    status = 'completed' if done == requested else 'failed'
    if status == 'completed':
        failure_step = -1
        failure_message = ''
    return {
        'theta_est': np.asarray(theta_est),
        'theta_est_delta': np.asarray(theta_est_delta),
        'y_est': np.asarray(y_est),
        'status': status, 'requested': requested, 'done': done,
        'failure_step': failure_step, 'failure_message': failure_message,
    }


def put_status(payload, branch, n, status, requested, done,
               failure_step, failure_message):
    payload[f'status_{branch}_n{n}'] = np.asarray(status, dtype=str)
    payload[f'steps_requested_{branch}_n{n}'] = np.int64(requested)
    payload[f'steps_done_{branch}_n{n}'] = np.int64(done)
    payload[f'failure_step_{branch}_n{n}'] = np.int64(failure_step)
    payload[f'failure_message_{branch}_n{n}'] = np.asarray(
        failure_message, dtype=str
    )


def raw_failure_message(source_dir, n, done, requested):
    if done == requested:
        return ''
    report = pathlib.Path(source_dir) / f'failure_n{n}_step{done + 1}' / 'report.txt'
    if report.exists():
        first_line = report.read_text(encoding='utf-8').splitlines()
        if first_line:
            return first_line[0]
    return f'Сохранённая raw-ветвь остановилась после {done} из {requested} шагов'


def make_base_payload(source, obs_raw, obs_winsorized, ns, source_dir,
                      quantile, threshold, alpha):
    payload = {
        'theta': source['theta'], 'y': source['y'], 't': source['t'],
        'obs_raw': obs_raw, 'obs_winsorized': obs_winsorized,
        'ns': np.asarray(ns, dtype=np.int64),
        'n_exact': source['n_exact'], 'ht_exact': source['ht_exact'],
        'delta': source['delta'], 't_net_exact': source['t_net_exact'],
        'quantile': np.float64(quantile),
        'threshold': np.float64(threshold),
        'alpha': np.float64(alpha),
        'source_dir': np.asarray(str(pathlib.Path(source_dir).resolve()), dtype=str),
        'theta_est_exact': source['theta_est_exact'],
        'theta_est_delta_exact': source['theta_est_delta_exact'],
        'y_est_exact': source['y_est_exact'],
    }
    n_exact, channels = obs_raw.shape
    for n in ns:
        n_blocks = n_exact // n
        payload[f't_net_n{n}'] = source[f't_net_n{n}']
        raw_n = np.asarray(source[f'obs_n{n}'])
        if raw_n.ndim == 1:
            raw_n = raw_n[:, np.newaxis]
        payload[f'obs_raw_n{n}'] = raw_n
        payload[f'obs_winsorized_n{n}'] = obs_winsorized.reshape(
            n_blocks, n, channels
        ).sum(axis=1)
        payload[f'n_winsorized_n{n}'] = (obs_raw > threshold).reshape(
            n_blocks, n, channels
        ).sum(axis=1)

        raw_theta = source[f'theta_est_clt_n{n}']
        raw_theta_delta = source[f'theta_est_delta_clt_n{n}']
        raw_y = source[f'y_est_clt_n{n}']
        payload[f'theta_est_raw_n{n}'] = raw_theta
        payload[f'theta_est_delta_raw_n{n}'] = raw_theta_delta
        payload[f'y_est_raw_n{n}'] = raw_y
        requested = raw_n.shape[0]
        done = np.asarray(raw_theta).shape[0] - 1
        status = 'completed' if done == requested else 'failed'
        failure_step = -1 if status == 'completed' else done + 1
        failure_message = raw_failure_message(source_dir, n, done, requested)
        put_status(
            payload, 'raw', n, status, requested, done,
            failure_step, failure_message,
        )
    return payload


def main():
    args = parse_args()
    source_dir = pathlib.Path(args.source_dir).resolve()
    out_dir = pathlib.Path(saved_path_dir(args.exp_id)).resolve()
    if source_dir == out_dir:
        raise ValueError(
            'каталог результата совпадает с --source-dir; исходный '
            'comparison.npz перезаписывать нельзя'
        )
    source, obs_raw, archive_path = load_source(source_dir, args.ns)

    os.environ['DFILTER_PARETO_QUANTILE'] = repr(args.quantile)
    cfg = set_config(WINSORIZED_CONFIG)
    alpha, threshold = winsorization_metadata(cfg, args.quantile)
    if obs_raw.shape[1] != cfg.K:
        raise ValueError(
            f'obs_raw.shape[1]={obs_raw.shape[1]} не совпадает с cfg.K={cfg.K}'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    source_config = source_dir / 'config_clt.py'
    if not source_config.is_file():
        raise FileNotFoundError(f'не найден snapshot исходного конфига: {source_config}')
    shutil.copy2(source_config, out_dir / 'config_raw.py')
    shutil.copy2(pathlib.Path(cfg._config_path), out_dir / 'config_winsorized.py')

    obs_winsorized = np.minimum(obs_raw, threshold)
    payload = make_base_payload(
        source, obs_raw, obs_winsorized, args.ns, source_dir,
        args.quantile, threshold, alpha,
    )
    partial_path = out_dir / 'comparison.partial.npz'

    print(f'источник: {archive_path}', flush=True)
    print(
        f'q={args.quantile}, U={threshold:.6g}, ns={args.ns}; '
        f'затронуто наблюдений: {np.count_nonzero(obs_raw > threshold)}',
        flush=True,
    )
    for n in args.ns:
        observations = payload[f'obs_winsorized_n{n}']
        print(
            f'\n=== winsorized n={n}: {len(observations)} шагов, '
            f'ht(Filter)={float(n)}',
            flush=True,
        )
        result = run_arm(
            n, cfg, observations, args.progress, args.checkpoint, out_dir
        )
        payload[f'theta_est_winsorized_n{n}'] = result['theta_est']
        payload[f'theta_est_delta_winsorized_n{n}'] = result['theta_est_delta']
        payload[f'y_est_winsorized_n{n}'] = result['y_est']
        put_status(
            payload, 'winsorized', n, result['status'], result['requested'],
            result['done'], result['failure_step'], result['failure_message'],
        )
        atomic_savez(partial_path, payload)
        print(
            f'  статус: {result["status"]}, '
            f'{result["done"]}/{result["requested"]}',
            flush=True,
        )

    final_path = out_dir / 'comparison.npz'
    atomic_savez(final_path, payload)
    print(f'\nсохранено: {final_path}', flush=True)


if __name__ == '__main__':
    main()
