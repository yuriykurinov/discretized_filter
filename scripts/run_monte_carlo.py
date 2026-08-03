"""
Оценка RMSE фильтра по Монте-Карло на большом числе случайных траекторий.
Порт multiple_paths_for_server.py на систему configs/.

Многочасовая задача, поэтому накопленные res_theta/res_y и число пройденных
траекторий периодически чекпойнтятся в
saved_path_<exp_id>/rmse_<paths>_paths/checkpoint.npz -- при обрыве (нода
упала, кончился лимит времени SLURM) можно продолжить с --resume, не теряя
всю уже посчитанную работу.

При --resume уже пройденные траектории (theta, y, t и наблюдения)
перегенерируются заново -- чтобы ГПСЧ (и джитованный numba-поток, и host
numpy) продвинулся ровно до того состояния, в котором он был в момент
чекпойнта, и продолжение бит-в-бит совпало с непрерывным запуском -- но
дорогой цикл обновления фильтра для них не повторяется: он детерминирован
по уже сгенерированным наблюдениям и в накопление res_theta/res_y уже вошёл
через сохранённый чекпойнт.

Запуск: python scripts/run_monte_carlo.py --config <имя> --paths 10000
(или переменная окружения DFILTER_CONFIG).
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import os
import pickle
from time import time

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.paths import saved_path_dir
from discretized_filter.utils.grids import to_discrete
from discretized_filter.utils.io import save_config_copy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Оценка RMSE фильтра по Монте-Карло.'
    )
    parser.add_argument(
        '--config', default=os.environ.get('DFILTER_CONFIG'),
        help='имя конфига из configs/ (по умолчанию -- переменная окружения '
             'DFILTER_CONFIG)',
    )
    parser.add_argument(
        '--paths', type=int, default=10_000,
        help='число случайных траекторий (по умолчанию 10000)',
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=500,
        help='чекпойнтить накопленные res_theta/res_y каждые N траекторий '
             '(по умолчанию 500)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='продолжить с последнего чекпойнта, если он есть',
    )
    args = parser.parse_args()
    if args.config is None:
        parser.error(
            '--config не задан, и переменная окружения DFILTER_CONFIG не установлена'
        )
    return args


def _config_digest(cfg):
    """
    Хеш файла активного конфига -- кладётся в чекпойнт, чтобы --resume не
    продолжил чужое накопление (см. проверку в main).
    """
    return hashlib.sha256(cfg._config_path.read_bytes()).hexdigest()[:16]


def _save_checkpoint(path, res_theta, res_y, path_num, config_digest):
    """
    Атомарная запись чекпойнта: .npz -- это zip, который пишется в файл
    постепенно, поэтому падение посреди np.savez прямо в целевой файл
    оставило бы битый архив ВМЕСТО последнего целого. Пишем во временный файл
    рядом и переставляем os.replace (атомарен в пределах одной ФС).
    """
    # имя обязано оканчиваться на .npz, иначе np.savez допишет расширение сам
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(
        tmp, res_theta=res_theta, res_y=res_y, path_num=path_num,
        config_digest=config_digest,
    )
    os.replace(tmp, path)


def main():
    args = parse_args()
    cfg = set_config(args.config)

    exp_dir = saved_path_dir(cfg.exp_id) / f'rmse_{args.paths}_paths'
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_path = exp_dir / 'checkpoint.npz'

    config_digest = _config_digest(cfg)

    completed = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        saved_digest = str(checkpoint['config_digest'])
        if saved_digest != config_digest:
            # иначе в одно накопление RMSE молча смешались бы две разные модели
            raise SystemExit(
                f'чекпойнт {checkpoint_path} посчитан другим конфигом '
                f'(хеш {saved_digest}, у текущего {config_digest}). '
                'Уберите --resume, чтобы начать заново, либо укажите тот же '
                'конфиг, которым считался чекпойнт.'
            )
        res_theta = checkpoint['res_theta']
        res_y = checkpoint['res_y']
        completed = int(checkpoint['path_num'])
        print(f'resuming from checkpoint: {completed} paths already done', flush=True)
    else:
        res_theta = np.zeros((cfg.t_net_filtering.shape[0], cfg.N))
        res_y = np.zeros((cfg.t_net_filtering.shape[0], cfg.M))

    # копию конфига сохраняем ПОСЛЕ проверки чекпойнта: иначе она затёрла бы
    # копию, сохранённую исходным запуском, вместе с возможностью потом
    # понять, чем именно он считался
    save_config_copy(cfg.exp_id)

    start = time()

    for path_num in range(1, args.paths + 1):
        theta, y, t = sparse_mc(
            cfg.p0, cfg.Lambda, cfg.lam, cfg.T, cfg.get_y, cfg.y_intervals
        )
        observations = cfg.get_obs(cfg.t_net_filtering, theta, y, t)

        if path_num <= completed:
            # уже учтено в чекпойнте -- траектория сгенерирована только
            # ради продвижения ГПСЧ, дорогой фильтр для неё не гоняем
            continue

        filt = Filter(
            cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
            cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density,
            n_points=cfg.n_points, two_jumps=cfg.two_jumps,
        )

        est = filt.estimate()
        theta_est = [est[0]]
        y_est = [est[1]]

        for i, obs in enumerate(observations, start=1):
            filt.update(obs)
            est = filt.estimate()
            if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
                print(f'nan on {i}-th iter')
                break
            theta_est.append(est[0])
            y_est.append(est[1])

        dtheta = to_discrete(
            np.vstack([np.int64(theta == i) for i in range(cfg.N)]).T,
            t, cfg.T, cfg.ht,
        )
        res_theta += (dtheta - theta_est) ** 2 / args.paths
        dY = to_discrete(y, t, cfg.T, cfg.ht)
        res_y += (dY - y_est) ** 2 / args.paths

        if path_num % max(args.paths // 100, 1) == 0:
            curr_time = time() - start
            hours = int(curr_time // 3600)
            minutes = int((curr_time % 3600) // 60)
            print(f'path {path_num}, time: {hours}h {minutes}m', flush=True)

        if path_num % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path, res_theta, res_y, path_num, config_digest
            )

    end = time()

    with open(exp_dir / 'res_theta.pkl', 'wb') as f:
        pickle.dump(res_theta, f)

    with open(exp_dir / 'res_y.pkl', 'wb') as f:
        pickle.dump(res_y, f)

    execution_time = end - start
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    print(f'execution time: {hours}h {minutes}m')
    print(f'exp_id: {cfg.exp_id}')


if __name__ == '__main__':
    main()
