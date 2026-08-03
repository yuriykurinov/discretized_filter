"""
Прогон фильтра на одной случайной траектории (theta, Y) с сохранением
результата в saved_path_<exp_id>/. Порт single_path_for_server.py на
систему configs/.

Запуск: python scripts/run_single_path.py --config <имя>
(или переменная окружения DFILTER_CONFIG).
"""
import _bootstrap  # noqa: F401

import argparse
import os
from time import time

import numpy as np

from discretized_filter.config import set_config
from discretized_filter.core.filter import Filter
from discretized_filter.core.smjp import sparse_mc
from discretized_filter.utils.io import save_config_copy, save_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Прогон дискретизированного фильтра на одной траектории.'
    )
    parser.add_argument(
        '--config', default=os.environ.get('DFILTER_CONFIG'),
        help='имя конфига из configs/ (по умолчанию -- переменная окружения '
             'DFILTER_CONFIG)',
    )
    args = parser.parse_args()
    if args.config is None:
        parser.error(
            '--config не задан, и переменная окружения DFILTER_CONFIG не установлена'
        )
    return args


def main():
    args = parse_args()
    cfg = set_config(args.config)

    save_config_copy(cfg.exp_id)

    theta, y, t = sparse_mc(
        cfg.p0, cfg.Lambda, cfg.lam, cfg.T, cfg.get_y, cfg.y_intervals
    )

    # cfg.get_obs возвращает приращения формы (T-1, K) -- без .squeeze()
    # и без np.array([obs]) при подаче в filter.update, как в старом скрипте:
    # старый get_obs отдавал (T-1, 1, K), новый -- сразу (T-1, K).
    observations = cfg.get_obs(cfg.t_net_filtering, theta, y, t)

    filt = Filter(
        cfg.pi_init, cfg.pi, cfg.M_net, cfg.C,
        cfg.N, cfg.Lambda, cfg.ht, cfg.delta, cfg.obs_density,
        n_points=cfg.n_points, two_jumps=cfg.two_jumps,
    )

    est = filt.estimate()
    theta_est = [est[0]]
    y_est = [est[1]]

    start = time()

    for i, obs in enumerate(observations, start=1):
        filt.update(obs)
        est = filt.estimate()
        if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
            print(f'nan on {i}-th iter')
            break
        theta_est.append(est[0])
        y_est.append(est[1])

    end = time()

    execution_time = end - start
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    print(f'execution time: {hours}h {minutes}m')
    print(f'exp_id: {cfg.exp_id}')

    theta_est = np.array(theta_est)
    y_est = np.array(y_est)

    save_path(cfg.exp_id, theta, y, t, theta_est, y_est, observations)


if __name__ == '__main__':
    main()
