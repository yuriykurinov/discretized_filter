import numpy as np

from config import *
from utils import save_path
from SMJP import (
    sparse_mc, get_y_uniform, 
    make_discretized_xi, make_discretized_eta
)
from filter import Filter

exp_id = 'non_intersecting_intervals'

theta, y, t = sparse_mc(p0, Lambda, lam, T, get_y_uniform, y_intervals)

dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t)
deta = make_discretized_eta(t_net_filtering, h, theta, y, t)

tmp = np.stack([g(-1, M_net, -1), h(-1, M_net, -1)], axis=-1)
F = np.repeat(tmp[np.newaxis, ...], N, axis=0)
tmp = np.stack([sigma(-1, M_net, -1)**2, h(-1, M_net, -1)], axis=-1)
G = np.repeat(tmp[np.newaxis, ...], N, axis=0)

filter = Filter(
    p0[:, np.newaxis] * pi_uniform, 
    pi_uniform, M_net, F, G,
    N, Lambda, ht, delta
)


est = filter.estimate()
theta_est = [est[0]]
y_est = [est[1]]


for i, obs in enumerate(np.stack([dxi[1:], deta[1:]], axis=-1), start=1):
    filter.update(obs)
    est = filter.estimate()
    if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
        print(f'nan on {i}-th iter')
        break
    theta_est.append(est[0])
    y_est.append(est[1])

theta_est = np.array(theta_est)
y_est = np.array(y_est)

save_path(exp_id, theta, y, t, theta_est, y_est, dxi, deta)
