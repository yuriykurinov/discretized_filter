import numpy as np

from config import *
from utils import save_path
from SMJP import (
    sparse_mc, get_y_uniform, 
)
from filter import Filter

exp_id = 'non_intersecting_intervals'

theta, y, t = sparse_mc(p0, Lambda, lam, T, get_y_uniform, y_intervals)

observations = get_obs(t_net_filtering, theta, y, t)

filter = Filter(
    p0[:, np.newaxis] * pi_uniform, 
    pi_uniform, M_net, F, G,
    N, Lambda, ht, delta
)


est = filter.estimate()
theta_est = [est[0]]
y_est = [est[1]]


for i, obs in enumerate(observations, start=1):
    filter.update(obs)
    est = filter.estimate()
    if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
        print(f'nan on {i}-th iter')
        break
    theta_est.append(est[0])
    y_est.append(est[1])

theta_est = np.array(theta_est)
y_est = np.array(y_est)

save_path(exp_id, theta, y, t, theta_est, y_est, observations)
