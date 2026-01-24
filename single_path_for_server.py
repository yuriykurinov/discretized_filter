import numpy as np
from time import time

from config import *
from utils import save_path, save_config_copy
from SMJP import (
    sparse_mc, get_y_uniform, 
    make_discretized_xi, make_discretized_eta
)
from filter import Filter

save_config_copy(exp_id)

theta, y, t = sparse_mc(p0, Lambda, lam, T, get_y_uniform, y_intervals)

observations = get_obs(t_net_filtering, theta, y, t).squeeze()


filter = Filter(
    p0[:, np.newaxis] * pi_uniform, 
    pi_uniform, M_net, F, G,
    N, Lambda, ht, delta
)


est = filter.estimate()
theta_est = [est[0]]
y_est = [est[1]]


start = time()

for i, obs in enumerate(observations, start=1):
    filter.update(obs.squeeze())
    est = filter.estimate()
    if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
        print(f'nan on {i}-th iter')
        break
    theta_est.append(est[0])
    y_est.append(est[1])

end = time()


execution_time = end - start
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
print(f"execution time: {hours}h {minutes}m")
print(f"exp_id: {exp_id}")

theta_est = np.array(theta_est)
y_est = np.array(y_est)

save_path(exp_id, theta, y, t, theta_est, y_est, observations)
