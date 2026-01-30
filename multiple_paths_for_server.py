import numpy as np
from time import time

from config import *
from utils import save_path, save_config_copy, to_discrete
from SMJP import (
    sparse_mc, get_y_uniform, 
    make_discretized_xi, make_discretized_eta
)
from filter import Filter

save_config_copy(exp_id)

N_paths = 10

res_theta = np.zeros((t_net_filtering.shape[0], N))
res_y = np.zeros((t_net_filtering.shape[0], M))

start = time()

for _ in range(N_paths):
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

    for i, obs in enumerate(observations, start=1):
        filter.update(np.array([obs]))
        est = filter.estimate()
        if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
            print(f'nan on {i}-th iter')
            break
        theta_est.append(est[0])
        y_est.append(est[1])

    dtheta = to_discrete(np.vstack([np.int64(theta == i) for i in range(N)]).T, t, T, ht)
    res_theta += (dtheta - theta_est) ** 2 / N_paths
    dY = to_discrete(y, t, T, ht)
    res_y += (dY - y_est) ** 2 / N_paths
    
end = time()


import os, pickle
exp_path = f'saved_path_{exp_id}/rmse_{N_paths}_paths'
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

with open(exp_path + "/res_theta.pkl", "wb") as f:
    pickle.dump(res_theta, f)

with open(exp_path + "/res_y.pkl", "wb") as f:
    pickle.dump(res_y, f)

execution_time = end - start
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
print(f"execution time: {hours}h {minutes}m")
print(f"exp_id: {exp_id}")
