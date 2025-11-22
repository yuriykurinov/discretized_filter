import numpy as np
import pickle
import os

from config import *
from SMJP import (
    sparse_mc, get_y_uniform, 
    make_discretized_xi, make_discretized_eta
)
from filter import Filter

exp_id = '0'

theta, y, t = sparse_mc(p0, Lambda, lam, T, get_y_uniform)

dxi = make_discretized_xi(t_net_filtering, g, sigma, theta, y, t)
deta = make_discretized_eta(t_net_filtering, h, theta, y, t)

tmp = np.stack([sigma(-1, M_net, -1)**2, h(-1, M_net, -1)], axis=-1)
F = np.repeat(tmp[np.newaxis, ...], N, axis=0)
tmp = np.stack([g(-1, M_net, -1), h(-1, M_net, -1)], axis=-1)
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
    # if np.any(np.isnan(est[0])) or np.any(np.isnan(est[1])):
    #     print(f'nan on {i}-th iter')
    #     break
    theta_est.append(est[0])
    y_est.append(est[1])

theta_est = np.array(theta_est)
y_est = np.array(y_est)


exp_path = f'saved_path_{exp_id}'
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

with open(os.path.join(exp_path, 'theta_est.pkl'), 'wb') as f:
    pickle.dump(theta_est, f)
with open(os.path.join(exp_path, 'y_est.pkl'), 'wb') as f:
    pickle.dump(y_est, f)
with open(os.path.join(exp_path, 'theta.pkl'), 'wb') as f:
    pickle.dump(theta, f)
with open(os.path.join(exp_path, 'y.pkl'), 'wb') as f:
    pickle.dump(y, f)
with open(os.path.join(exp_path, 't.pkl'), 'wb') as f:
    pickle.dump(t, f)
with open(os.path.join(exp_path, 'dxi.pkl'), 'wb') as f:
    pickle.dump(dxi, f)
with open(os.path.join(exp_path, 'deta.pkl'), 'wb') as f:
    pickle.dump(deta, f)

