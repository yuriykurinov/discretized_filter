import matplotlib.pyplot as plt
import numpy as np
from config import N, T, ht, M, t_net_filtering
from math import ceil

theta_labels_default = ['$\\theta_t = e_1$', '$\\theta_t = e_2$',
                '$\\theta_t = e_3$', '$\\theta_t = e_4$']

theta_colors_default = ['green', 'chocolate', 'darkred', 'black']

abc_default = ['а', 'б', 'в', 'г']

def plot_theta_background(ax, theta, t, theta_labels, theta_colors, T, alpha=0.3):
    start = np.roll(t, 1)
    start[0] = 0
    lines = []
    for n in range(len(theta_labels)):
        line = ax.fill_between(
            np.sort(np.concatenate([start, t])), 0, T,
            where=np.repeat((theta==n), 2),
            label=theta_labels[n],
            color=theta_colors[n],
            alpha=alpha, 
            transform=ax.get_xaxis_transform()
        )
        lines.append(line)
    return lines
