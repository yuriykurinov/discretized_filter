import matplotlib.pyplot as plt
import numpy as np
from config import N, T, ht, M, t_net_filtering
from math import ceil

theta_labels = ['$\\theta_t = e_1$', '$\\theta_t = e_2$',
                '$\\theta_t = e_3$', '$\\theta_t = e_4$']

theta_colors = ['green', 'chocolate', 'darkred', 'black']

abc = ['а', 'б', 'в', 'г']

def plot_theta_background(ax, theta, t, theta_labels, theta_colors, T, alpha=0.3):
    start = np.roll(t, 1)
    start[0] = 0
    lines = []
    for n in range(N):
        line = ax.fill_between(np.sort(np.concatenate([start, t])), 0, T,
                               where=np.repeat((theta==n), 2),
                               color=theta_colors[n],
                               alpha=alpha, transform=ax.get_xaxis_transform())
        lines.append(line)
    return lines

def plot_theta(theta, t, theta_opt):

    fig, ax = plt.subplots(N, figsize=((1 + np.sqrt(5))/2 * 8, 8), layout='constrained')

    for n in range(N):
        line1, = ax[n].step(
            [0] + list(t), 
            [(theta[0] == n)] + list(theta == n), 
            lw=1.5
        )

        line4, = ax[n].plot(t_net_filtering, theta_opt.T[n],
                            color='red', lw=1)

        ax[n].set(
            xlim=(0, T),
            ylim=(-0.1, 1.1)
        )
        ax[n].set_xlabel('({})'.format(abc[n]), fontsize=13)
        ax[n].set_ylabel('$\\theta_t^{}$'.format(n+1), fontsize=14)
        ax[n].tick_params(labelsize=14)
        ax[n].set_xticks(
            [0, 20, 40, 60, 80, 100], 
            ['0', '20', '40', '60', '80', '$t$']
        )
        ax[n].set_yticks(ticks=[0, 1])
        ax[n].spines["bottom"].set_color("k")
        ax[n].spines["top"].set_color("k")
        ax[n].spines["left"].set_color("k")
        ax[n].spines["right"].set_color("k")

    fig.legend(
        fontsize=16,
        handles=[line1,
                    line4,
                    #line5
                ],
        labels=['$\\theta_t$',
                '$\\hat \\theta_t$',
                #'$\\tilde \\theta_t$'
                ],
        loc='outside lower center',
        ncol=4,
        framealpha=1,
        frameon=False
    )

    return fig, ax

def plot_y(theta, y, t, y_opt):
    fig, ax = plt.subplots(M, figsize=(16, 8), layout='constrained')

    lines = []
    labels = []

    for m in range(M):
        ax[m].set_ylabel('$Y^{}_t$'.format(m+1), fontsize=16)
        y_tmp = y[:, m]
        line1, = ax[m].step(
            [0] + list(t),
            [y_tmp[0]] + list(y_tmp),
            where='pre', lw=3, color='white'
        )
        line4, = ax[m].plot(
            t_net_filtering, y_opt[:, m],
            color='red', lw=1.2
        )

        ax[m].set_xlim(0, T)
        ax[m].set_xlabel(f'({abc[m]})', fontsize=16)
        ax[m].set_xticks(
            [0, 20, 40, 60, 80, 100], 
            ['0', '20', '40', '60', '80', '$t$'],
        )
        ax[m].set_frame_on(True)
        ax[m].spines["bottom"].set_color("k")
        ax[m].spines["top"].set_color("k")
        ax[m].spines["left"].set_color("k")
        ax[m].spines["right"].set_color("k")
        ax[m].tick_params(labelsize=14)

        line1t, line2t, line3t, line4t = plot_theta_background(
            ax[m], theta, t, theta_labels, theta_colors, T, alpha=0.3
        )

    #ax[0].set_ylim(0.015, 0.065)
    #ax[0].set_yticks([0.02, 0.04, 0.06], ['0,02', '0,04', '0,06'])
    #ax[1].set_ylim(0.00, 0.11)
    #ax[1].set_yticks([0.01, 0.04, 0.07, 0.1], ['0,01', '0,04', '0,07', '0,10'])

    labels = theta_labels + ['$Y_t$', '$\\hat{Y}_t$']

    handles = [line1t, line2t, line3t, line4t, line1, line4]

    fig.legend(
        fontsize=18,
        handles=handles,
        labels=labels,
        loc='outside lower center',
        ncol=len(labels),
        framealpha=1,
        facecolor='gainsboro'
    )

    fig.align_ylabels()

    return fig, ax


def plot_obs(theta, y, t, dxi, deta):

    eta = np.cumsum(deta)

    fig, ax = plt.subplots(M, figsize=(16, 8), layout='constrained')

    ######   Y1   ######

    line1, = ax[0].plot(
        t_net_filtering[1:dxi.shape[0]], 
        dxi[1:] / ht, 
        lw=0.6, 
        color='firebrick'
    )
    line2, = ax[0].step(
        np.concatenate([[0], t]),
        np.concatenate([[y[0, 0]], y[:, 0]]), 
        color='k'
    )
    
    plot_theta_background(ax[0], theta, t, theta_labels, theta_colors, T)

    ax[0].set_xlabel(f'({abc[0]})', fontsize=14)
    ax[0].set_ylabel('$\\frac{\\Delta \\xi}{\\Delta_t}$', fontsize=16)
    #ax[0].set_ylim(1e-2, 7e-2)

    ######   Y2   ######

    ax_2 = ax[1].twinx()
    line4, = ax_2.step([0] + list(t), [y[0, 1]] + list(y[:, 1]), color='navy')
    ax_2.axis([0, T, 0, np.max(y[:, 1])+0.005])
    ax_2.tick_params(labelsize=14)
    ax_2.set_ylabel('$Y_t^2$', fontsize=16)
    ax_2.set_yticks([0, round(np.max(y.T[1]), 2)])

    line3, = ax[1].plot(t_net_filtering, eta, color='white')

    line1t, line2t, line3t, line4t = plot_theta_background(
        ax[1], theta, t, theta_labels, theta_colors, T
    )
    ax[1].set_xlabel(f'({abc[1]})', fontsize=13)
    ax[1].set_ylabel('$\\eta_t^1$', fontsize=16)
    ax[1].set_ylim(0, eta[-1]+5)
    ax[1].set_yticks([0, eta[-1]])

    for m in range(M):
        ax[m].set_xticks(
            [0, 20, 40, 60, 80, 100], 
            ['0', '20', '40', '60', '80', '$t$']
        )
        ax[m].spines["bottom"].set_color("k")
        ax[m].spines["top"].set_color("k")
        ax[m].spines["left"].set_color("k")
        ax[m].spines["right"].set_color("k")
        ax[m].tick_params(labelsize=16)
        ax[m].set_xlim(0, T)
    #ax[0].set_yticks([0.01, 0.03, 0.05, 0.07], ['0,01', '0,03', '0,05', '0,07'])
    #ax_2.set_yticks([0., 0.05, 0.1], ['0,00', '0,05', '0,10'])

    handles = [
        line1t, line2t, line3t, line4t, 
        line1, line2, line3, line4
    ]

    labels = theta_labels + ['$\\Delta \\xi / \\Delta_t$', '$Y^1_t$', '$\\eta_t$', '$Y_t^2$']

    fig.legend(
        fontsize=16,
        handles=handles,
        labels=labels,
        loc='outside lower center',
        ncols=len(handles),
        facecolor='gainsboro'
    )

    return fig, ax