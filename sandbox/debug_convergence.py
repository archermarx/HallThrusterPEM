import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from pathlib import Path
import pickle
import time

sys.path.append('..')

from utils import UniformRV, ax_default, print_stats
from models.cc import cc_pem
from surrogates.sparse_grids import TensorProductInterpolator
from surrogates.system import SystemSurrogate
from models.simple_models import borehole_system, wing_weight_system


def cc_system():
    d = 5
    idx = list(np.arange(d))
    # UniformRV(200, 400, 'Va')
    exo_vars = [UniformRV(-8, -3, 'PB'), UniformRV(1, 5, 'T_ec'), UniformRV(0, 60, 'V_vac'),
                UniformRV(1, 10, 'P*'), UniformRV(1, 10, 'PT')]
    vars = [var for i, var in enumerate(exo_vars) if i in idx]
    coupling_vars = [UniformRV(0, 60, 'V_cc')]
    cathode = {'name': 'Cathode', 'model': cc_pem, 'truth_alpha': (), 'exo_in': idx, 'local_in': {},
               'global_out': [0], 'max_alpha': (), 'max_beta': (3,)*d, 'type': 'lagrange',
               'model_args': (), 'model_kwargs': {}}
    sys = SystemSurrogate([cathode], vars, coupling_vars, root_dir='build', suppress_stdout=True)

    return sys


def test_convergence(system='Borehole', max_iter=30, N=1000):
    # Test convergence of different model surrogates
    sys = None
    plot_idx = None
    match system:
        case 'Borehole':
            sys = borehole_system()
            plot_idx = [0, 1]
        case 'Wing':
            sys = wing_weight_system()
            plot_idx = [0, 1, 2, 3, 4, 5]
        case 'Cathode':
            sys = cc_system()
            plot_idx = [0, 1, 2, 3, 4]

    # Random test set for percent error
    comp = sys.get_component(system)
    xt = sys.sample_exo_inputs((N,))
    yt = sys(xt, ground_truth=True)

    # 1d slice test set(s) for plotting
    bds = [var.bounds() for var in comp.x_vars]
    xs = np.zeros((len(plot_idx), N, len(bds)))
    for j in range(len(plot_idx)):
        for i in range(len(bds)):
            if i == plot_idx[j]:
                xs[j, :, i] = np.linspace(bds[i][0], bds[i][1], N)  # 1d slice of input parameter of interest
            else:
                # xs[j, :, i] = (bds[i][0] + bds[i][1]) / 2  # Middle of domain constant
                xs[j, :, i] = bds[i][0] * 1.05
    ys = sys(xs, ground_truth=True)

    stats = np.zeros((max_iter+1, 9))
    def print_iter(i, time_s):
        ysurr = sys(xt)
        # error = 100 * (np.abs(ysurr - yt) / yt)
        with np.errstate(divide='ignore'):
            error = 2 * np.abs(ysurr - yt) / (np.abs(ysurr) + np.abs(yt))
            error = error[:, 0]
            idx = np.logical_and(ysurr[:, 0] == 0, yt[:, 0] == 0)
            error[idx] = 0
        Ik = len(comp.index_set) + len(comp.candidate_set)
        res = np.array([sys.refine_level, Ik, np.min(error), np.percentile(error, 25), np.percentile(error, 50),
                        np.percentile(error, 75), np.max(error), np.mean(error), time_s])
        stats[i, :] = res
        print(f'{sys.refine_level: 10d} {Ik: 10d} {res[2]: 10.2f} {res[3]: 10.2f} '
              f'{res[4]: 10.2f} {res[5]: 10.2f} {res[6]: 10.2f} {res[7]: 10.2f}')

    print(f'{"Iteration":>10} {"len(I_k)":>10} {"Min":>10} {"25th pct":>10} {"50th pct":>10} {"75th pct":>10} '
          f'{"Max":>10} {"Mean":>10}')
    print_iter(0, 0)

    # Plot 1d surrogate slice before/after training
    fig, ax = plt.subplots(2, len(plot_idx), sharex='col')
    ysurr = sys(xs)
    for j in range(len(plot_idx)):
        ax[0, j].plot(xs[j, :, plot_idx[j]], ys[j, :, 0], '-k', label='Truth')
        ax[0, j].plot(xs[j, :, plot_idx[j]], ysurr[j, :, 0], '--r', label='Surrogate')
        ax[0, j].set_title(f'Input: {str(comp.x_vars[plot_idx[j]])}')
        ylabel = 'Initial' if j == 0 else ''
        legend = j == len(plot_idx) - 1
        ax_default(ax[0, j], '', ylabel, legend=legend)

    for i in range(1, max_iter+1):
        t1 = time.time()
        sys.refine(qoi_ind=None, N_refine=100, update_bounds=False)
        print_iter(i, time.time()-t1)

    ysurr = sys(xs)
    for j in range(len(plot_idx)):
        ax[1, j].plot(xs[j, :, plot_idx[j]], ys[j, :, 0], '-k', label='Truth')
        ax[1, j].plot(xs[j, :, plot_idx[j]], ysurr[j, :, 0], '--r', label='Surrogate')
        ylabel = 'Final' if j == 0 else ''
        ax_default(ax[1, j], 'Input parameter 1d slice', ylabel, legend=False)
    fig.set_size_inches(len(plot_idx)*3, 6)
    fig.tight_layout()
    plt.show()

    # Show convergence plot for median, mean, and max error
    fig, ax = plt.subplots()
    i = np.arange(0, stats.shape[0])
    ax.plot(i, stats[:, 4], '-g', label='Median')
    ax.plot(i, stats[:, 7], '-k', label='Mean')
    ax.plot(i, stats[:, 6], '-r', label='Max')
    ax_default(ax, 'Iteration', r'Relative percent difference', legend=True)
    ax.set_yscale('log')
    plt.show()

    # Show scaling of refinement time vs. number of candidates
    fig, ax = plt.subplots()
    ax.plot(stats[:, 1], stats[:, 8], '-k')
    ax.set_yscale('log')
    ax_default(ax, 'Number of MISC candidates', 'Refinement time (s)', legend=False)
    plt.show()


def plot_cathode():
    N = 50
    pb = np.linspace(-8, -3, N)
    Te = np.linspace(1, 5, N)

    xg, yg = np.meshgrid(pb, Te)
    xg = xg.reshape((N, N, 1))
    yg = yg.reshape((N, N, 1))
    x = np.concatenate((xg, yg), axis=-1)
    z = cc_pem(x)

    # Set up interpolant
    x_vars = [UniformRV(-8, -3, 'PB'), UniformRV(1, 5, 'Tec')]
    interp = TensorProductInterpolator((3, 3), x_vars, model=cc_pem)
    interp.set_yi()

    # Look at results near interpolation points
    pert = np.random.rand(interp.xi.shape[0], 1)*(1.01-0.99) + 0.99
    xi_pert = interp.xi * pert
    yi_pert = interp(xi_pert)
    yi_truth = cc_pem(xi_pert)
    error = 2 * np.abs(yi_pert - yi_truth) / (np.abs(yi_pert) + np.abs(yi_truth))
    idx = np.logical_and(yi_pert == 0, yi_truth == 0)
    error[idx] = 0
    print(r'1% perturbation stats near interpolation grid points:')
    print_stats(error[:, 0])

    z_interp = interp(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        error = 2 * np.abs(z_interp - z) / (np.abs(z_interp) + np.abs(z))
        idx = np.logical_and(z_interp == 0, z == 0)
        error[idx] = 0

    # Look at 2d input space (Pb, Tec)
    vmin = min(np.min(z_interp), np.min(z))
    vmax = max(np.max(z_interp), np.max(z))
    fig, ax = plt.subplots(1, 3)
    c1 = ax[0].contourf(xg.squeeze(), yg.squeeze(), z.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax[0])
    ax[0].set_title(r'$V_{cc}$ truth')
    ax_default(ax[0], r'$P_B$', r'$T_{e,c}$', legend=False)
    c2 = ax[1].contourf(xg.squeeze(), yg.squeeze(), z_interp.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    ax[1].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c2, ax=ax[1])
    ax[1].set_title(r'$V_{cc}$ Interpolant')
    ax_default(ax[1], r'$P_B$', '', legend=False)
    c3 = ax[2].contourf(xg.squeeze(), yg.squeeze(), error.squeeze(), 60, cmap=cm.viridis)
    ax[2].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c3, ax=ax[2])
    ax[2].set_title('Relative percent difference')
    ax_default(ax[2], r'$P_B$', '', legend=False)
    fig.set_size_inches(10, 4)
    fig.tight_layout()
    plt.show()

    # Look at 1d cross-section
    Tec = np.ones((N, 1)) * 3
    x = np.concatenate((pb[..., np.newaxis], Tec), axis=-1)
    ysurr = interp(x)
    ytruth = cc_pem(x)
    fig, ax = plt.subplots()
    ax.plot(pb, ytruth[:, 0], '-k', label='Model')
    ax.plot(pb, ysurr[:, 0], '--r', label='Surrogate')
    xin = np.concatenate((interp.x_grids[0][..., np.newaxis], np.ones((interp.x_grids[0].shape[0], 1))*3), axis=-1)
    yin = interp(xin)
    ax.plot(interp.x_grids[0], yin[:, 0], 'o', markersize=6, markerfacecolor='green')
    ax_default(ax, 'Background pressure magnitude (torr)', 'Cathode coupling voltage (V)', legend=True)
    ax.set_title(r'$T_{e,c} = 3$')
    plt.show()


if __name__ == '__main__':
    # test_convergence(system='Borehole')
    # test_convergence(system='Wing', max_iter=20)
    # test_convergence(system='Cathode', max_iter=30)
    plot_cathode()
