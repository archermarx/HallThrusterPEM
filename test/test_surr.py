import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from matplotlib import cm
import time
import warnings
import sys
import logging
import dill
from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
# from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)
# from mpi4py.futures import MPICommExecutor

sys.path.append('..')

from utils import ax_default, print_stats, UniformRV
from surrogates.system import SystemSurrogate
from surrogates.sparse_grids import SparseGridSurrogate, TensorProductInterpolator
from models.simple_models import tanh_func, custom_nonlinear, fire_sat_system, fake_pem

FORMATTER = logging.Formatter("%(asctime)s — [%(levelname)s] — %(name)s — %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(FORMATTER)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def test_tensor_product_1d():
    beta = [2]
    x_var = [UniformRV(0, 1)]
    x_grid = np.linspace(0, 1, 100).reshape((100, 1))
    y_grid = tanh_func(x_grid)
    interp = TensorProductInterpolator(beta, x_var, model=tanh_func)
    interp.set_yi()
    y_interp = interp(x_grid)

    # Refine
    beta2 = [4]
    interp2 = interp.refine(beta2)
    y2_interp = interp2(x_grid)

    # Compute errors
    N = 1000
    xtest = np.random.rand(N, 1)
    ytest = interp(xtest)
    ytruth = tanh_func(xtest)
    yerror = (np.abs(ytest - ytruth) / ytruth) * 100
    y2test = interp2(xtest)
    y2error = (np.abs(y2test - ytruth) / ytruth) * 100

    # Print results
    print('Interpolation errors:')
    print_stats(yerror)
    print('\nRefined interpolation errors:')
    print_stats(y2error)

    fig, ax = plt.subplots()
    ax.plot(x_grid, y_grid, '-k', label='Truth')
    ax.plot(interp.xi, interp.yi, 'or', markersize=8, label=r'$x_{interp}$')
    ax.plot(x_grid, y_interp, '-r', label=r'Interpolant')
    ax.plot(interp2.xi, interp2.yi, 'ob', markersize=4, label=r'$x$ refined')
    ax.plot(x_grid, y2_interp, '-b', label=r'Refined')
    ax_default(ax, r'$x$', r'$f(x)$', legend=True)
    fig.tight_layout()
    plt.show()


def test_tensor_product_2d():
    # Test 2d interpolation
    # def bb_2d_func(x):
    #     y = np.cos(2*np.pi*x[..., 0])*np.cos(2*np.pi*x[..., 1])
    #     return y[..., np.newaxis]
    bb_2d_func = lambda x: custom_nonlinear(x, env_var=0.2**2, wave_amp=0.3)
    beta = [5, 5]
    x_vars = [UniformRV(0, 1), UniformRV(0, 1)]
    N = 50
    x_grid = np.linspace(0, 1, N)
    xg, yg = np.meshgrid(x_grid, x_grid)
    xg = xg.reshape((N, N, 1))
    yg = yg.reshape((N, N, 1))
    x = np.concatenate((xg, yg), axis=-1)
    z = bb_2d_func(x)

    # Set up interpolant
    interp = TensorProductInterpolator(beta, x_vars, model=bb_2d_func)
    interp.set_yi()
    z_interp = interp(x)
    error = np.abs(z_interp - z)

    # Refine interpolant
    beta2 = [5, 9]
    interp2 = interp.refine(beta2)
    z2_interp = interp2(x)
    error2 = np.abs(z2_interp - z)
    vmin = min(np.min(z_interp), np.min(z), np.min(z2_interp))
    vmax = max(np.max(z_interp), np.max(z), np.max(z2_interp))
    emin = min(np.min(error), np.min(error2))
    emax = max(np.max(error), np.max(error2))

    fig, ax = plt.subplots(2, 3)
    c1 = ax[0, 0].contourf(xg.squeeze(), yg.squeeze(), z.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax[0, 0])
    ax[0, 0].set_title('True function')
    ax_default(ax[0, 0], r'$x_1$', r'$x_2$', legend=False)
    c2 = ax[0, 1].contourf(xg.squeeze(), yg.squeeze(), z_interp.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    ax[0, 1].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c2, ax=ax[0, 1])
    ax[0, 1].set_title('Interpolant')
    ax_default(ax[0, 1], r'$x_1$', '', legend=False)
    c3 = ax[0, 2].contourf(xg.squeeze(), yg.squeeze(), error.squeeze(), 60, cmap=cm.viridis, vmin=emin, vmax=emax)
    ax[0, 2].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c3, ax=ax[0, 2])
    ax[0, 2].set_title('Absolute error')
    ax_default(ax[0, 2], r'$x_1$', '', legend=False)
    c1 = ax[1, 0].contourf(xg.squeeze(), yg.squeeze(), z.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax[1, 0])
    ax[1, 0].set_title('True function')
    ax_default(ax[1, 0], r'$x_1$', r'$x_2$', legend=False)
    c2 = ax[1, 1].contourf(xg.squeeze(), yg.squeeze(), z2_interp.squeeze(), 60, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    ax[1, 1].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c2, ax=ax[1, 1])
    ax[1, 1].set_title('Refined')
    ax_default(ax[1, 1], r'$x_1$', '', legend=False)
    c3 = ax[1, 2].contourf(xg.squeeze(), yg.squeeze(), error2.squeeze(), 60, cmap=cm.viridis, vmin=emin, vmax=emax)
    ax[1, 2].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
    plt.colorbar(c3, ax=ax[1, 2])
    ax[1, 2].set_title('Absolute error')
    ax_default(ax[1, 2], r'$x_1$', '', legend=False)
    fig.set_size_inches(15, 11)
    fig.tight_layout()
    plt.show()


def test_component():
    # Simple cos test from Jakeman (2022)
    def model(x, alpha):
        alpha = np.atleast_1d(alpha)  # (1,)
        eps = (1/5) * 2.0**(-alpha[0])
        y = np.cos(np.pi/2 * (x + 4/5 + eps))
        return y

    def model_truth(x):
        return np.cos(np.pi/2 * (x + 4/5))

    # Construct MISC surrogate from an index set
    Ik = [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((2,), (0,)), ((1,), (1,)), ((0,), (2,)), ((1,), (2,)),
          ((2,), (1,)), ((2,), (2,))]
    x_vars = [UniformRV(-1, 1)]
    truth_alpha = (15,)
    comp = SparseGridSurrogate(Ik, x_vars, model, truth_alpha)
    N = 100
    xg = np.linspace(-1, 1, N).reshape((N, 1))
    yt = model_truth(xg)
    y_surr = comp(xg)
    print(comp)

    # Plot results for each fidelity of the MISC surrogate
    fig, axs = plt.subplots(3, 3, sharey='row', sharex='col')
    for alpha in range(3):
        for beta in range(3):
            ax = axs[2-alpha, beta]
            surr = comp.get_sub_surrogate((alpha,), (beta,), include_grid=True)
            s = f'$\hat{{f}}_{{{alpha}, {beta}}}$'
            ax.plot(xg, surr(xg), '--k', label=r'{}'.format(s), linewidth=1.5)
            s = f'$\hat{{f}}_{alpha}$'
            ax.plot(xg, model(xg, alpha), '--b', label=r'{}'.format(s), linewidth=2)
            ax.plot(xg, yt, '-r', label=r'$f$', linewidth=2)
            ax.plot(surr.xi, surr.yi, 'or')
            xlabel = r'$x$' if alpha == 0 else ''
            ylabel = r'$f(x)$' if beta == 0 else ''
            ax_default(ax, xlabel, ylabel, legend=True)

    fig.text(0.5, 0.02, r'Increasing surrogate fidelity ($\beta$) $\rightarrow$', ha='center', fontweight='bold')
    fig.text(0.02, 0.5, r'Increasing model fidelity ($\alpha$) $\rightarrow$', va='center', fontweight='bold', rotation='vertical')
    fig.set_size_inches(3 * 3, 3 * 3)
    fig.tight_layout(pad=3, w_pad=1, h_pad=1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(xg, yt, '-r', linewidth=2, label='Model')
    ax.plot(xg, y_surr, '--k', linewidth=1.5, label='MISC surrogate')
    ax_default(ax, r'$x$', r'$f(x)$', legend=True)
    plt.show()


def test_high_dimension():
    def rosenbrock(x):
        y = np.zeros(x.shape[:-1] + (1,))
        for i in range(x.shape[-1] - 1):
            y[..., 0] += 100 * (x[..., i+1] - x[..., i]**2)**2 + (x[..., i]-1)**2
        return y + 1

    # Construct interpolant
    dim = 8
    x_bds = [(-5, 10) for i in range(dim)]
    beta = [2]*dim
    interp = TensorProductInterpolator(beta, x_bds, model=rosenbrock)
    interp.set_yi()

    # Test and compute error
    N = 1000
    x_test = np.random.rand(N, dim)*(10 - (-5)) + (-5)
    t1 = time.time()
    y_test = interp(x_test)
    print(f'Interpolation time: {time.time() - t1:.3} s')
    y_truth = rosenbrock(x_test)
    error = np.abs(y_test - y_truth) / y_truth
    fig, ax = plt.subplots()
    ax.boxplot(error, showfliers=False)
    # ax.hist(error * 100, density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
    ax_default(ax, '', r'Percent error (%)', legend=False)
    plt.show()

    print_stats(error)


def test_system_surrogate():
    # Figure 6 in Jakeman 2022
    def coupled_system(D1, D2, Q1, Q2):
        def f1(x, alpha, *args, **kwargs):
            eps = 10 ** (-float(alpha[0]))
            q = np.arange(1, Q1+1).reshape((1,)*len(x.shape[:-1]) + (Q1,))
            return (x[..., 0, np.newaxis] ** (q-1)) * np.sin(np.sum(x, axis=-1, keepdims=True) + eps)

        def f2(x, alpha, *args, **kwargs):
            eps = 10 ** (-float(alpha[0]))
            q = np.arange(1, Q2+1).reshape((1,)*len(x.shape) + (Q2,))
            prod1 = np.prod(x[..., D2:, np.newaxis] ** (q) - eps, axis=-2)  # (..., Q2)
            prod2 = np.prod(x[..., :D2], axis=-1, keepdims=True)              # (..., 1)
            return prod1 * prod2

        def f3(x, alpha, *args, D3=D1, **kwargs):
            eps = 10 ** (-float(alpha[0]))
            prod1 = np.exp(-np.sum((x[..., D3:] - eps) ** 2, axis=-1))  # (...,)
            prod2 = 1 + (25/16)*np.sum(x[..., :D3], axis=-1) ** 2       # (...,)
            return np.expand_dims(prod1 / prod2, axis=-1)               # (..., 1)

        def f(x):
            # Ground truth (arbitrary high alpha)
            alpha = (15,)
            x1 = x[..., :D1]
            y1 = f1(x1, alpha)
            x2 = np.concatenate((x[..., D1:], y1), axis=-1)
            y2 = f2(x2, alpha)
            x3 = np.concatenate((x1, y2), axis=-1)
            y3 = f3(x3, alpha)
            return np.concatenate((y1, y2, y3), axis=-1)

        return f1, f2, f3, f

    # Hook up the 'wiring' for this example feedforward system
    D1 = 1
    D2 = D1
    Q1 = 1
    Q2 = Q1
    alpha = (15,)
    f1, f2, f3, f = coupled_system(D1, D2, Q1, Q2)
    comp1 = {'name': 'Cathode', 'model': f1, 'truth_alpha': alpha, 'exo_in': list(np.arange(0, D1)),
             'local_in': {}, 'global_out': list(np.arange(0, Q1)), 'max_alpha': (5,), 'max_beta': (3,)*D1}
    comp2 = {'name': 'Thruster', 'model': f2, 'truth_alpha': alpha, 'exo_in': list(np.arange(D1, D1+D2)),
             'max_alpha': (5,), 'max_beta': (3,)*(D2+Q1), 'local_in': {'Cathode': list(np.arange(0, Q1))},
             'global_out': list(np.arange(Q1, Q1+Q2))}
    comp3 = {'name': 'Plume', 'model': f3, 'truth_alpha': alpha, 'exo_in': list(np.arange(0, D1)), 'max_alpha': (5,),
             'local_in': {'Thruster': list(np.arange(0, Q2))}, 'global_out': [Q1+Q2], 'max_beta': (3,)*(D1+Q2)}
    components = [comp1, comp2, comp3]
    exo_vars = [UniformRV(0, 1) for i in range(D1+D2)]
    coupling_bds = [UniformRV(0, 1) for i in range(Q1+Q2+1)]
    sys = SystemSurrogate(components, exo_vars, coupling_bds)

    # Test example
    N = 5000
    x = np.random.rand(N, D1+D2)
    y = f(x)
    y_surr = sys(x, ground_truth=True)

    error = np.abs(y - y_surr)  # (N, ydim)
    for i in range(error.shape[-1]):
        print(f'Absolute error in y{i}:')
        print_stats(error[:, i])

    # Show coupling variable pdfs
    fig, ax = plt.subplots()
    ls = ['-r', '--k', ':b']
    pts = np.linspace(0, 1, 100)
    for i in range(3):
        label_str = f'$\\rho(y_{{{i+1}}})$'
        kernel = gaussian_kde(y_surr[:, i])
        # ax[i].hist(y_surr[:, i], density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
        ax.plot(pts, kernel(pts), ls[i], label=label_str)
    ax_default(ax, r'$y$', 'PDF', legend=True)
    # fig.set_size_inches(9, 3)
    fig.tight_layout()
    plt.show()


def test_feedforward():
    # Figure 5 in Jakeman 2022
    def coupled_system():
        def f1(x, alpha):
            return x * np.sin(np.pi * x)
        def f2(x, alpha):
            return 1 / (1 + 25*x**2)
        def f(x, alpha):
            return f2(f1(x, alpha), alpha)
        return f1, f2, f

    f1, f2, f = coupled_system()
    comp1 = {'name': 'Model1', 'model': f1, 'truth_alpha': (), 'max_alpha': (), 'max_beta': (3,),
             'exo_in': [0], 'local_in': {}, 'global_out': [0], 'type': 'lagrange'}
    comp2 = {'name': 'Model2', 'model': f2, 'truth_alpha': (), 'max_alpha': (), 'max_beta': (3,),
             'exo_in': [], 'local_in': {'Model1': [0]}, 'global_out': [1], 'type': 'lagrange'}
    exo_vars = [UniformRV(0, 1)]
    coupling_bds = [UniformRV(0, 1), UniformRV(0, 1)]
    sys = SystemSurrogate([comp1, comp2], exo_vars, coupling_bds)

    x = np.linspace(0, 1, 100).reshape((100, 1))
    y1 = f1(x, ())
    y2 = f(x, ())
    y_surr = sys(x, ground_truth=True)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, y1, '-r', label='$f_1(x)$')
    ax[0].plot(np.squeeze(x), y_surr[:, 0], '--k', label='Surrogate')
    ax_default(ax[0], '$x$', '$f(x)$', legend=True)
    ax[1].plot(x, y2, '-r', label='$f(x)$')
    ax[1].plot(np.squeeze(x), y_surr[:, 1], '--k', label='Surrogate')
    ax_default(ax[1], '$x$', '$f(x)$', legend=True)
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    plt.show()


def test_lls():
    # Test constrained linear least squares
    from scipy.linalg import lapack
    import time
    X = 100
    Y = 100
    M = 10
    N = 10
    P = 1
    tol = 1e-10

    A = np.random.rand(X, Y, M, N)
    b = np.random.rand(X, Y, M, 1)
    C = np.random.rand(X, Y, P, N)
    d = np.random.rand(X, Y, P, 1)

    # custom solver
    t1 = time.time()
    alpha = np.squeeze(SystemSurrogate.constrained_lls(A, b, C, d), axis=-1)  # (*, N)
    t2 = time.time()

    # Built in scipy solver
    alpha2 = np.zeros((X, Y, N))
    t3 = time.time()
    for i in range(X):
        for j in range(Y):
            Ai = A[i, j, ...]
            bi = b[i, j, ...]
            Ci = C[i, j, ...]
            di = d[i, j, ...]
            ret = lapack.dgglse(Ai, Ci, bi, di)
            alpha2[i, j, :] = ret[3]
    t4 = time.time()

    # Results
    diff = alpha - alpha2
    if np.any(np.max(np.abs(diff), axis=-1) > tol):
        print(f'Greater than tolerance detected:')
        print(np.max(np.abs(diff), axis=-1))
    print(f'Custom time: {t2-t1} s. Scipy time: {t4-t3} s.')


def test_fire_sat(filename=None):
    # Test the fire satellite coupled system from Chaudhuri (2018)
    if filename is not None:
        sys = SystemSurrogate.load_from_file(filename)
    else:
        sys = fire_sat_system()
        sys.build_system(max_iter=5, max_tol=1e-3, max_runtime=3600)
        e = 1
        # with MPICommExecutor(MPI.COMM_WORLD, root=0) as e:
        #     if e is not None:
        #         sys = fire_sat_system()
        #         sys.set_executor(e)
        #         sys.build_system(max_iter=10, max_tol=1e-3, max_runtime=3600)

    if e is not None:
        x = sys.sample_exo_inputs((1000,))
        logger.info('---Evaluating ground truth system on test set---')
        yt = sys(x, ground_truth=True, verbose=True)
        logger.info('---Evaluating system surrogate on test set---')
        ysurr = sys(x, verbose=True)

        # Print test results
        error = 100 * (np.abs(ysurr - yt)) / yt
        for i in range(yt.shape[1]):
            logger.info(f'Test set percent error results for QoI {i}')
            print_stats(error[:, i], logger=logger)

        # Plot some output histograms
        fig, ax = plt.subplots(1, 3)
        ax[0].hist(yt[:, 0], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[0].hist(ysurr[:, 0], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4, label='Surrogate')
        ax[1].hist(yt[:, 7], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[1].hist(ysurr[:, 7], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4, label='Surrogate')
        ax[2].hist(yt[:, 8], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[2].hist(ysurr[:, 8], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4, label='Surrogate')
        ax_default(ax[0], 'Satellite velocity ($m/s$)', '', legend=True)
        ax_default(ax[1], 'Solar panel area ($m^2$)', '', legend=True)
        ax_default(ax[2], 'Attitude control power ($W$)', '', legend=True)
        fig.set_size_inches(9, 3)
        fig.tight_layout()
        plt.show()
        fig.savefig('test_surr.png', dpi=300, format='png')


def test_fpi():
    # Test fixed point iteration implementation against scipy fsolve
    f1 = lambda x, alpha: -x[..., 0:1]**3 + 2 * x[..., 1:2]**2
    f2 = lambda x, alpha: 3*x[..., 0:1]**2 + 4 * x[..., 1:2]**(-2)
    comp1 = {'name': 'm1', 'model': f1, 'truth_alpha': (), 'exo_in': [0], 'local_in': {'m2': [0]}, 'global_out': [0],
             'max_alpha': (), 'max_beta': 5}
    comp2 = {'name': 'm2', 'model': f2, 'truth_alpha': (), 'exo_in': [0], 'local_in': {'m1': [0]}, 'global_out': [1],
             'max_alpha': (), 'max_beta': 5}
    exo_vars = [UniformRV(0, 4)]
    coupling_bds = [UniformRV(1, 10), UniformRV(1, 10)]
    sys = SystemSurrogate([comp1, comp2], exo_vars, coupling_bds)

    # Test on random x against scipy.fsolve
    N = 100
    tol = 1e-12
    x0 = np.array([5.5, 5.5])
    exo = sys.sample_exo_inputs((N,))
    y_surr = sys(exo, ground_truth=True, anderson_mem=10, max_fpi_iter=200, fpi_tol=tol)  # (N, 2)
    nan_idx = list(np.any(np.isnan(y_surr), axis=-1).nonzero()[0])
    y_true = np.zeros((N, 2))
    bad_idx = []
    warnings.simplefilter('error')
    for i in range(N):
        def fun(x):
            y1 = x[0]
            y2 = x[1]
            res1 = -exo[i, 0]**3 + 2*y2**2 - y1
            res2 = 3*exo[i, 0]**2 + 4*y1**(-2) - y2
            return [res1, res2]

        try:
            y_true[i, :] = fsolve(fun, x0, xtol=tol)
        except Exception as e:
            bad_idx.append(i)

    print(f'Bad fpi indices: {nan_idx}. Input = {exo[nan_idx, 0]}')
    print(f'Bad fsolve indices: {bad_idx}. Input = {exo[bad_idx, 0]}')

    y_surr = np.delete(y_surr, nan_idx + bad_idx, axis=0)
    y_true = np.delete(y_true, nan_idx + bad_idx, axis=0)
    error = np.max(np.abs(y_surr - y_true), axis=-1)
    print(f'Max error: {np.max(error)}')

    fig, ax = plt.subplots()
    ax.hist(error, density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Absolute error')
    plt.show()


def test_fake_pem():
    models = fake_pem()
    cathode = {'name': 'Cathode', 'model': models[0], 'truth_alpha': (), 'exo_in': [0], 'local_in': {},
               'global_out': [0]}
    thruster = {'name': 'Thruster', 'model': models[1], 'truth_alpha': (), 'exo_in': [0, 1],
                'local_in': {'Cathode': [0], 'Plume': [0], 'Dump': [0]}, 'global_out': [1, 2]}
    plume = {'name': 'Plume', 'model': models[2], 'truth_alpha': (), 'exo_in': [0], 'local_in': {'Thruster': [0]},
             'global_out': [3, 4]}
    dump = {'name': 'Dump', 'model': models[3], 'truth_alpha': (), 'exo_in': [2], 'local_in': {'Plume': [1]},
            'global_out': [5]}
    chamber = {'name': 'Chamber', 'model': models[4], 'truth_alpha': (), 'exo_in': [2, 3],
               'local_in': {'Plume': [1], 'Spacecraft': [0]}, 'global_out': [6]}
    spacecraft = {'name': 'Spacecraft', 'model': models[5], 'truth_alpha': (), 'exo_in': [],
                  'local_in': {'Thruster': [1], 'Chamber': [0]}, 'global_out': [7, 8]}
    components = [cathode, thruster, plume, dump, chamber, spacecraft]
    exo_vars = [UniformRV(1e-4, 1e-2), UniformRV(200, 400), UniformRV(0, 100), UniformRV(0, 1)]  # (Pa, V, V, -)
    coupling_bds = [(0, 60), (0, 10), (0.05, 0.15), (1, 10), (1, 20), (1, 10), (-1000, 5000), (0, 100), (15, 20)]
    coupling_vars = [UniformRV(a, b) for a, b in coupling_bds]
    # coupling_bds = [(0, 60), (0, 10), (0.05, 0.15), (1e17, 1e19), (15e3, 20e3), (1e21, 1e23), (0, 60), (0, 100),
    #                 (10, 20)]  # [V, A, N, m^-3, m/s, 1/m^2-s, V, ohms, years]
    sys = SystemSurrogate(components, exo_vars, coupling_vars)

    N = 1000
    sys.estimate_coupling_bds(N, max_fpi_iter=100, anderson_mem=10)
    x = sys.sample_exo_inputs((N,))
    y = sys(x, ground_truth=True)

    # Plot some output histograms
    fig, ax = plt.subplots(1, 4)
    ax[0].hist(y[:, 0], color='red', bins=20, edgecolor='black', linewidth=1.2)
    ax[1].hist(y[:, 1], color='red', bins=20, edgecolor='black', linewidth=1.2)
    ax[2].hist(y[:, 2] * 1000, color='red', bins=20, edgecolor='black', linewidth=1.2)
    ax[3].hist(y[:, -1], color='red', bins=20, edgecolor='black', linewidth=1.2)
    ax_default(ax[0], 'Cathode coupling ($V$)', '', legend=False)
    ax_default(ax[1], 'Beam current ($A$)', '', legend=False)
    ax_default(ax[2], 'Thrust ($mN$)', '', legend=False)
    ax_default(ax[3], 'Lifetime (years)', '', legend=False)
    fig.set_size_inches(12, 4)
    fig.tight_layout()
    plt.show()


def test_system_refine():
    # Figure 5 in Jakeman 2022
    def coupled_system():
        def f1(x, alpha):
            return x * np.sin(np.pi * x)
        def f2(x, alpha):
            return 1 / (1 + 25 * x ** 2)
        return f1, f2

    f1, f2 = coupled_system()
    comp1 = {'name': 'Model1', 'model': f1, 'truth_alpha': (), 'exo_in': [0], 'local_in': {}, 'global_out': [0],
             'max_beta': (3,)}
    comp2 = {'name': 'Model2', 'model': f2, 'truth_alpha': (), 'exo_in': [], 'local_in': {'Model1': [0]},
             'global_out': [1], 'max_beta': (3,)}
    exo_vars = [UniformRV(0, 1)]
    coupling_bds = [UniformRV(0, 1), UniformRV(0, 1)]
    sys = SystemSurrogate([comp1, comp2], exo_vars, coupling_bds)

    Niter = 3
    x = np.linspace(0, 1, 100).reshape((100, 1))
    y1 = f1(x, ())
    y2 = f2(x, ())
    y3 = f2(y1, ())
    fig, ax = plt.subplots(Niter, 3, sharex='col', sharey='row')
    for i in range(Niter):
        # Plot actual function values
        ax[i, 0].plot(x, y1, '-r', label='$f_1(x)$')
        ax[i, 1].plot(x, y2, '-r', label='$f_2(y)$')
        ax[i, 2].plot(x, y3, '-r', label='$f(x)$')

        # Plot first component surrogates
        comp = sys.get_component('Model1')
        ax[i, 0].plot(x, comp(x, training=True), '--k', label='$f_1$ current')
        beta_max = 0
        for alpha, beta in comp.index_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        surr = comp.get_sub_surrogate((), (beta_max,))
        ax[i, 0].plot(surr.xi, surr.yi, 'ok', markersize=8, label='')
        for alpha, beta in comp.iterate_candidates():
            comp.update_misc_coeffs()
            yJ1 = sys(x, training=True)
            ax[i, 0].plot(x, comp(x, training=True), ':b', label='$f_1$ candidate')
            surr = comp.get_sub_surrogate(alpha, beta)
            ax[i, 0].plot(surr.xi, surr.yi, 'xb', markersize=8, label='')
        comp.update_misc_coeffs()

        # Plot second component surrogates
        comp = sys.get_component('Model2')
        ax[i, 1].plot(x, comp(x, training=True), '--k', label='$f_2$ current')
        beta_max = 0
        for alpha, beta in comp.index_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        surr = comp.get_sub_surrogate((), (beta_max,))
        ax[i, 1].plot(surr.xi, surr.yi, 'ok', markersize=8, label='')
        for alpha, beta in comp.iterate_candidates():
            comp.update_misc_coeffs()
            yJ2 = sys(x, training=True)
            ax[i, 1].plot(x, comp(x, training=True), '-.g', label='$f_2$ candidate')
            surr = comp.get_sub_surrogate(alpha, beta)
            ax[i, 1].plot(surr.xi, surr.yi, 'xg', markersize=8, label='')
        comp.update_misc_coeffs()

        # Plot integrated surrogates
        ysurr = sys(x, training=True)
        ax[i, 2].plot(x, ysurr[:, 1:2], '--k', label='$f_J$')
        ax[i, 2].plot(x, yJ1[:, 1:2], ':b', label='$f_{J_1}$')
        ax[i, 2].plot(x, yJ2[:, 1:2], '-.g', label='$f_{J_2}$')
        ax_default(ax[i, 0], '$x$', '$f_1(x)$', legend=True)
        ax_default(ax[i, 1], '$y$', '$f_2(y)$', legend=True)
        ax_default(ax[i, 2], '$x$', '$f_2(f_1(x))$', legend=True)

        # Refine the system
        sys.refine(qoi_ind=None, N_refine=100)

    fig.set_size_inches(3.5*3, 3.5*Niter)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    """Each one of these tests was used to iteratively build up the SystemSurrogate class for MD, MF interpolation"""
    # test_tensor_product_1d()
    # test_tensor_product_2d()
    # test_component()
    # test_high_dimension()
    # test_lls()
    # test_feedforward()
    # test_system_surrogate()
    # test_system_refine()
    test_fire_sat(filename=None)
    # test_fire_sat(filename=Path('save')/'sys_error.pkl')
    # test_fpi()
    # test_fake_pem()