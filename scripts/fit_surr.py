""" `fit_surr.py`

Script to be used with `train_hpc.sh` for building an MD surrogate managed by Slurm on an
HPC system. Runs the training procedure for the given PEM configuration file.

Call as:

`python fit_surr.py <config_file> [--output_dir <output_dir>] [--search] [--executor <executor>]
                                  [--max_workers <max_workers>] [--runtime_hr <runtime_hr>]
                                  [--max_iter <max_iter>] [--targets <targets>] [--train_single_fidelity]`

Arguments:

- `config_file` - the path to the `amisc` YAML configuration file with the model and input/output variable information.
- `output_dir` - the directory to save all surrogate data. Defaults to the same path as the config file.
                 If not specified as an 'amisc_{timestamp}' directory, a new directory will be created.
- `search` - whether to search for the most recent compression save file in the output directory. Defaults to False.
             Typically, should only let this be set by `train_hpc.sh`, since the compression data will be generated
             immediately before training the surrogate. If you are calling `fit.py` on your own, leave this as False,
             but your config_file should have all the data it needs to train the surrogate.
- `executor` - the parallel executor for training surrogate. Options are `thread` or `process`. Default (`process`).
- `max_workers` - the maximum number of workers to use for parallel processing. Defaults to max available CPUs.
- `runtime_hr` - the runtime in hours for training the surrogate. Defaults to 3 hours. Will run until completion of
                 last iteration past this runtime, which may end up being longer.
- `max_iter` - the maximum number of iterations to run the surrogate training. Defaults to 200.
- `targets` - the target output variables to train the surrogate on. Defaults to all output variables.
- `train_single_fidelity` - whether to train a single-fidelity surrogate in addition to the multi-fidelity surrogate,
                            (mainly to compare cost during training). Defaults to False.

!!! Note
    The compression and test set data should be generated **first** by running `gen_data.py`.

Includes:

- `train_surrogate()` - train a surrogate from a PEM configuration file
"""
import argparse
import copy
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from uqtils import ax_default
from amisc import System


parser = argparse.ArgumentParser(description='Train a surrogate from a PEM configuration file.')
parser.add_argument('config_file', type=str,
                    help='the path to the `amisc` YAML config file with model and input/output variable information.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='the directory to save the generated surrogate data. Defaults to same '
                         'directory as <config_file>.')
parser.add_argument('--search', action='store_true', default=False,
                    help='whether to search for the most recent compression save file in the output directory.')
parser.add_argument('--executor', type=str, default='process',
                    help='the parallel executor for training the surrogate. Options are `thread` or `process`. '
                         'Default (`process`).')
parser.add_argument('--max_workers', type=int, default=None,
                    help='the maximum number of workers to use for parallel processing. Defaults to using max'
                         'number of available CPUs.')
parser.add_argument('--runtime_hr', type=float, default=3.0,
                    help='the maximum runtime in hours for training the surrogate. Defaults to 3 hours.')
parser.add_argument('--max_iter', type=int, default=200,
                    help='the maximum number of iterations to run the surrogate training. Defaults to 200.')
parser.add_argument('--targets', type=str, nargs='+', default=None,
                    help='the target output variables to train the surrogate on. Defaults to all output variables.')
parser.add_argument('--train_single_fidelity', action='store_true', default=False,
                    help='whether to train a single-fidelity surrogate in addition to the multi-fidelity surrogate.')

args, _ = parser.parse_known_args()


def train_surrogate(system: System, executor: Executor, runtime_hr: float, max_iter: int, targets: list[str],
                    train_single_fidelity: bool = False):
    """Train an `amisc.System` surrogate."""
    test_set = pth if (pth := system.root_dir / 'test_set' / 'test_set.pkl').exists() else None

    fit_kwargs = dict(targets=targets, num_refine=500, max_iter=max_iter, runtime_hr=runtime_hr,
                      save_interval=10, max_tol=1e-3, estimate_bounds=True, update_bounds=True,
                      test_set=test_set, plot_interval=1, executor=executor)
    system.fit(**fit_kwargs)
    system.plot_allocation()

    # Compare single-fidelity and multi-fidelity surrogates
    if train_single_fidelity:
        mf_cost_alloc, mf_eval_alloc, mf_cost_cum, mf_eval_cum = system.get_allocation()
        mf_train_history = copy.deepcopy(system.train_history)

        # Reset data for single-fidelity training
        system.clear()
        for comp in system.components:
            comp.model_fidelity = ()   # Will use each model's default value for model_fidelity if applicable

        system.root_dir = system.root_dir / 'amisc_single_fidelity'
        system.fit(**fit_kwargs)

        sf_cost_alloc, sf_eval_alloc, sf_cost_cum, sf_eval_cum = system.get_allocation()
        sf_train_history = copy.deepcopy(system.train_history)

        # Gather test set performance for plotting
        targets = targets or list(mf_train_history[-1]['test_error'].keys())
        num_plot = min(len(targets), 3)
        mf_test = np.full((len(mf_train_history), num_plot), np.nan)
        sf_test = np.full((len(sf_train_history), num_plot), np.nan)
        for j, (mf_res, sf_res) in enumerate(zip(mf_train_history, sf_train_history)):
            for i, var in enumerate(targets[:num_plot]):
                if (perf := mf_res.get('test_error')) is not None:
                    mf_test[j, i] = perf[var]
                if (perf := sf_res.get('test_error')) is not None:
                    sf_test[j, i] = perf[var]

        # Get the cost of a single high-fidelity evaluation (for each model separately, and then sum)
        single_sf_cost = []
        sf_alpha = ()  # No fidelity indices for single-fidelity
        for comp in system.components:
            if comp in sf_cost_alloc:
                single_sf_cost.append(sf_cost_alloc[comp][sf_alpha] / sf_eval_alloc[comp][sf_alpha])
        single_sf_cost = sum(single_sf_cost)

        # Plot QoI L2 error on test set vs. cost
        labels = [system.outputs()[var].get_tex(units=True) for var in targets]
        fig, axs = plt.subplots(1, num_plot, sharey='row', figsize=(3.5 * num_plot, 4), layout='tight', squeeze=False)
        for i in range(num_plot):
            ax = axs[0, i]
            ax.plot(mf_cost_cum / single_sf_cost, mf_test[:, i], '-k', label='Multi-fidelity (MF)')
            ax.plot(sf_cost_cum / single_sf_cost, sf_test[:, i], '--k', label='Single-fidelity (SF)')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.grid()
            ax.set_title(labels[i])
            ylabel = r'Relative error' if i == 0 else ''
            ax_default(ax, r'Cost (number of SF evals)', ylabel, legend=i+1 == num_plot)
        fig.savefig(system.root_dir / 'error_v_cost.png', dpi=300, format='png')


if __name__ == '__main__':
    output_dir = args.output_dir or Path(args.config_file).parent
    system = None

    # Search for an amisc compression save directory generated by `gen_data.py`
    if args.search:
        if not output_dir.startswith('amisc_'):
            # Search for the most recent amisc timestamp
            most_recent = None
            timestamp = 'amisc_2023-01-01T00:00:00'
            for f in os.listdir(Path(output_dir)):
                if (Path(output_dir) / f).is_dir() and f.startswith('amisc_') and f > timestamp:
                    timestamp = f
                    most_recent = f

            if most_recent is not None:
                output_dir = Path(output_dir) / most_recent

        # Now try to load from a compression save file
        if str(Path(output_dir).name).startswith('amisc_'):
            if (Path(output_dir) / 'compression').exists():
                compression_files = list(Path(output_dir).glob('*_compression.yml'))  # See gen_data.py
                if compression_files:
                    compression_file = compression_files[0]
                    system = System.load_from_file(Path(output_dir) / 'compression' / compression_file)

    # If not searching or couldn't find a compression save file, load from the config file directly
    if system is None:
        system = System.load_from_file(args.config_file, output_dir)

    if Path(args.config_file).name not in os.listdir(system.root_dir):
        shutil.copy(args.config_file, system.root_dir)

    match args.executor.lower():
        case 'thread':
            pool_executor = ThreadPoolExecutor
        case 'process':
            pool_executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    with pool_executor(max_workers=args.max_workers) as executor:
        train_surrogate(system, executor, args.runtime_hr, args.max_iter, args.targets,
                        train_single_fidelity=args.train_single_fidelity)
