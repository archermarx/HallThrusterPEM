import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import numpy as np
from typing import Optional, Callable
import matplotlib as mpl
import matplotlib.pyplot as plt

import MCMCIterators.samplers as samplers

import hallmd
from hallmd.data.loader import spt100_data
from amisc import YamlLoader, System

parser = argparse.ArgumentParser(
    description="Generate compression (SVD) data and test set data for training a surrogate."
)
parser.add_argument(
    "config_file",
    type=str,
    help="the path to the `amisc` YAML config file with model and input/output variable information.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="the directory to save the generated SVD data and test set data. Defaults to same "
    "directory as <config_file>.",
)
parser.add_argument(
    "--executor",
    type=str,
    default="process",
    help="the parallel executor for training the surrogate. Options are `thread` or `process`. "
    "Default (`process`).",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=None,
    help="the maximum number of workers to use for parallel processing. Defaults to using max"
    "number of available CPUs.",
)


def get_nominal_inputs(system: System) -> dict[str, float]:
    inputs: dict[str, float] = {}
    for v in system.inputs():
        inputs[v.name] = v.get_nominal()

    return inputs


def log_prior(
    system: System, params_to_calibrate: list[str], params: np.ndarray
) -> float:
    logp = 0.0
    for key, value in zip(params_to_calibrate, params):
        logp += np.log(system.inputs()[key].distribution.pdf(value))[0]

    return logp


type DataSet = dict[str, list[hallmd.ExpData]]
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_spt100_ui(
    spt100_data: DataSet, output: Optional[dict], casename: str
) -> tuple[Figure, Axes]:

    data = spt100_data["uion"][0]
    colors = ["red", "green", "blue"]
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("$z / L_{ch}$")
    ax.set_ylabel("$u_{ion, z}$ [km/s]")
    z_data = data["loc"]

    for i, (cond, ui, var_ui, color) in enumerate(
        zip(data["x"], data["y"], data["var_y"], colors)
    ):
        pressure_base = round(1e5 * 10 ** cond[0], 1)

        label = f"$P_B = {pressure_base}\\times 10^{{-5}}$ Torr"
        ax.errorbar(
            z_data / 0.025,
            ui / 1000,
            yerr=np.sqrt(var_ui) / 1000,
            label=label,
            color=color,
            fmt="--o",
            markersize=4.0,
        )

        if output is not None:
            z_sim, u_sim = output["u_ion_coords"][i], output["u_ion"][i]
            plt.plot(z_sim / 0.025, u_sim / 1000, color=color)

    plt.legend()
    plt.tight_layout()
    fig.savefig(casename + ".png", dpi=200)
    return fig, ax


def _calc_log_likelihood(data, output):
    u_data = data["uion"][0]["y"]
    z_data = data["uion"][0]["loc"]
    var_u = data["uion"][0]["var_y"]

    # Assume Gaussian likelihood at each datapoint
    #
    # f(u) = 1/sqrt(2 * pi )/s * exp(-(u - u_data)^2 / (2 * s^2))
    #
    # log(f(u)) = -0.5 log(2 pi s^2) - 0.5 * ((u-u_data)/s)^2

    logp = 0.0
    for i, ui in enumerate(u_data):
        z_sim, u_sim = output["u_ion_coords"][i], output["u_ion"][i]
        u_itp = np.interp(z_data, z_sim, u_sim)
        logp += np.mean(
            -0.5 * np.log(2 * np.pi * var_u) - 0.5 * (u_itp - ui) ** 2 / var_u
        )

    return logp


def _run_model(
    system: System,
    executor: Callable,
    base_params: dict[str, float | np.ndarray | np.float64],
    params_to_calibrate: list[str],
    params: np.ndarray,
) -> dict:

    # Construct input dictionary
    sample_dict = base_params.copy()
    for key, val in zip(params_to_calibrate, params):
        # sample_dict[key] = system.inputs()[key].normalize(val)
        # Work directly with normalized values
        sample_dict[key] = val

    # Run model
    with executor(max_workers=args.max_workers) as executor:
        outputs = system.predict(
            sample_dict,
            use_model=(0, 0),
            model_dir=mcmc_dir,
            executor=executor,
            verbose=False,
        )

    return outputs


def log_likelihood(
    system: System,
    data: dict[str, list[hallmd.ExpData]],
    executor: Callable,
    base_params: dict[str, float | np.ndarray | np.float64],
    params_to_calibrate: list[str],
    params: np.ndarray,
) -> float:

    outputs = _run_model(system, executor, base_params, params_to_calibrate, params)

    return _calc_log_likelihood(data, outputs)


def log_posterior(
    system: System,
    data: dict[str, list[hallmd.ExpData]],
    executor: Callable,
    base_params: dict[str, float | np.ndarray | np.float64],
    params_to_calibrate: list[str],
    params: np.ndarray,
) -> float:

    prior = log_prior(system, params_to_calibrate, params)
    if not np.isfinite(prior):
        return -np.inf

    likelihood = log_likelihood(
        system, data, executor, base_params, params_to_calibrate, params
    )
    if not np.isfinite(likelihood):
        return -np.inf

    return prior + likelihood


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    system = YamlLoader.load(args.config_file)
    system.root_dir = args.output_dir or Path(args.config_file).parent
    system.set_logger(stdout=True)

    # load data
    data = spt100_data(["uion"])
    params_to_calibrate = ["a_1", "a_2"]
    operating_params = ["P_b", "V_a", "mdot_a"]
    base = {
        key: system.inputs()[key].normalize(val)
        for key, val in get_nominal_inputs(system).items()
        if key not in operating_params
    }

    base["a_1"], base["a_2"] = -2.04105612, 6.17871887

    for i, var in enumerate(operating_params):
        value = data["uion"][0]["x"][:, i]
        base[var] = value

    if Path(args.config_file).name not in os.listdir(system.root_dir):
        shutil.copy(args.config_file, system.root_dir)

    mcmc_dir = Path(system.root_dir) / "mcmc-test"
    os.mkdir(mcmc_dir)

    match args.executor.lower():
        case "thread":
            executor = ThreadPoolExecutor
        case "process":
            executor = ProcessPoolExecutor
        case _:
            raise ValueError(f"Unsupported executor type: {args.executor}")

    logpdf = lambda params: log_posterior(
        system, data, executor, base, params_to_calibrate, params
    )

    init_sample = np.array([base["a_1"], base["a_2"]])
    init_cov = np.array([[1, -0.5], [-0.5, 1]])

    sampler = samplers.DelayedRejectionAdaptiveMetropolis(
        logpdf,
        init_sample,
        init_cov,
        adapt_start=10,
        eps=1e-6,
        sd=None,
        interval=1,
        level_scale=1e-1,
    )

    samples: list[np.ndarray] = []
    logps: list[float] = []
    max_logp: float = -np.inf
    best_sample: np.ndarray = init_sample
    accepted: int = 0
    max_samples: int = 1_000

    outputs = _run_model(system, executor, base, params_to_calibrate, init_sample)
    plot_spt100_ui(data, outputs, "init")

    for i, (sample, logp, accepted_bool) in enumerate(sampler):
        if i >= max_samples:
            break

        print(f"Sample {i+1}: {sample}")
        print(f"\t Logpdf: {logp}")
        print(f"\t Accepted? -> {accepted_bool}")

        if accepted_bool:
            if logp > max_logp:
                best_sample = sample
                max_logp = logp
            samples.append(sample)
            logps.append(logp)
            accepted += accepted_bool

        print(f"\t Best sample: {best_sample}")
        print(f"\t Acceptance ratio: {accepted / (i+1)} ({accepted}/{i+1})")

        if i % 50 == 0:
            outputs = _run_model(
                system, executor, base, params_to_calibrate, best_sample
            )
            plot_spt100_ui(data, outputs, f"map_{i}")

    # predictions from best sample
    outputs = _run_model(system, executor, base, params_to_calibrate, best_sample)
    plot_spt100_ui(data, outputs, "best")