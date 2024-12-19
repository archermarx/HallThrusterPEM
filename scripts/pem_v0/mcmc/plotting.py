import ash
import numpy as np
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_traces(
    samples: np.ndarray, names: list[str], file: Path | str
) -> tuple[Figure, Axes]:
    """Plot MCMC sample history across all iterations and variables"""
    width = 6
    height = 2
    num_vars = len(names)
    fig, axes = plt.subplots(num_vars, 1, figsize=(width, num_vars * height))
    sample_indices = np.arange(samples.shape[0])

    for i, ax in enumerate(axes):
        ax.set_ylabel(f"${names[i]}$")
        ax.set_xlabel("Step")
        ax.plot(sample_indices, samples[:, i], color="black")
        ax.set_xlim(0, sample_indices[-1])

    plt.tight_layout()
    fig.savefig(file, dpi=300)
    return fig, axes


def _num_bins(samples: np.ndarray, lims: tuple[float, float]) -> int:
    """Calculate optimal histogram bin count using Friedman-Draconis rule"""
    q3 = np.percentile(samples, 75)
    q1 = np.percentile(samples, 25)
    iqr = q3 - q1

    bin_width = 2 * iqr / np.cbrt(samples.size)
    num_bins = np.ceil((lims[1] - lims[0]) / bin_width).astype(int)
    return num_bins


def _ax_hist1d(ax: Axes, samples: np.ndarray, xlims: tuple[float, float]) -> None:
    ax.set_xlim(xlims)
    nbins = _num_bins(samples, xlims)
    nshifts = 5
    bins, heights = ash.ash1d(samples, nbins, nshifts, range=xlims)
    ax.hist(samples, nbins, color="lightgrey", density=True)
    ax.plot(bins, heights, zorder=2, color="black", linewidth=2)
    ax.axvline(float(np.mean(samples)), color="black", linestyle="--", linewidth=2)


def _ax_hist2d(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    logpdfs: Optional[np.ndarray] = None,
) -> None:
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    nbins_x = _num_bins(x, xlims)
    nbins_y = _num_bins(y, ylims)
    nbins = min(nbins_x, nbins_y)
    nshifts = 5
    grid, heights = ash.ash2d(x, y, nbins, nshifts, xrange=xlims, yrange=ylims)

    scatter_kwargs = {
        "s": 6**2,
        "alpha": 1 / np.log10(x.size),
        "zorder": 1,
    }

    if logpdfs is None:
        ax.scatter(x, y, color="black", **scatter_kwargs)
    else:
        ax.scatter(x, y, c=logpdfs, **scatter_kwargs)

    ax.contour(grid[0], grid[1], heights, zorder=0)


def _determine_limits(x: np.ndarray) -> tuple[float, float]:
    min = np.min(x)
    max = np.max(x)
    diff = max - min
    pad = 0.1 * diff
    return min - pad, max + pad


def plot_corner(
    samples: np.ndarray,
    names: list[str],
    file: Path | str,
    logpdfs: Optional[np.ndarray] = None,
) -> tuple[Figure, Axes]:

    names_latex = [f"${name}$" for name in names]
    size = 8
    fontsize = 15
    num_vars = len(names)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(size, size))

    lims = [_determine_limits(samples[:, i]) for i in range(num_vars)]

    for j, row in enumerate(axes):
        for i, ax in enumerate(row):
            if i > j:
                ax.set_axis_off()
                continue
            elif i == j:
                _ax_hist1d(ax, samples[:, i], lims[i])
                ax.set_yticks([])
                if j < num_vars - 1:
                    ax.set_xticklabels([])

            elif i < j:
                _ax_hist2d(ax, samples[:, i], samples[:, j], lims[i], lims[j], logpdfs)

                if i == 0:
                    ax.set_ylabel(names_latex[j], fontsize=fontsize)
                else:
                    ax.set_yticklabels([])

                if j == num_vars - 1:
                    ax.set_xlabel(names_latex[i], fontsize=fontsize)
                else:
                    ax.set_xticklabels([])

            ax.tick_params(axis="x", rotation=-45)

    plt.tight_layout()
    fig.savefig(file, dpi=300)
    return fig, axes


if __name__ == "__main__":
    sample_dir = Path("scripts/pem_v0/amisc_2024-12-18T21.21.01")

    data = np.genfromtxt(sample_dir / "samples.txt", delimiter=",")
    samples = data[:, :-2]
    unique_samples, unique_inds = np.unique(samples, axis=0, return_index=True)
    unique_logpdfs = data[unique_inds, 2]

    names = ["a_1", "a_2"]

    plot_traces(samples, names, sample_dir / "traces.png")
    plot_corner(unique_samples, names, sample_dir / "corner.png", unique_logpdfs)
    cov = np.cov(samples.T)
    print(cov)
