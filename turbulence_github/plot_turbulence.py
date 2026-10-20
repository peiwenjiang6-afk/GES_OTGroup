"""Publication figures from saved forecasts; no fitting or error-based selection.

Example: python plot_turbulence.py --results results/60s
"""
import argparse
from pathlib import Path

import numpy as np

HEIGHTS = {3: 0.7, 2: 1.4, 4: 2.1, 5: 2.8, 1: 3.5}
COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
         "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}


def load_results(folder):
    """Require all five heights with identical, regular timestamps and grids."""
    panels = {}
    for sensor in HEIGHTS:
        with np.load(folder / f"sensor_{sensor}" / "forecasts.npz") as archive:
            p = {key: archive[key] for key in archive.files}
        observed, ges, u = p["observed"], p["ges"], p["u_grid"]
        if observed.ndim != 2 or ges.ndim != 2 or u.ndim != 1:
            raise ValueError(f"Sensor {sensor}: expected matrix quantiles and vector grid.")
        n, width = observed.shape
        n_eval, n_train = len(ges), n - len(ges)
        if n_train < 1 or n_eval < 1 or width != len(u):
            raise ValueError(f"Sensor {sensor}: inconsistent training/grid dimensions.")
        for key in ("ges", "atmd", "random_walk", "frechet_mean"):
            if p[key].shape != (n_eval, width):
                raise ValueError(f"Sensor {sensor}: incorrect shape for {key}.")
        for key in ("alpha", "beta"):
            if p[key].shape != (n_eval,):
                raise ValueError(f"Sensor {sensor}: incorrect shape for {key}.")
        for key in ("observed", "ges", "atmd", "random_walk", "frechet_mean", "alpha", "beta", "u_grid"):
            if not np.isfinite(p[key]).all():
                raise ValueError(f"Sensor {sensor}: nonfinite values in {key}.")
        starts, forecasts = p["block_start"], p["forecast_block_start"]
        if starts.shape != (n,) or forecasts.shape != (n_eval,):
            raise ValueError(f"Sensor {sensor}: incorrect timestamp dimensions.")
        if not np.issubdtype(starts.dtype, np.datetime64) or np.isnat(starts).any():
            raise ValueError(f"Sensor {sensor}: invalid block timestamps.")
        steps = np.diff(starts) / np.timedelta64(1, "s")
        if steps[0] <= 0 or not np.all(steps == steps[0]):
            raise ValueError(f"Sensor {sensor}: blocks are not regularly spaced.")
        if not np.array_equal(starts[n_train:], forecasts):
            raise ValueError(f"Sensor {sensor}: forecasts do not align with observations.")
        if not np.all(np.diff(u) > 0) or not (0 < u[0] < u[-1] < 1):
            raise ValueError(f"Sensor {sensor}: invalid quantile grid.")
        if panels:
            reference = next(iter(panels.values()))
            for key in ("block_start", "forecast_block_start", "u_grid"):
                if not np.array_equal(p[key], reference[key]):
                    raise ValueError(f"Sensor {sensor}: {key} differs between heights.")
        panels[sensor] = p
    return panels, float(steps[0])


def trajectory(ax, p):
    """Observed and GES quantiles on the elapsed-minute time scale."""
    t = (p["block_start"] - p["block_start"][0]) / np.timedelta64(1, "m")
    n_train = len(p["observed"]) - len(p["ges"])
    boundary = (t[n_train - 1] + t[n_train]) / 2
    ax.axvspan(boundary, t[-1] + (t[1] - t[0]) / 2, color="#f1f1f1", zorder=-5)
    for probability, color in zip((0.1, 0.5, 0.9), COLORS):
        index = np.argmin(abs(p["u_grid"] - probability))
        ax.plot(t, p["observed"][:, index], color=color, lw=0.65, alpha=0.78)
        ax.plot(t[n_train:], p["ges"][:, index], color=color, lw=1.25, ls="--")
    ax.axvline(boundary, color=".4", lw=0.75, ls=":")
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="y", color=".9", lw=0.5)
    ax.set_ylabel("Vertical velocity (m/s)")


def make_figures(panels, seconds, output):
    """Middle-height main figure, all-height trajectories, and alpha traces."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for extension in ("pdf", "png"):
            fig.savefig(output / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)

    p = panels[4]  # Middle measurement height, selected independently of accuracy.
    n_train = len(p["observed"]) - len(p["ges"])
    start_clock = np.datetime_as_string(p["block_start"][0], unit="m").split("T")[1]
    xlabel = f"Minutes since {start_clock} (block start)"
    handles = [Line2D([0], [0], color=c, label=f"$Q_{{{q:.1f}}}$")
               for c, q in zip(COLORS, (0.1, 0.5, 0.9))]
    handles += [Line2D([0], [0], color=".25", lw=0.8, label="Observed"),
                Line2D([0], [0], color=".25", ls="--", lw=1.2, label="GES forecast")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 4.8))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.08, 1], hspace=0.55, wspace=0.38)
        ax = fig.add_subplot(grid[0, :])
        trajectory(ax, p)
        ax.set_title("(a) Quantile trajectories at 2.1 m", loc="left")
        ax.set_xlabel(xlabel)
        ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1),
                  frameon=False, fontsize=7)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.35)
        indices = np.array([0, len(p["ges"]) // 2, len(p["ges"]) - 1])
        shown = np.concatenate([p["observed"][n_train + indices], p["ges"][indices]])
        for j, index in enumerate(indices):
            ax = fig.add_subplot(grid[1, j])
            ax.plot(p["u_grid"], p["observed"][n_train + index], color=".22", lw=1, label="Observed")
            ax.plot(p["u_grid"], p["ges"][index], color=COLORS[0], lw=1.25, ls="--", label="GES forecast")
            unit = "m" if seconds % 60 == 0 else "s"
            timestamp = np.datetime_as_string(p["forecast_block_start"][index], unit=unit).split("T")[1]
            ax.set_title(f"({chr(98 + j)}) {timestamp}", loc="left")
            ax.set(xlabel="Probability $u$", xlim=(0, 1), xticks=[0, 0.5, 1],
                   ylim=(shown.min() - 0.1, shown.max() + 0.1))
            if j == 0:
                ax.set_ylabel("Quantile $Q(u)$ (m/s)")
            if j == 1:
                ax.legend(loc="lower right", frameon=False, fontsize=7)
            ax.grid(color=".9", lw=0.5)
        fig.subplots_adjust(left=0.095, right=0.99, bottom=0.105, top=0.95)
        save(fig, f"fig3_turbulence_{seconds:g}s")

        fig, axes = plt.subplots(5, 1, figsize=(7.2, 8), sharex=True)
        for ax, (sensor, height) in zip(axes, HEIGHTS.items()):
            trajectory(ax, panels[sensor])
            ax.set_title(f"{height:.1f} m", loc="left")
            ax.set_ylabel("Velocity (m/s)")
        axes[-1].set_xlabel(xlabel)
        fig.legend(handles=handles, ncol=5, loc="upper center", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8)
        save(fig, "fig_s_turbulence_all_heights")

        fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey=True)
        upper = max(0.12, 1.1 * max(p["alpha"].max() for p in panels.values()))
        for ax, (sensor, height) in zip(axes.flat, HEIGHTS.items()):
            p = panels[sensor]
            t = (p["forecast_block_start"] - p["block_start"][0]) / np.timedelta64(1, "m")
            ax.plot(t, p["alpha"], lw=1.2, color=COLORS[0])
            ax.set_title(f'{height:.1f} m; mean = {p["alpha"].mean():.3f}', loc="left", fontsize=8)
            ax.set_ylim(0, upper)
            ax.set_xticks(np.linspace(t[0], t[-1], 3))
            ax.grid(color=".9", lw=0.5)
        axes.flat[-1].axis("off")
        fig.supxlabel(f"Minutes since {start_clock} (forecast block start)", fontsize=8.5)
        fig.supylabel(r"Selected smoothing parameter $\hat\alpha$", fontsize=8.5)
        fig.tight_layout(rect=(0.02, 0.025, 1, 1), w_pad=1, h_pad=1.1)
        save(fig, "fig_s_turbulence_alpha")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/60s"))
    parser.add_argument("--output", type=Path, help="Default: <results>/figures")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    panels, seconds = load_results(args.results)
    output = args.output or args.results / "figures"
    make_figures(panels, seconds, output)
    print(f"Saved three figures as vector PDFs and 300-dpi PNGs in {output.resolve()}")


if __name__ == "__main__":
    main()
