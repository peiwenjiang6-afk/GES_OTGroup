"""Generate aggregate heatmaps, alpha/loss plots and client comparisons."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from functional_models import METHODS


def save(fig, output, name):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(results, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(results / "aggregate_forecasts.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    dates = pd.to_datetime(data["dates"])
    observed, forecast = data["observed"], data["ges"]
    error = observed - forecast
    limits = (min(observed.min(), forecast.min()), max(observed.max(), forecast.max()))
    error_limit = max(float(np.abs(error).max()), np.finfo(float).eps)
    ticks = np.linspace(0, len(dates) - 1, 6, dtype=int)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for name, values in (("observed", observed), ("forecast", forecast), ("error", error)):
        fig, ax = plt.subplots(figsize=(6, 4.8))
        options = ({"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)}
                   if name == "error" else {"vmin": limits[0], "vmax": limits[1], "cmap": "viridis"})
        im = ax.imshow(values, aspect="auto", origin="lower", extent=(0, 168, 0, len(dates)), **options)
        ax.set(xlabel="Hour of week", ylabel="Forecast week", xticks=np.arange(0, 169, 24),
               yticks=ticks + 0.5, yticklabels=[dates[i].strftime("%Y-%m-%d") for i in ticks])
        fig.colorbar(im, ax=ax, label="Observed − forecast (kW)" if name == "error" else "Cross-client mean load (kW)")
        save(fig, output, f"figure4_heatmap_{name}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, data["alpha"])
    ax.set(xlabel="Forecast week", ylabel=r"Estimated $\widehat{\alpha}_t$")
    save(fig, output, "electricity_aggregate_alpha_path")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["alpha_grid"], data["profile"])
    ax.set(xlabel=r"$\alpha$", ylabel="Expanding mean one-step squared error (kW²)")
    save(fig, output, "electricity_aggregate_loss_profile")
    weekly = pd.read_csv(results / "electricity_aggregate_weekly_losses.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, label in METHODS.items():
        ax.plot(dates, np.cumsum(weekly[f"{key}_loss"]), label=label)
    ax.set(xlabel="Forecast week", ylabel="Cumulative squared functional loss (kW²)")
    ax.legend(fontsize=8)
    save(fig, output, "electricity_aggregate_cumulative_losses")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    example = len(dates) // 2
    ax.plot(np.arange(168), observed[example], color="black", label="Observed", linewidth=1.7)
    for key, label in METHODS.items():
        ax.plot(np.arange(168), data[key][example], label=label, linewidth=1)
    ax.set(xlabel="Hour of week", ylabel="Cross-client mean load (kW)",
           title=f"One-week-ahead forecast: {dates[example]:%Y-%m-%d}", xticks=np.arange(0, 169, 24))
    ax.legend(fontsize=8, ncol=2)
    save(fig, output, "electricity_aggregate_example_forecast")
    summary = pd.read_csv(results / "electricity_client_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(summary["median_alpha"], bins=20, edgecolor="white")
    ax.set(xlabel=r"Client median $\widehat{\alpha}_t$", ylabel="Number of clients")
    save(fig, output, "electricity_panel_alpha_histogram")
    for key in ("far", "seasonal_naive"):
        fig, ax = plt.subplots(figsize=(5, 5))
        x, y = summary[f"{key}_nmse"], summary["ges_nmse"]
        ax.scatter(x, y, alpha=0.65, s=18)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=f"{METHODS[key]} NMSE", ylabel="GES NMSE")
        save(fig, output, f"electricity_panel_ges_vs_{key}")
    print(f"Saved 10 figures as PDF and PNG to {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output or args.results / "figures")


if __name__ == "__main__":
    main()
