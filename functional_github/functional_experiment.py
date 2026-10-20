"""Reproduce the aggregate and 156-client weekly electricity experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from functional_models import ALPHA_GRID, K_GRID, RIDGE_GRID, METHODS, dm_test, forecast_all

DATA_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"


def last_sunday(year, month):
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return end - pd.Timedelta(days=(end.dayofweek + 1) % 7)


def prepare_data(path):
    """Notebook's float32 parsing, DST repair, fixed eligibility and weekly curves."""
    with ZipFile(path) as archive, archive.open("LD2011_2014.txt") as stream:
        data = pd.read_csv(stream, sep=";", decimal=",", index_col=0,
                           parse_dates=True, dtype=np.float32)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if data.shape != (140256, 370) or data.isna().any().any():
        raise ValueError("Expected the complete UCI 2011–2014 data (140256 x 370), without missing values.")
    log = []
    years = range(data.index.min().year, data.index.max().year + 1)
    for year in years:
        day = last_sunday(year, 3)
        window = data.loc[day + pd.Timedelta(minutes=30):day + pd.Timedelta(hours=3)]
        if window.empty:
            continue
        zero = window.index[np.isclose(window.sum(axis=1).to_numpy(dtype=float), 0,
                                       atol=1e-12, rtol=0)]
        if len(zero) not in (0, 4):
            raise ValueError(f"Unexpected spring DST pattern in {year}.")
        data.loc[zero] = np.nan
        log.append(dict(year=year, season="spring", diagnostic=np.nan,
                        action="interpolated 4 zero slots" if len(zero) else "none"))
    data = data.interpolate(method="time", limit_area="inside")
    for year in years:
        stamps, ratios = [], []
        for offset in pd.to_timedelta(["01:15:00", "01:30:00", "01:45:00", "02:00:00"]):
            stamp = last_sunday(year, 10) + offset
            if stamp not in data.index:
                continue
            neighbors = [data.loc[stamp + pd.Timedelta(days=d)].to_numpy(dtype=float)
                         for d in (-7, 7) if stamp + pd.Timedelta(days=d) in data.index]
            if not neighbors:
                continue
            denominator = float(np.sum(np.mean(np.stack(neighbors), axis=0)))
            if denominator > 0:
                stamps.append(stamp)
                ratios.append(float(np.sum(data.loc[stamp].to_numpy(dtype=float)) / denominator))
        if ratios:
            r = np.asarray(ratios)
            halve = len(r) == 4 and np.sum((r >= 1.5) & (r <= 2.5)) >= 3
            if halve:
                data.loc[pd.DatetimeIndex(stamps)] /= 2.0
            log.append(dict(year=year, season="autumn", diagnostic=float(r.mean()),
                            action="halved repeated-hour slots" if halve else "none"))
    data.index -= pd.Timedelta(minutes=15)  # interval end -> interval start
    starts = (data.index - pd.to_timedelta(data.index.dayofweek, unit="D")).normalize()
    counts = pd.Series(1, index=starts).groupby(level=0).sum()
    weeks = counts[counts == 672].index
    train_size = int(0.70 * len(weeks))
    train = data.loc[(data.index >= weeks[0]) &
                     (data.index < weeks[train_size - 1] + pd.Timedelta(days=7))]
    active = train.resample("D").sum() > 0
    weekly_active = (train.resample("W-SUN", label="left", closed="left").sum() > 0).mean()
    keep = (active.iloc[:30].any() & active.iloc[-30:].any() & (active.mean() >= 0.90)
            & (weekly_active >= 0.95) & (train.resample("1h").mean().std() > 1e-8))
    clients = keep[keep].index.tolist()
    hourly = data[clients].resample("1h").mean()

    def curves(series):
        frame = pd.DataFrame({"load": series})
        frame["week"] = (frame.index - pd.to_timedelta(frame.index.dayofweek, unit="D")).normalize()
        frame["hour"] = frame.index.dayofweek * 24 + frame.index.hour
        return frame.pivot(index="week", columns="hour", values="load").reindex(
            index=weeks, columns=range(168)).to_numpy(dtype=float)

    aggregate = curves(hourly.mean(axis=1))
    panel = np.stack([curves(hourly[c]) for c in clients])
    if panel.shape != (156, 208, 168) or not np.isfinite(panel).all():
        raise ValueError(f"Unexpected selected panel shape: {panel.shape}.")
    return aggregate, panel, weeks, clients, train_size, pd.DataFrame(log)


def dm_table(losses, normalized=False):
    rows = []
    for key, label in METHODS.items():
        if key == "ges":
            continue
        d = losses["ges"] - losses[key]
        statistic, p = dm_test(d)
        field = "Mean normalized loss differential" if normalized else "Mean loss differential"
        rows.append({"Benchmark": label, field: d.mean(), "DM statistic": statistic, "p-value": p})
    return pd.DataFrame(rows)


def run(data_path, output):
    print("Reading data and constructing weekly curves ...", flush=True)
    y, panel, dates, clients, train, dst = prepare_data(data_path)
    test_dates = dates[train:]
    dst.to_csv(output / "electricity_dst_correction_log.csv", index=False)
    np.savez_compressed(output / "weekly_curves.npz", aggregate=y, panel=panel,
                        dates=dates.to_numpy(), clients=np.asarray(clients), train_size=train)
    print("Aggregate: 208 weeks; 145 training and 63 evaluation weeks.", flush=True)
    fc, losses, alpha, profile, tuning, radii, _ = forecast_all(y, train)
    np.savez_compressed(output / "aggregate_forecasts.npz", observed=y[train:],
                        dates=test_dates.to_numpy(), alpha=alpha, profile=profile,
                        alpha_grid=ALPHA_GRID, **fc)
    aggregate_table = pd.DataFrame([
        {"Method": METHODS[k], "Forecast error": v.mean(), "RMSE": np.sqrt(v.mean()),
         "Relative to random walk": v.mean() / losses["random_walk"].mean(),
         "Improvement vs random walk (%)": 100 * (1 - v.mean() / losses["random_walk"].mean())}
        for k, v in losses.items()]).sort_values("Forecast error").reset_index(drop=True)
    aggregate_table.to_csv(output / "electricity_aggregate_forecast_table.csv", index=False)
    aggregate_weekly = pd.DataFrame({"forecast_week": test_dates, "alpha_hat": alpha,
                                     **{f"{k}_loss": v for k, v in losses.items()}})
    aggregate_weekly.to_csv(output / "electricity_aggregate_weekly_losses.csv", index=False)
    dm_table(losses).to_csv(output / "electricity_aggregate_dm_tests.csv", index=False)
    aggregate_tuning = {**tuning, "spectral_radius_max": float(radii.max())}
    summary_rows, weekly_frames = [], []
    for i, (client, yi) in enumerate(zip(clients, panel)):
        _, loss, alpha, _, tune, radii, scale = forecast_all(yi, train)
        summary_rows.append({"client": client, "scale_mse": scale,
            **{f"{k}_mse": v.mean() for k, v in loss.items()},
            **{f"{k}_nmse": v.mean() / scale for k, v in loss.items()},
            "mean_alpha": alpha.mean(), "median_alpha": np.median(alpha),
            "min_alpha": alpha.min(), "max_alpha": alpha.max(), "final_alpha": alpha[-1],
            **{f"far_{k}": v for k, v in tune.items()},
            "far_spectral_radius_median": np.median(radii),
            "far_spectral_radius_max": radii.max(),
            "far_unstable_origin_fraction": np.mean(radii >= 1.0)})
        weekly_frames.append(pd.DataFrame({"client": client, "forecast_week": test_dates,
            "scale_mse": scale, "alpha_hat": alpha, **{f"{k}_loss": v for k, v in loss.items()}}))
        if (i + 1) % 10 == 0 or i + 1 == len(clients):
            print(f"Clients completed: {i + 1}/{len(clients)}", flush=True)
    summary = pd.DataFrame(summary_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True)
    summary.to_csv(output / "electricity_client_summary.csv", index=False)
    weekly.to_csv(output / "electricity_panel_weekly_losses.csv", index=False)
    winners = summary[[f"{k}_nmse" for k in METHODS]].idxmin(axis=1).value_counts()
    table = pd.DataFrame([{"Method": label, "Median NMSE": summary[f"{k}_nmse"].median(),
        "Mean NMSE": summary[f"{k}_nmse"].mean(), "Client wins": int(winners.get(f"{k}_nmse", 0)),
        "Client win rate (%)": 100 * winners.get(f"{k}_nmse", 0) / len(clients)}
        for k, label in METHODS.items()]).sort_values("Median NMSE").reset_index(drop=True)
    table.to_csv(output / "electricity_panel_main_table.csv", index=False)
    pooled = {}
    for key in METHODS:
        # Average normalized differentials by forecast week, as in the notebook.
        weekly["d_norm"] = (weekly["ges_loss"] - weekly[f"{key}_loss"]) / weekly["scale_mse"]
        pooled[key] = -weekly.groupby("forecast_week")["d_norm"].mean().sort_index().to_numpy()
    dm_table(pooled, normalized=True).to_csv(output / "electricity_panel_pooled_normalized_dm.csv", index=False)
    counts, per_client = [], []
    for client, group in weekly.groupby("client"):
        group = group.sort_values("forecast_week")
        for key, label in METHODS.items():
            if key != "ges":
                statistic, p = dm_test(group["ges_loss"].to_numpy() - group[f"{key}_loss"].to_numpy())
                per_client.append({"client": client, "Benchmark": label, "DM statistic": statistic, "p-value": p})
    per_client = pd.DataFrame(per_client)
    per_client.to_csv(output / "electricity_per_client_dm.csv", index=False)
    for label, group in per_client.groupby("Benchmark", sort=False):
        significant = group["p-value"] < 0.05
        counts.append({"Benchmark": label, "GES significantly better": int((significant & (group["DM statistic"] < 0)).sum()),
                       "GES significantly worse": int((significant & (group["DM statistic"] > 0)).sum()), "Total clients": len(clients)})
    pd.DataFrame(counts).to_csv(output / "electricity_per_client_dm_counts.csv", index=False)
    with open(data_path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"python": platform.python_version(), "data_sha256": digest, "data_url": DATA_URL,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "threadpoolctl")},
        "clients": len(clients), "selected_clients": clients, "weeks": len(dates),
        "training_weeks": train, "evaluation_weeks": len(test_dates), "HAC_lags": 5,
        "alpha_grid": {"min": 0.001, "max": 0.999, "count": 999},
        "FAR_K_grid": K_GRID, "FAR_ridge_grid": RIDGE_GRID, "FAR_validation_weeks": 20,
        "BLAS_threads": 1, "aggregate_FAR": aggregate_tuning}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(aggregate_table.to_string(index=False))
    print(table.to_string(index=False))


def verify(output):
    """Compare compact final notebook tables (not rounded paper values)."""
    reference = Path(__file__).resolve().parent / "reference"
    reports = []
    for path in sorted(reference.glob("*.csv")):
        expected, actual = pd.read_csv(path), pd.read_csv(output / path.name)
        keys = [c for c in expected if not pd.api.types.is_numeric_dtype(expected[c])]
        if keys:
            expected = expected.sort_values(keys).reset_index(drop=True)
            actual = actual.sort_values(keys).reset_index(drop=True)
        if len(expected) != len(actual):
            raise AssertionError(f"Row count differs: {path.name}")
        max_diff = 0.0
        for col in expected:
            if pd.api.types.is_numeric_dtype(expected[col]):
                np.testing.assert_allclose(actual[col], expected[col], rtol=2e-6, atol=1e-8, equal_nan=True,
                                           err_msg=f"{path.name}: {col}")
                max_diff = max(max_diff, float(np.nanmax(np.abs(actual[col] - expected[col]))))
            elif not actual[col].equals(expected[col]):
                raise AssertionError(f"Labels differ: {path.name}: {col}")
        reports.append({"file": path.name, "rows": len(actual), "max_absolute_difference": max_diff})
    if not reports:
        raise FileNotFoundError("No reference tables found.")
    report = {"status": "passed", "rtol": 2e-6, "atol": 1e-8, "tables": reports}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("Reference-table verification passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="UCI electricityloaddiagrams20112014.zip")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder must be empty; choose a new --output path.")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.data, args.output)
    if args.verify:
        verify(args.output)


if __name__ == "__main__":
    main()
