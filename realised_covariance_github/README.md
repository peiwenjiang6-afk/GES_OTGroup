# Realised covariance forecasting

- `covariance_models.py`: SPD geometry and seven forecasting methods.
- `covariance_experiment.py`: price cleaning, rolling forecasts, scores and inference.
- `plot_covariance.py`: observed/GES dynamics, all components, cumulative losses and alpha.

## Setup and input

Validated with Python **3.14.6** and the pinned versions in `requirements.txt`:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Supply your licensed Refinitiv DataScope extract as `data/real_cov_5d.csv.gz`.
It must contain `#RIC`, `Date-Time`, and `Last`, exactly five assets, and
exchange-local ISO timestamps with UTC offsets. The licensed raw data are not
redistributed. The release validation report records the original input SHA-256
and asset order; another extract need not reproduce the paper's numbers.

```bash
python covariance_experiment.py --data data/real_cov_5d.csv.gz --output results
python plot_covariance.py --results results
```

The complete rolling refit is computationally expensive. For a shorter check,
use `--limit 31 --output results_smoke`; this covers two refits and skips DM/MCS.
Existing nonempty output directories are rejected. The package uses one BLAS
thread for consistent small-matrix computation.

## Fixed experiment

Prices from 2000 onward are aligned to the regular NYSE 09:30–16:00 five-minute
grid, excluding early closes. The isolated-bad-print thresholds are 0.10 for
each jump and 0.02 for the round trip. Each asset-day may have at most two
missing prices and no consecutive missing prices; the opening quote is
required. Missing prices are forward-filled only within the session.
Covariances sum outer products of percentage log returns, so their entries
have squared-percentage units.

All seven methods use a **1000-day rolling window** and refits every **30 days**.
GES, EWMA and Geodesic AR tune over 0.01, 0.02, ..., 0.99 using AIRM².
GES/EWMA states update daily. The Fréchet mean is fixed between refits.
Scalar vech-HAR has element-specific intercepts and three shared slopes, with
overlapping lag averages 1, 1–5, 1–20. HAR-DRD uses individual log-variance
regressions and constrained correlation slopes with lags 1, 2–5, 6–20.
SPD eigenvalues are floored at 1e-10 where specified in the model functions.

Scores are AIRM², the **unsquared** Frobenius norm and Gaussian NLL. NLL uses
the daily percentage log return, demeaned by the past 1000-day mean. Daily
returns use the raw exact opening quote and last valid quote by 16:00.
GMV portfolios allow short positions and use simple decimal returns; tail
scores use demeaned decimal log returns. Annualisation uses 252 days; maximum
drawdown is non-positive. QLoss is summed and joint VaR/ES loss is averaged.

DM comparisons use GES minus benchmark loss, Bartlett HAC bandwidth 5 and
two-sided normal p-values. MCS uses the notebook's range statistic, moving
blocks of 22 days, 5000 replications and seed 42. Forecast score summaries,
portfolio/tail tables, DM/MCS tables, parameters, all forecast matrices and
run metadata are saved. Figures are exported as PDF and 300 dpi PNG.
The optional economic-MCS and volatility-regime diagnostics are omitted.

## Validation and scope

Small `reference_*_summary.csv` files preserve the original notebook's saved
tables. `validation_report.json` records exactly what was checked. It separates
the 31-target fresh forecast check from full-sample tables regenerated using
the previously saved daily results; it does **not** claim a complete fresh
forecast rerun. Numerical differences across BLAS/library versions are possible.

Raw data, generated results and notebooks are excluded from the package.
No code license has been assigned; the repository owner should select one
before public redistribution. Data access remains subject to the data licence.
