# Wasserstein exponential smoothing: turbulence experiment

Numerical implementation for Section 5.3: raw vertical-velocity distributions at
five measurement heights. The main experiment uses 60-second blocks; 30-second
blocks provide the supplementary robustness check. Models are fitted separately
at each height. GES in the output tables denotes Wasserstein exponential smoothing.

## Run

Tested with **Python 3.13.15**. From this folder:

```sh
python -m venv .venv
# Activate: Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Activate: macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `long_time_series_2011_july_07.txt` from the
[original dataset](https://doi.org/10.5281/zenodo.20045165) into `data/`.
The script checks the original file's MD5 (`b6cfab0e8af9b4b2052d754ccfb29ee5`).
The file name says July 07; its timestamps are **10 July 2011, 09:30–17:00**.

```sh
python turbulence_experiment.py --data data/long_time_series_2011_july_07.txt --blocks 60 30 --verify
python plot_turbulence.py --results results/60s
```

Use a quoted full path with `--data` if the file is elsewhere. To repeat a run,
choose a fresh directory, for example `--output results_repeat`; existing results
are not overwritten. `--blocks 60` or `--blocks 30` runs only that configuration.
Plotting also accepts `--results results/30s`.

## Files and outputs

- `wes_models.py`: GES, random walk, running Fréchet mean and corrected ATM_d(1).
- `turbulence_experiment.py`: validated data loading, quantiles, evaluation, DM/MCS and verification.
- `plot_turbulence.py`: the main distribution figure, all-height trajectories and parameter paths, as PDF/PNG.
- `reference_results.csv`: the 40 previously saved corrected method/height/configuration results.

Each `results/60s` or `results/30s` folder contains the quantile panel, per-height
forecasts and losses, `all_height_results.csv`, and settings/package metadata.
`--verify` checks mean/root losses, DM statistics and p-values, MCS p-values and
membership against the reference; a mismatch exits with an error. The tolerances
are relative `1e-8` and absolute `1e-10`; MCS membership must match exactly.
`results/verification.json` records the check. The bundled `reproduction_check.json`
records the release-time verification, including comparisons of individual forecasts.

## Fixed paper settings

| Setting | Main | Robustness |
|---|---:|---:|
| Block duration | 60 seconds | 30 seconds |
| Raw samples per distribution | 1,200 | 600 |
| Initial training / evaluation blocks | 315 / 135 | 630 / 270 |
| DM maximum lag | 5 | 10 |
| MCS expected bootstrap block length | 10 | 20 |

Both use all five heights (0.7, 1.4, 2.1, 2.8, 3.5 m), 1,000 midpoint empirical
inverse-CDF quantiles, expanding-origin one-step forecasts and an alpha grid
`0.001, …, 0.999`. Measurements are not detrended, centered, rotated or rescaled.
One trailing endpoint sample is excluded; no incomplete blocks are imputed.

DM uses Bartlett HAC with available-pair autocovariances and two-sided t calibration.
The 90% MCS uses the range statistic, 5,000 stationary-bootstrap draws and seed 0.
These settings preserve the same physical lag and bootstrap spans across block sizes.
Heights are dependent; no inference pools them as independent replications.

ATM uses common endpoints, monotone linear transport interpolation, direct
reverse-pair inverses, and 2,201-point spatial trapezoidal integration. Support
uses only past observations with 5% padding; quantile regularization is `1e-4`,
and coefficients are constrained to `[-0.999, 0.999]`.

## Sources

- Chamecki and Isard (2026), [turbulence dataset](https://doi.org/10.5281/zenodo.20045165), CC BY 4.0; raw data are not bundled.
- Zhu and Müller (2023), [Autoregressive optimal transport models](https://doi.org/10.1093/jrsssb/qkad051).

