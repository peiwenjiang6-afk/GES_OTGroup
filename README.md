# GES numerical experiments

Code for the empirical experiments in: **Exponential Smoothing for Time Series of Random Objects**

**Authors:** Takuo Matsubara, Peiwen Jiang, Wilson Ye Chen, Minh-Ngoc Tran

**Affiliation:** The University of Sydney Business School

**Paper:** https://arxiv.org/abs/2609.20274

## Citation

If you use this code in your research, please cite the accompanying paper.

This repository contains the code for the three empirical experiments in Section 5
of the GES paper. Each folder has model code, an experiment script, a plotting
script, and a README with more details.

| Experiment | Folder and full instructions | Tested Python version |
|---|---|---|
| Realised covariance | [realised_covariance_github](realised_covariance_github/README.md) | 3.14.6 |
| Turbulence distributions (WES) | [turbulence_github](turbulence_github/README.md) | 3.13.15 |
| Functional electricity load | [functional_github](functional_github/README.md) | 3.13.15 |

## Set up one experiment

Open a new terminal **inside the experiment folder** you want to run.
Use the Python version listed above. You can check it with `python --version`.
Set up each experiment separately because the required package versions differ.

In Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, replace the activation command with `source .venv/bin/activate`.

Create a folder called `data` inside the experiment folder and put the required
data file there. Raw data are not included in this repository. The linked READMEs
explain where to get the data and which file to use.

## Run the realised covariance experiment

From `realised_covariance_github`, use your licensed Refinitiv DataScope file
named `real_cov_5d.csv.gz`:

```sh
python covariance_experiment.py --data data/real_cov_5d.csv.gz
python plot_covariance.py --results results
```

The full run can take several hours. For a short check, add
`--limit 31 --output results_smoke` to the first command. Then plot with
`--results results_smoke`. This short run skips the DM and MCS tests.

## Run the turbulence experiment

From `turbulence_github`, use `long_time_series_2011_july_07.txt`:

```sh
python turbulence_experiment.py --data data/long_time_series_2011_july_07.txt --blocks 60 30 --verify
python plot_turbulence.py --results results/60s
python plot_turbulence.py --results results/30s
```

This runs both the main 60-second experiment and the 30-second robustness check.

## Run the functional electricity experiment

From `functional_github`, use `electricityloaddiagrams20112014.zip`.
Keep the data file zipped.

```sh
python functional_experiment.py --data data/electricityloaddiagrams20112014.zip --verify
python plot_functional.py --results results
```

This runs the aggregate series and the 156-client panel.

## Results

The scripts save results in `results/` and figures in its `figures/` subfolders.
For turbulence, each block size has its own results folder. The `--verify` option
compares the new results with the included reference tables.

Existing results are not overwritten by the experiment scripts. To repeat a run,
choose a new folder with `--output` and use that folder when plotting.
Each experiment also includes a report describing the checks already completed.


This code is available under the [MIT License](LICENSE). Datasets remain subject to their own licences and access conditions.
