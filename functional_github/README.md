# Functional electricity-load experiment

The code includes GES, FPCA-ridge FAR(1), random walk, 52-week seasonal naive and running Fréchet mean.

## Files

- `functional_models.py`: forecasting methods, FAR tuning and DM test.
- `functional_experiment.py`: data preparation, aggregate/panel runs and tables.
- `plot_functional.py`: heatmaps and aggregate/client comparison figures.
- `reference/`: small, unrounded result tables exported by the original notebook.
- `reproduction_check.json`: release verification against the original outputs.

## Run

Tested with Python 3.13.15. From this folder:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python functional_experiment.py --data data/electricityloaddiagrams20112014.zip --verify
python plot_functional.py --results results
```

Download the original [UCI Electricity Load Diagrams 2011–2014 ZIP](https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip)
and place it in `data/` without extracting it. Dataset provenance:
[UCI record, DOI 10.24432/C58C86](https://doi.org/10.24432/C58C86).
UCI reports quarter-hour average power in kW; the hourly average curves retain
kW units, and squared functional losses have units kW². Cite Artur Trindade
(2015), *ElectricityLoadDiagrams20112014*, UCI Machine Learning Repository.
Use the full archive containing `LD2011_2014.txt`; raw data are not bundled.
The archive SHA-256 is recorded in `results/run_metadata.json`.
The runner refuses to overwrite a nonempty output folder; use `--output another_folder`.

## Fixed numerical design

- Repair the four all-zero spring DST slots by time interpolation. Halve the
  four autumn repeated-hour slots only when at least three have a cross-client
  total between 1.5 and 2.5 times the adjacent-Sunday baseline. Save the repair log.
- Shift interval-end timestamps back 15 minutes. Select clients using only the
  initial training interval: at least 90% active days, 95% active weeks, activity
  in the first/last 30 days, and hourly standard deviation above 1e-8.
- Average quarter-hours into hours. Each curve has 168 hourly coordinates;
  retain 208 complete calendar weeks, with 145 training and 63 evaluation weeks.
  The aggregate is the cross-sectional mean of the fixed 156-client panel.
- GES chooses alpha from 999 values between 0.001 and 0.999 using expanding past
  one-step losses. FAR tunes K in `{2,3,5,8,10,15}` and ridge in
  `{0,0.0001,0.001,0.01,0.1,1,10}` on the final 20 initial-training targets,
  separately for each series. The FPCA basis and score regression are then
  refitted at each origin. Ridge is scaled by the score Gram matrix's mean diagonal.
- Loss is mean squared error across the 168 hours. Client NMSE divides by the
  client's training mean squared deviation from its training mean curve.
- DM uses Bartlett HAC with five lags, lag covariance denominator `n-lag`, and
  two-sided Student t with `n-1` degrees of freedom. Panel inference first
  averages normalized client loss differentials within each forecast week,
  preserving cross-client dependence. Individual-client significance counts
  are descriptive and are not independent tests.

The saved outputs contain weekly losses, selected alpha paths, FAR tuning,
client IDs, aggregate forecasts, data/settings metadata, and comparison tables.
`--verify` compares five final reference tables at `rtol=2e-6, atol=1e-8`;
minor platform/BLAS differences are expected. BLAS is limited to one thread
for these small matrix problems. Optional MCS, exploratory outlier sensitivity
cells and interactive notebook diagnostics are omitted.

This code is available under the [MIT License](LICENSE).
Datasets remain subject to their own licences and access conditions.
