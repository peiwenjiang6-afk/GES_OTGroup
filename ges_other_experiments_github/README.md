| Folder | Experiment | Tested Python |
|---|---|---|
| [`realised_covariance/`](realised_covariance/README.md) | Five-asset covariance forecasting with seven methods, portfolio/tail scores and DM/MCS inference | 3.14.6 |
| [`functional/`](functional/README.md) | Weekly electricity-load curves: aggregate and 156-client panel, five methods and DM inference | 3.13.15 |

Follow the README inside each folder. Use a separate virtual environment for each
experiment to preserve its tested dependency versions. Scripts run without the
original notebooks. Data and generated outputs are excluded; each README explains
the required input and the commands to generate results and figures.

## Validation

- **Functional:** the complete aggregate and 156-client experiment was rerun from
  the original UCI ZIP and compared with the notebook's saved results.
- **Covariance:** all 6,060 input covariance matrices were reconstructed and checked;
  31 fresh forecast targets cover the first two refits. Full-period summary and
  inference tables were also regenerated from saved daily results. The complete
  5,060-target forecast run was not repeated during packaging.

The per-experiment JSON reports give the comparisons and numerical tolerances.
The package retains the notebooks' distinct numerical conventions and excludes
exploratory add-ons. No code license has been assigned.
