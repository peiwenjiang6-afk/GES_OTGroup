"""Reproduce the 60-second main experiment and 30-second robustness analysis."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

from wes_models import rolling_atmd1, rolling_ges, rolling_references

HEIGHTS = {3: 0.7, 2: 1.4, 4: 2.1, 5: 2.8, 1: 3.5}
RAW_MD5 = 'b6cfab0e8af9b4b2052d754ccfb29ee5'
SOURCE = 'https://doi.org/10.5281/zenodo.20045165'
METHODS = ('GES', 'ATM_d(1)', 'Random walk', 'Frechet mean')
PACKAGES = ('numpy', 'pandas', 'scipy', 'matplotlib', 'arch')


def load_data(path):
    """Require the unedited published file, continuous 20-Hz data and zero flags."""
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    if digest != RAW_MD5:
        raise ValueError('Raw file checksum differs; download the complete original TXT.')
    frame = pd.read_csv(path, skiprows=[0, 2, 3], na_values=['NAN', 'NaN'])
    times = pd.to_datetime(frame['TIMESTAMP'], format='mixed')
    if len(frame) != 540001 or times.isna().any() or not (
        times.diff().iloc[1:] == pd.Timedelta(milliseconds=50)
    ).all():
        raise ValueError('Unexpected sample count, missing timestamps or irregular spacing.')
    flags = [f'{kind}_f_{s}' for s in HEIGHTS
             for kind in ('del_T', 'track', 'amp_h', 'amp_l')]
    if frame[flags].isna().any().any() or (frame[flags] != 0).any().any():
        raise ValueError('Missing/nonzero diagnostic flags; explicit quality control required.')
    values = frame[[f'Uz_{s}' for s in HEIGHTS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite vertical velocities; no imputation is performed.')
    return values, times.to_numpy(dtype='datetime64[ns]')


def build_panel(values, times, seconds):
    """Complete nonoverlapping blocks; raw empirical inverse-CDF midpoint quantiles."""
    size = 20 * seconds
    count = len(values) // size
    u = (np.arange(1000) + .5) / 1000
    Q = np.empty((count, len(HEIGHTS), len(u)))
    for j in range(len(HEIGHTS)):
        blocks = values[:count * size, j].reshape(count, size)
        Q[:, j] = np.quantile(blocks, u, axis=1, method='inverted_cdf').T
    return Q, u, times[np.arange(count) * size]


def dm_test(difference, bandwidth):
    """GES-minus-benchmark; Bartlett HAC with available-pair covariance and t(n-1)."""
    d = np.asarray(difference, dtype=float)
    if len(d) < 3 or not np.isfinite(d).all() or not 0 <= bandwidth < len(d):
        raise ValueError('Invalid loss differential or DM bandwidth.')
    center = d - float(d.mean())
    variance = float(np.mean(center**2))
    for lag in range(1, bandwidth + 1):
        covariance = float(np.mean(center[lag:] * center[:-lag]))
        variance += 2 * (1 - lag / (bandwidth + 1.0)) * covariance
    if variance <= 0:
        return np.nan, np.nan
    statistic = float(d.mean() / np.sqrt(variance / len(d)))
    return statistic, float(2 * stats.t.sf(abs(statistic), df=len(d) - 1))


def summarize(losses, bandwidth, bootstrap_length):
    """Height-specific DM tests and 90% range-statistic MCS; no pooling of heights."""
    mcs = MCS(losses, size=.10, reps=5000, block_size=bootstrap_length,
              method='R', bootstrap='stationary', seed=0)
    mcs.compute()
    rows = []
    for method in METHODS:
        mean = float(losses[method].mean())
        dm, p = (np.nan, np.nan) if method == 'GES' else dm_test(
            (losses['GES'] - losses[method]).to_numpy(), bandwidth)
        mcs_p = float(mcs.pvalues.loc[method, 'Pvalue'])
        rows.append({'Method': method, 'Mean squared W2 error': mean,
                     'Root mean squared W2 error': np.sqrt(mean),
                     'DM statistic vs GES': dm, 'DM p-value vs GES': p,
                     'MCS p-value': mcs_p, 'In 90% MCS': mcs_p > .10})
    return pd.DataFrame(rows).sort_values('Mean squared W2 error').reset_index(drop=True)


def run_experiment(values, times, seconds, output):
    Q, u, starts = build_panel(values, times, seconds)
    ntrain = int(.7 * len(Q))
    bandwidth, bootstrap_length = (5, 10) if seconds == 60 else (10, 20)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'quantile_panel.npz', Q=Q, u_grid=u,
                        block_start=starts, sensor_ids=list(HEIGHTS),
                        heights_m=list(HEIGHTS.values()), axis_order='block, height, quantile')
    tables = []
    for j, (sensor, height) in enumerate(HEIGHTS.items()):
        print(f'{seconds}s: height {height:.1f} m ...', flush=True)
        observed = Q[:, j]
        ges, alpha = rolling_ges(observed, ntrain)
        rw, mean = rolling_references(observed, ntrain)
        atm, beta = rolling_atmd1(observed, u, ntrain)
        predictions = dict(zip(METHODS, (ges, atm, rw, mean)))
        for method, forecast in predictions.items():
            if forecast.shape != observed[ntrain:].shape or not np.isfinite(forecast).all() or (
                np.diff(forecast, axis=1) < -1e-10
            ).any():
                raise RuntimeError(f'Invalid forecast quantiles: {method}')
        losses = pd.DataFrame({name: np.mean((pred - observed[ntrain:])**2, axis=1)
                               for name, pred in predictions.items()},
                              index=pd.Index(np.arange(ntrain, len(Q)), name='target_block'))
        table = summarize(losses, bandwidth, bootstrap_length)
        target = output / f'sensor_{sensor}'
        target.mkdir()
        losses.assign(target_block_start=starts[ntrain:]).to_csv(target / 'block_losses.csv')
        table.to_csv(target / 'summary.csv', index=False)
        np.savez_compressed(target / 'forecasts.npz', observed=observed, u_grid=u,
                            block_start=starts, forecast_block_start=starts[ntrain:],
                            ges=ges, atmd=atm, random_walk=rw, frechet_mean=mean,
                            alpha=alpha, beta=beta)
        tables.append(table.assign(Sensor=sensor, Height_m=height, Block_seconds=seconds))
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output / 'all_height_results.csv', index=False)
    metadata = dict(data_source=SOURCE, raw_md5=RAW_MD5, block_seconds=seconds,
                    raw_rows=len(values), samples_per_block=20 * seconds,
                    trailing_samples_excluded=len(values) - len(Q) * 20 * seconds,
                    train_fraction=.7, training_blocks=ntrain, evaluation_blocks=len(Q)-ntrain,
                    n_quantiles=1000, quantile_method='inverted_cdf; midpoint probabilities',
                    preprocessing='None; raw Uz', dm_bandwidth=bandwidth,
                    mcs_size=.10, mcs_reps=5000, mcs_block_size=bootstrap_length,
                    mcs_method='R', mcs_bootstrap='stationary', seed=0,
                    alpha_grid='linspace(0.001,0.999,999)', atmd_n_x=2201,
                    atmd_eps=1e-4, atmd_padding=.05, atmd_coefficient_bound=.999,
                    atmd_criterion='Common spatial interval; trapezoidal Lebesgue integration',
                    atmd_maps='Endpoint-preserving linear interpolation; reverse-pair inverse',
                    units='(m/s)^2', python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in PACKAGES})
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return result


def verify_results(result, reference):
    """Check every paper-summary field against independently saved corrected results."""
    keys = ['Block_seconds', 'Sensor', 'Height_m', 'Method']
    expected = pd.read_csv(reference)
    expected = expected[expected.Block_seconds.isin(result.Block_seconds.unique())]
    actual = result.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        raise AssertionError('Result rows do not match the reference configurations.')
    numbers = [c for c in expected if c != 'In 90% MCS']
    np.testing.assert_allclose(actual[numbers], expected[numbers],
                               rtol=1e-8, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(actual['In 90% MCS'], expected['In 90% MCS'])
    return dict(status='PASS', rows=len(actual), rtol=1e-8, atol=1e-10,
                fields=numbers + ['In 90% MCS'],
                maximum_absolute_difference=float(np.nanmax(abs(
                    actual[numbers].to_numpy() - expected[numbers].to_numpy()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Original downloaded TXT file.')
    parser.add_argument('--blocks', nargs='+', type=int, choices=(60, 30), default=[60, 30])
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--verify', action='store_true', help='Compare with reference_results.csv.')
    args = parser.parse_args()
    durations = list(dict.fromkeys(args.blocks))
    for seconds in durations:
        folder = args.output / f'{seconds}s'
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            parser.error(f'{folder} is not empty. Choose a new --output directory.')
    values, times = load_data(args.data)
    results = pd.concat([run_experiment(values, times, s, args.output / f'{s}s')
                         for s in durations], ignore_index=True)
    if args.verify:
        report = verify_results(results, Path(__file__).with_name('reference_results.csv'))
        (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'PASS: {report["rows"]} result rows match the saved paper results.', flush=True)
    print(f'Results saved to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
