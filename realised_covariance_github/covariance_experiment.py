"""Reproduce the rolling covariance experiment; run with --help for input options."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from covariance_models import (
    airm_geodesic, covariance_scores, fit_ewma, fit_gar, fit_ges,
    fit_har_drd, fit_vech_har, frechet_mean, gar_forecast,
    har_drd_forecast, spd_project, vech_har_forecast,
)

METHODS = ['GES', 'Geodesic AR', 'Scalar vech-HAR', 'HAR-DRD',
           'EWMA', 'Random walk', 'Frechet mean']
METRICS = ['AIRM2', 'Frobenius', 'NLL']
WINDOW, REFIT = 1000, 30
GRID = np.arange(0.01, 1.00, 0.01)


def load_data(path):
    """Apply the original DataScope cleaning rules and recover daily returns."""
    raw = pd.read_csv(path, usecols=['#RIC', 'Date-Time', 'Last'],
                      dtype={'#RIC': 'string', 'Date-Time': 'string', 'Last': 'float64'})
    assets = sorted(raw['#RIC'].dropna().unique().tolist())
    if len(assets) != 5 or raw.empty:
        raise ValueError('Expected a nonempty DataScope extract with exactly five RICs.')
    raw['date'] = pd.to_datetime(raw['Date-Time'].str[:10], format='%Y-%m-%d')
    raw['time'] = raw['Date-Time'].str[11:16]
    start, end = pd.Timestamp('2000-01-01'), raw['date'].max()
    cal = xcals.get_calendar('XNYS', start=start - pd.Timedelta(days=10),
                            end=end + pd.Timedelta(days=10))
    schedule = cal.schedule.loc[start:end, ['open', 'close']]
    schedule = schedule[(schedule['close'] - schedule['open']) == pd.Timedelta(minutes=390)]
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    rows = raw[raw['date'].isin(sessions) & raw['time'].between('09:30', '16:00')]
    if rows.duplicated(['date', 'time', '#RIC']).any():
        raise ValueError('Duplicate RIC/date/time observations require explicit treatment.')
    times = pd.date_range('09:30', '16:00', freq='5min').strftime('%H:%M')
    index = pd.MultiIndex.from_product([sessions, times], names=['date', 'time'])
    prices = rows.pivot(index=['date', 'time'], columns='#RIC', values='Last').reindex(
        index=index, columns=assets)
    if (prices <= 0).any().any():
        raise ValueError('Non-positive prices cannot be log transformed.')
    logp = np.log(prices)
    prev, nxt = logp.groupby(level='date').shift(1), logp.groupby(level='date').shift(-1)
    jump_in, jump_out = logp - prev, nxt - logp
    bad = ((jump_in.abs() >= .10) & (jump_out.abs() >= .10)
           & (jump_in * jump_out < 0) & ((nxt - prev).abs() <= .02))
    prices = prices.mask(bad)
    missing = prices.isna().to_numpy().reshape(len(sessions), len(times), 5)
    run = np.zeros((len(sessions), 5), dtype=np.int16)
    longest = np.zeros_like(run)
    for j in range(len(times)):
        run = np.where(missing[:, j], run + 1, 0)
        longest = np.maximum(longest, run)
    kept = (~missing[:, 0].any(axis=1) & (missing.sum(axis=1) <= 2).all(axis=1)
            & (longest <= 1).all(axis=1))
    dates = sessions[kept]
    clean = prices.groupby(level='date', sort=False).ffill().loc[pd.IndexSlice[dates, :], :]
    p = clean.to_numpy(dtype=float).reshape(len(dates), len(times), 5)
    r = 100 * np.diff(np.log(p), axis=1)
    sigma = np.einsum('dti,dtj->dij', r, r)

    # NLL/GMV use the notebook's raw exact open and last valid session quote.
    dt = pd.to_datetime(raw['Date-Time'], errors='coerce', format='mixed', utc=True)
    raw['datetime'] = dt.dt.tz_convert('America/New_York').dt.tz_localize(None)
    valid = raw['datetime'].notna() & np.isfinite(raw['Last']) & (raw['Last'] > 0)
    clock_time = raw['datetime'].dt.time
    regular = ((clock_time >= pd.Timestamp('09:30').time())
               & (clock_time <= pd.Timestamp('16:00').time()))
    panel = raw[valid & regular].pivot_table(
        index='datetime', columns='#RIC', values='Last', aggfunc='last').sort_index()[assets]
    opening = panel.reindex(dates + pd.Timedelta(hours=9, minutes=30)).to_numpy()
    closing = panel.groupby(panel.index.normalize()).last().reindex(dates).to_numpy()
    log_returns = 100 * (np.log(closing) - np.log(opening))
    if len(dates) <= WINDOW or not np.isfinite(sigma).all() or not np.isfinite(log_returns).all():
        raise ValueError('Insufficient dates or invalid covariance/return observations.')
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError('The retained covariance matrices must be positive definite.')
    return dates, assets, sigma, log_returns, np.exp(log_returns / 100) - 1


def gmv_weights(H):
    H = spd_project(H)
    one = np.ones(H.shape[0])
    x = np.linalg.solve(H, one)
    denom = float(one @ x)
    if abs(denom) < 1e-14:
        raise FloatingPointError('GMV denominator is numerically zero.')
    return x / denom

def annualized_return(r, periods=252):
    r = np.asarray(r, dtype=float)
    return float(np.mean(r) * periods)

def annualized_volatility(r, periods=252):
    r = np.asarray(r, dtype=float)
    return float(np.std(r, ddof=1) * np.sqrt(periods))

def maximum_drawdown(r):
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(np.min(drawdown))

def sharpe_zero_rf(r, periods=252):
    ar = annualized_return(r, periods)
    av = annualized_volatility(r, periods)
    return float(ar / av) if av > 0 else np.nan

def gaussian_var_es(sigma, alpha):
    sigma = float(max(sigma, 1e-12))
    z = norm.ppf(alpha)
    q = z * sigma
    es = -norm.pdf(z) / alpha * sigma
    return (q, es)

def quantile_loss_one(y, q, alpha):
    return float((alpha - float(y < q)) * (y - q))

def joint_var_es_loss_one(y, q, es, alpha):
    term1 = -np.log((alpha - 1.0) / es)
    term2 = -((y - q) * (alpha - float(y <= q))) / (alpha * es)
    return float(term1 + term2)

def summarize_portfolio(group):
    r = group['portfolio_return'].dropna().to_numpy(dtype=float)
    mdd = maximum_drawdown(r)
    return pd.Series({'AR': annualized_return(r), 'AV': annualized_volatility(r), 'MDD': mdd, 'MDD magnitude': -mdd, 'Sharpe (0% rf, extra)': sharpe_zero_rf(r), 'Mean gross leverage': group['gross_leverage'].mean()})

def hac_mean_test(x, max_lag=5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    xbar = x.mean()
    u = x - xbar
    gamma0 = np.dot(u, u) / n
    long_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = np.dot(u[lag:], u[:-lag]) / n
        long_var += 2 * weight * gamma
    se = np.sqrt(long_var / n)
    z = xbar / se
    p = 2 * norm.sf(abs(z))
    return (xbar, z, p)

def _moving_block_bootstrap_means(L, B=5000, block_len=22, seed=0):
    """Bootstrap distribution of column means of the T x m loss matrix L"""
    rng = np.random.default_rng(seed)
    T = L.shape[0]
    n_blocks = int(np.ceil(T / block_len))
    boot_means = np.empty((B, L.shape[1]))
    for b in range(B):
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:T]
        boot_means[b] = L[idx].mean(axis=0)
    return boot_means

def model_confidence_set(loss_df, B=5000, block_len=22, seed=0):
    """MCS with the T_R (range) statistic."""
    loss_df = loss_df.dropna()
    methods = list(loss_df.columns)
    L = loss_df.to_numpy(dtype=float)
    sample_means = L.mean(axis=0)
    boot_means = _moving_block_bootstrap_means(L, B=B, block_len=block_len, seed=seed)
    active = list(range(len(methods)))
    records = []
    p_running = 0.0
    while len(active) > 1:
        mu = sample_means[active]
        bm = boot_means[:, active]
        dbar = mu[:, None] - mu[None, :]
        dboot = bm[:, :, None] - bm[:, None, :]
        var_ij = ((dboot - dbar[None]) ** 2).mean(axis=0)
        sd_ij = np.sqrt(np.maximum(var_ij, 1e-300))
        np.fill_diagonal(sd_ij, 1.0)
        t_ij = dbar / sd_ij
        T_R = np.max(np.abs(t_ij))
        t_boot = np.abs(dboot - dbar[None]) / sd_ij[None]
        T_R_boot = t_boot.reshape(B, -1).max(axis=1)
        p_step = float(np.mean(T_R_boot > T_R))
        p_running = max(p_running, p_step)
        worst_local = int(np.argmax(t_ij.max(axis=1)))
        worst = active[worst_local]
        records.append({'method': methods[worst], 'mean loss': sample_means[worst], 'MCS p-value': p_running})
        active.remove(worst)
    records.append({'method': methods[active[0]], 'mean loss': sample_means[active[0]], 'MCS p-value': 1.0})
    out = pd.DataFrame(records)
    out['in 90% MCS'] = out['MCS p-value'] >= 0.1
    out['in 75% MCS'] = out['MCS p-value'] >= 0.25
    return out.set_index('method')


def summarize(scores, portfolio, output, inference=True):
    """Save the notebook's forecast, portfolio, tail, DM and MCS tables."""
    scores.groupby('method')[METRICS].mean().sort_values('NLL').to_csv(output / 'forecast_summary.csv')
    portfolio.groupby('method').apply(summarize_portfolio).sort_values('AV').to_csv(
        output / 'portfolio_summary.csv')
    portfolio[portfolio.method != 'Equal weight (1/N)'].groupby('method').agg({
        'QLoss1%': 'sum', 'QLoss5%': 'sum', 'JointLoss1%': 'mean',
        'JointLoss5%': 'mean'}).sort_values('JointLoss5%').to_csv(output / 'tail_summary.csv')
    if not inference:
        return
    rows, mcs = [], {}
    for metric in METRICS:
        wide = scores.pivot(index='date', columns='method', values=metric)[METHODS]
        for method in METHODS[1:]:
            mean, z, p = hac_mean_test((wide['GES'] - wide[method]).to_numpy(), max_lag=5)
            rows.append({'metric': metric, 'GES vs': method, 'mean diff': mean, 'z': z, 'p-value': p})
        table = model_confidence_set(wide, B=5000, block_len=22, seed=42)
        table.to_csv(output / f'mcs_{metric.lower()}.csv')
        mcs[metric] = table['MCS p-value']
    pd.DataFrame(rows).to_csv(output / 'dm_tests.csv', index=False)
    pd.DataFrame(mcs).reindex(METHODS).to_csv(output / 'mcs_summary.csv')


def run(dates, assets, sigma, log_returns, simple_returns, output, limit=None):
    """One-day-ahead forecasts; all estimation information ends at the origin."""
    stop = len(sigma) - 1 if limit is None else min(WINDOW - 1 + limit, len(sigma) - 1)
    score_rows, portfolio_rows, paths, saved = [], [], [], []
    for origin in range(WINDOW - 1, stop):
        train = sigma[origin - WINDOW + 1:origin + 1]
        mu = log_returns[origin - WINDOW + 1:origin + 1].mean(axis=0)
        if (origin - WINDOW + 1) % REFIT == 0:
            print(f'Refit {dates[origin].date()} ({origin - WINDOW + 2}/{stop - WINDOW + 1})', flush=True)
            alpha, level = fit_ges(train, GRID)
            ew_alpha, ew_level = fit_ewma(train, GRID)
            gar_mu, phi = fit_gar(train, GRID)
            vh, drd = fit_vech_har(train), fit_har_drd(train)
            fm = frechet_mean(train)
        else:
            level = airm_geodesic(level, sigma[origin], alpha)
            ew_level = spd_project((1 - ew_alpha) * ew_level + ew_alpha * sigma[origin])
        # HAR forecast functions only need the latest 20 observations.
        history = sigma[origin - 19:origin + 1]
        forecasts = [level, gar_forecast(sigma[origin], gar_mu, phi),
                     vech_har_forecast(history, vh), har_drd_forecast(history, drd),
                     ew_level, sigma[origin], fm]
        target, date = origin + 1, dates[origin + 1]
        demeaned = log_returns[target] - mu
        saved.append([spd_project(h) for h in forecasts])
        for method, h in zip(METHODS, saved[-1]):
            score_rows.append({'date': date, 'method': method,
                               **covariance_scores(sigma[target], h, demeaned)})
            w = gmv_weights(h)
            tail_return = float(w @ (demeaned / 100))
            vol = np.sqrt(max(float(w @ h @ w), 0.0)) / 100
            record = {'date': date, 'method': method,
                      'portfolio_return': float(w @ simple_returns[target]),
                      'tail_return': tail_return, 'predicted_volatility': vol,
                      'gross_leverage': float(np.abs(w).sum())}
            for probability in [.01, .05]:
                q, es = gaussian_var_es(vol, probability)
                label = f'{100 * probability:.0f}%'
                record['QLoss' + label] = quantile_loss_one(tail_return, q, probability)
                record['JointLoss' + label] = joint_var_es_loss_one(tail_return, q, es, probability)
            portfolio_rows.append(record)
        portfolio_rows.append({'date': date, 'method': 'Equal weight (1/N)',
                               'portfolio_return': float(np.ones(5) / 5 @ simple_returns[target]),
                               'gross_leverage': 1.0})
        paths.append({'origin': dates[origin], 'target': date,
                      'alpha': alpha, 'phi': phi, 'ewma_alpha': ew_alpha})
    scores, portfolio = pd.DataFrame(score_rows), pd.DataFrame(portfolio_rows)
    scores.to_csv(output / 'forecast_scores.csv', index=False)
    portfolio.to_csv(output / 'portfolio_daily.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'parameters.csv', index=False)
    np.savez_compressed(output / 'forecasts.npz', dates=dates[WINDOW:stop + 1].to_numpy(dtype='datetime64[ns]'),
                        assets=np.asarray(assets), methods=np.asarray(METHODS),
                        observed=sigma[WINDOW:stop + 1], forecast=np.asarray(saved))
    summarize(scores, portfolio, output, inference=limit is None)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Licensed DataScope CSV or CSV.GZ')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--limit', type=int, help='First N forecasts for a smoke check; skips DM/MCS')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new/empty output directory to avoid overwriting results')
    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        scores = run(*data, args.output, args.limit)
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'matplotlib', 'exchange-calendars', 'threadpoolctl']}
    with args.data.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'python': platform.python_version(), 'versions': versions, 'input_sha256': digest,
                'assets': data[1], 'retained_days': len(data[0]), 'evaluation_days': len(scores) // 7,
                'window': WINDOW, 'refit_every': REFIT, 'grid': GRID.tolist(),
                'smoke_check': args.limit is not None, 'DM_bandwidth': 5,
                'MCS': {'replications': 5000, 'block_length': 22, 'seed': 42, 'statistic': 'range'}}
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved results in {args.output}')


if __name__ == '__main__':
    main()
