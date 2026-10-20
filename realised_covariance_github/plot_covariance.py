"""Generate covariance, cumulative-loss and component figures from saved results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covariance_models import covariance_to_correlation


def plot(results, output):
    data = np.load(results / 'forecasts.npz', allow_pickle=False)
    dates = pd.to_datetime(data['dates'])
    actual = data['observed']
    ges = data['forecast'][:, list(data['methods']).index('GES')]
    assets = data['assets']
    if actual.shape != ges.shape or actual.shape[1:] != (5, 5):
        raise ValueError('Expected aligned five-asset observed and forecast matrices.')
    scores = pd.read_csv(results / 'forecast_scores.csv', parse_dates=['date'])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    def save(fig, name):
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(output / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def observed_forecast(ax, y, pred, label):
        ax.plot(dates, y, color='black', lw=.7, label='Observed')
        ax.plot(dates, pred, color='#0072B2', lw=.7, alpha=.85, label='GES forecast')
        ax.set_ylabel(label)
        ax.grid(alpha=.2, lw=.4)
        ax.margins(x=0)

    corr = np.asarray([covariance_to_correlation(a) for a in actual])
    ges_corr = np.asarray([covariance_to_correlation(a) for a in ges])
    pairs = np.tril_indices(5, -1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout='constrained')
    observed_forecast(axes[0], np.trace(actual, axis1=1, axis2=2),
                      np.trace(ges, axis1=1, axis2=2), 'Total realised variance (%²)')
    observed_forecast(axes[1], corr[:, pairs[0], pairs[1]].mean(axis=1),
                      ges_corr[:, pairs[0], pairs[1]].mean(axis=1), 'Mean realised correlation')
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel('Date')
    save(fig, 'figure_covariance_dynamics')

    fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True, layout='constrained')
    for j, asset in enumerate(assets):
        observed_forecast(axes[0, j], actual[:, j, j], ges[:, j, j], 'Variance (%²)' if j == 0 else '')
        axes[0, j].set_title(str(asset))
    for ax, (i, j) in zip(axes[1:].flat, zip(*pairs)):
        observed_forecast(ax, corr[:, i, j], ges_corr[:, i, j], '')
        ax.set_title(f'{assets[i]} / {assets[j]}')
    axes[1, 0].set_ylabel('Correlation')
    axes[2, 0].set_ylabel('Correlation')
    for ax in axes[-1]:
        ax.tick_params(axis='x', labelrotation=45)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False)
    save(fig, 'figure_components')

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='constrained')
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#777777']
    for ax, metric in zip(axes, ['AIRM2', 'Frobenius', 'NLL']):
        wide = scores.pivot(index='date', columns='method', values=metric)
        for j, method in enumerate(data['methods']):
            ax.plot(wide.index, wide[method].cumsum(), label=method, color=colors[j],
                    lw=1.3 if method == 'GES' else .9, ls='-' if j < 4 else '--')
        ax.set_ylabel(f'Cumulative {metric}')
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=4, frameon=False)
    save(fig, 'figure_cumulative_losses')

    parameters = pd.read_csv(results / 'parameters.csv', parse_dates=['target'])
    fig, ax = plt.subplots(figsize=(8, 2.8), layout='constrained')
    ax.plot(parameters.target, parameters.alpha, color='#0072B2', lw=1)
    ax.set(xlabel='Forecast target date', ylabel='GES smoothing parameter', ylim=(0, 1))
    save(fig, 'figure_alpha')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot(args.results, args.output or args.results / 'figures')


if __name__ == '__main__':
    main()
