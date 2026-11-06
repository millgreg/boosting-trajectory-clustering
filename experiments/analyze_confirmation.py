"""Pre-specified analysis of the confirmation run, written before any result existed.

Follows experiments/confirmation/PROTOCOL.md: dataset-level means, 95% bootstrap intervals with
20,000 resamples, and a claim of difference only when the interval excludes zero.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'experiments', 'results', 'confirmation_results.csv')
OUT = os.path.join(ROOT, 'experiments', 'results', 'confirmation_summary.csv')

COMPARISONS = [
    ('C1 primary', 'BTC', 'TreeSHAP'),
    ('C1 secondary', 'BTC', 'Per-feature sums'),
    ('C2 secondary', 'Historic residuals', 'Final margin'),
    ('C2 secondary', 'Historic residuals', 'Historic residuals, order shuffled'),
    ('C3 descriptive', 'BTC', 'RF leaves with SVD'),
    ('C3 descriptive', 'BTC', 'K-Means raw'),
    ('C3 descriptive', 'BTC', 'K-Means scaled'),
    ('C3 descriptive', 'BTC', 'UMAP raw'),
    ('C3 descriptive', 'BTC', 'UMAP scaled'),
]


def bootstrap(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    draws = x[np.random.RandomState(seed).randint(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def main():
    d = pd.read_csv(RESULTS)
    failed = d[d.Method == 'FAILED']
    d = d[d.Method != 'FAILED']
    print('Datasets with results', d.did.nunique(), '| failed', failed.did.nunique())
    if len(failed):
        print(failed[['Dataset', 'error']].to_string(index=False))
    per_grouping = d.groupby(['did', 'Dataset', 'grouping', 'Method']).ARI.mean().reset_index()
    table = per_grouping.groupby(['did', 'Dataset', 'Method']).ARI.mean().unstack()

    rows = []
    for label, a, b in COMPARISONS:
        pair = table[[a, b]].dropna()
        diff = pair[a] - pair[b]
        lo, hi = bootstrap(diff)
        rows.append({'Claim': label, 'Comparison': f'{a} minus {b}', 'Datasets': len(diff),
                     'Mean A': pair[a].mean(), 'Mean B': pair[b].mean(), 'Difference': diff.mean(),
                     'Lower': lo, 'Upper': hi, 'A higher on': int((diff > 5e-4).sum()), 'A lower on': int((diff < -5e-4).sum()),
                     'Verdict': 'difference' if lo > 0 or hi < 0 else 'no claim'})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(summary.round(3).to_string(index=False))

    print('\nMean ARI by method')
    print(table.mean().sort_values(ascending=False).round(3).to_string())

    gap = (table['K-Means raw'] - table['K-Means scaled']).abs()
    print(f'\nC4: mean absolute raw-versus-standardized K-Means gap {gap.mean():.3f}, '
          f'exceeding 0.05 on {int((gap > 0.05).sum())} of {len(gap)} datasets. BTC is invariant by construction.')

    btc = per_grouping[per_grouping.Method == 'BTC'].merge(
        d[d.Method == 'BTC'].groupby(['did', 'grouping']).Mean_homogeneity.mean().reset_index(), on=['did', 'grouping'])
    rho = spearmanr(btc.Mean_homogeneity, btc.ARI).statistic
    print(f'E1 (exploratory): Spearman correlation of homogeneity with BTC ARI across {len(btc)} groupings: {rho:.3f}')


if __name__ == '__main__':
    main()
