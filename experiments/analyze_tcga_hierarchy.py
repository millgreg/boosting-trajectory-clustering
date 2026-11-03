"""Pre-specified analysis of the TCGA hierarchy study, written before any result existed.

Implements the plan in experiments/run_tcga_hierarchy.py. Gains are measured against the
better of raw and scaled K-Means on the same cohort. Intervals are 95% bootstrap intervals
with 20,000 resamples over groupings (Level A) or tasks (Level B).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'experiments', 'results', 'tcga_hierarchy.csv')
OUT = os.path.join(ROOT, 'experiments', 'results', 'tcga_hierarchy_summary.csv')
SUPERVISED = ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']


def bootstrap(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    return np.percentile(x[np.random.RandomState(seed).randint(0, len(x), (n, len(x)))].mean(axis=1), [2.5, 97.5])


def table(d):
    t = d.groupby(['Level', 'Task', 'Proxy', 'grouping', 'Method']).ARI.mean().unstack()
    t['Best K-Means'] = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    for m in SUPERVISED:
        t[f'{m} gain'] = t[m] - t['Best K-Means']
    return t


def main():
    d = pd.read_csv(RESULTS)
    t = table(d)
    rows = []
    pd.set_option('display.width', 250)

    # Level A: real organ-system grouping versus random groupings of the same size.
    a = t.loc['A']
    print('Level A, mean ARI by grouping')
    print(a[SUPERVISED + ['K-Means raw', 'K-Means scaled']].round(3).to_string())
    for m in SUPERVISED:
        real = a.xs('real', level='Proxy')[f'{m} gain'].groupby('Task').mean()
        random = a.xs('random', level='Proxy')[f'{m} gain'].groupby('Task').mean()
        contrast = (real - random).dropna()
        lo, hi = bootstrap(contrast)
        rows.append({'Level': 'A', 'Comparison': f'{m}: real minus random gain', 'Units': len(contrast),
                     'Real gain': real.mean(), 'Random gain': random.mean(), 'Difference': contrast.mean(),
                     'Lower': lo, 'Upper': hi, 'Positive in': int((contrast > 5e-4).sum())})

    # Level B: molecular subtypes within a cancer type.
    b = t.loc['B']
    print('\nLevel B, mean ARI by task')
    print(b[SUPERVISED + ['K-Means raw', 'K-Means scaled']].round(3).to_string())
    for label, x in [('BTC gain over best K-Means', b['BTC gain']),
                     ('BTC minus TreeSHAP', b['BTC'] - b['TreeSHAP']),
                     ('BTC minus RF leaves with SVD', b['BTC'] - b['RF leaves with SVD']),
                     ('BTC minus RFC spectral', b['BTC'] - b['RFC spectral']),
                     ('RFC spectral gain over best K-Means', b['RFC spectral gain']),
                     ('TreeSHAP gain over best K-Means', b['TreeSHAP gain'])]:
        lo, hi = bootstrap(x)
        rows.append({'Level': 'B', 'Comparison': label, 'Units': len(x), 'Difference': x.mean(),
                     'Lower': lo, 'Upper': hi, 'Positive in': int((x > 5e-4).sum())})

    summary = pd.DataFrame(rows)
    summary['Verdict'] = np.where((summary.Lower > 0) | (summary.Upper < 0), 'difference', 'no claim')
    summary.to_csv(OUT, index=False)
    print('\nPre-specified comparisons')
    print(summary.round(3).to_string(index=False))

    # Level B diagnostic: homogeneity versus BTC gain across tasks.
    h = d[(d.Level == 'B') & (d.Method == 'BTC')].groupby('Task').Mean_homogeneity.mean()
    g = b['BTC gain'].droplevel(['Proxy', 'grouping'])
    rho = spearmanr(h.loc[g.index], g).statistic
    draws = []
    rng = np.random.RandomState(0)
    for _ in range(5000):
        i = rng.randint(0, len(g), len(g))
        draws.append(spearmanr(h.loc[g.index].values[i], g.values[i]).statistic)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    print(f'\nLevel B diagnostic: Spearman correlation of homogeneity with BTC gain {rho:.3f} [{lo:.3f}, {hi:.3f}] over {len(g)} tasks')


if __name__ == '__main__':
    main()
