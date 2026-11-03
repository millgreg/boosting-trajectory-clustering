"""Pre-specified analysis of the within-cohort proxy study, written before any result existed.

Implements Q1 and Q2 from experiments/run_tcga_within_cohort.py. Intervals resample cancer types,
because proxies within the same cancer type share a cohort and are not independent.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'experiments', 'results')
SUPERVISED = ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']


def cluster_bootstrap(values, groups, n=20000, seed=0):
    """Mean over units, resampling whole groups."""
    frame = pd.DataFrame({'v': values, 'g': groups})
    sums = frame.groupby('g').v.sum().values
    counts = frame.groupby('g').v.count().values
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(sums), (n, len(sums)))
    return np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [2.5, 97.5])


def gains(path, by):
    d = pd.read_csv(path)
    t = d.groupby(by + ['Method']).ARI.mean().unstack()
    best = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    return pd.DataFrame({m: t[m] - best for m in SUPERVISED}).reset_index()


def main():
    within = gains(os.path.join(RESULTS, 'tcga_within_cohort.csv'), ['Task', 'Proxy'])
    hierarchy = gains(os.path.join(RESULTS, 'tcga_hierarchy.csv'), ['Level', 'Task', 'Proxy', 'grouping'])
    hierarchy = hierarchy[hierarchy.Level == 'B'].set_index('Task')
    pd.set_option('display.width', 220)
    print(f'Task-proxy pairs: {len(within)} across {within.Task.nunique()} cancer types')
    print(within.Proxy.value_counts().to_string())

    rows = []
    for m in SUPERVISED:
        lo, hi = cluster_bootstrap(within[m].values, within.Task.values)
        rows.append({'Question': 'Q1 gain of within-cohort proxies', 'Method': m, 'Difference': within[m].mean(),
                     'Lower': lo, 'Upper': hi, 'Positive pairs': f'{int((within[m] > 5e-4).sum())} of {len(within)}'})
        per_type = within.groupby('Task')[m].mean()
        contrast = (per_type - hierarchy[m].reindex(per_type.index)).dropna()
        lo, hi = cluster_bootstrap(contrast.values, contrast.index.values)
        rows.append({'Question': 'Q2 within-cohort minus cohort-versus-rest gain', 'Method': m, 'Difference': contrast.mean(),
                     'Lower': lo, 'Upper': hi, 'Positive pairs': f'{int((contrast > 5e-4).sum())} of {len(contrast)} types'})
    summary = pd.DataFrame(rows)
    summary['Verdict'] = np.where(summary.Lower > 0, 'yes', np.where(summary.Upper < 0, 'opposite', 'no claim'))
    summary.to_csv(os.path.join(RESULTS, 'tcga_within_cohort_summary.csv'), index=False)
    print('\nPre-specified questions')
    print(summary.round(3).to_string(index=False))

    print('\nDescriptive: mean gain by proxy type')
    print(within.groupby('Proxy')[SUPERVISED].mean().round(3).to_string())
    print('\nDescriptive: per task-proxy pair')
    print(within.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
