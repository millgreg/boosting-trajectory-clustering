"""Pre-specified analysis of the proximity-variant study, written before any result existed."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'experiments', 'results')
VARIANTS = ['breiman', 'oob', 'rfgap']


def bootstrap(x, groups=None, n=20000, seed=0):
    frame = pd.DataFrame({'v': np.asarray(x, float), 'g': groups if groups is not None else np.arange(len(x))}).dropna()
    sums, counts = frame.groupby('g').v.sum().values, frame.groupby('g').v.count().values
    draws = np.random.RandomState(seed).randint(0, len(sums), (n, len(sums)))
    return np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [2.5, 97.5])


def row(study, label, x, groups=None):
    lo, hi = bootstrap(x, groups)
    x = pd.Series(x).dropna()
    return {'Study': study, 'Comparison': label, 'Units': len(x), 'Difference': x.mean(), 'Lower': lo, 'Upper': hi,
            'Positive in': int((x > 5e-4).sum()), 'Verdict': 'difference' if lo > 0 or hi < 0 else 'no claim'}


def main():
    d = pd.read_csv(os.path.join(RES, 'proximity_variants.csv'))
    rows = []

    c = d[d.Study == 'confirmation']
    ct = c.groupby(['did', 'grouping', 'Method']).ARI.mean().groupby(['did', 'Method']).mean().unstack()
    orig = pd.read_csv(os.path.join(RES, 'confirmation_results.csv'))
    orig = orig[orig.Method != 'FAILED'].groupby(['did', 'grouping', 'Method']).ARI.mean().groupby(['did', 'Method']).mean().unstack()
    rfc = pd.read_csv(os.path.join(RES, 'confirmation_rfc_exploratory.csv')).groupby(['did', 'grouping']).ARI.mean().groupby('did').mean()
    print('Confirmation mean ARI:', ct.mean().round(3).to_dict(), '| original BTC', round(orig['BTC'].mean(), 3), 'spectral RFC', round(rfc.mean(), 3))
    rows.append(row('confirmation', 'Breiman (this run) minus original spectral RFC', ct['breiman'] - rfc.reindex(ct.index)))
    for v in ['oob', 'rfgap']:
        rows.append(row('confirmation', f'P1: {v} minus Breiman', ct[v] - ct['breiman']))
    for v in VARIANTS:
        rows.append(row('confirmation', f'P3: {v} minus BTC', ct[v] - orig['BTC'].reindex(ct.index)))

    h = d[d.Study == 'tcga_hierarchy']
    ht = h.groupby(['Level', 'Task', 'Proxy', 'grouping', 'Method']).ARI.mean().unstack()
    ho = pd.read_csv(os.path.join(RES, 'tcga_hierarchy.csv')).groupby(['Level', 'Task', 'Proxy', 'grouping', 'Method']).ARI.mean().unstack()
    best = ho[['K-Means raw', 'K-Means scaled']].max(axis=1)
    for level, label in [('A', 'organ-system real groupings'), ('B', 'molecular subtypes')]:
        part = ht.loc[level]
        if level == 'A':
            part = part.xs('real', level='Proxy', drop_level=False)
        b = best.loc[level].reindex(part.index)
        print(f'TCGA {label} mean ARI:', part.mean().round(3).to_dict(), '| better K-Means', round(b.mean(), 3))
        for v in ['oob', 'rfgap']:
            rows.append(row(f'tcga {label}', f'P1: {v} minus Breiman', (part[v] - part['breiman']).values))
        for v in VARIANTS:
            rows.append(row(f'tcga {label}', f'P2: {v} gain over better K-Means', (part[v] - b).values))

    w = d[d.Study == 'tcga_within']
    wt = w.groupby(['Task', 'Proxy', 'Method']).ARI.mean().unstack()
    wo = pd.read_csv(os.path.join(RES, 'tcga_within_cohort.csv')).groupby(['Task', 'Proxy', 'Method']).ARI.mean().unstack()
    wbest = wo[['K-Means raw', 'K-Means scaled']].max(axis=1).reindex(wt.index)
    groups = wt.index.get_level_values('Task')
    print('TCGA within-cohort mean ARI:', wt.mean().round(3).to_dict(), '| better K-Means', round(wbest.mean(), 3))
    for v in ['oob', 'rfgap']:
        rows.append(row('tcga within-cohort', f'P1: {v} minus Breiman', (wt[v] - wt['breiman']).values, groups))
    for v in VARIANTS:
        rows.append(row('tcga within-cohort', f'P2: {v} gain over better K-Means', (wt[v] - wbest).values, groups))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RES, 'proximity_variants_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
