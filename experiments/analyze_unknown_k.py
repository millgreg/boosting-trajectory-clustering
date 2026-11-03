"""Pre-specified analysis of the unknown-K study, written before any result existed."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'experiments', 'results')
SUPERVISED = ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']


def bootstrap(x, groups, n=20000, seed=0):
    frame = pd.DataFrame({'v': np.asarray(x, float), 'g': groups}).dropna()
    sums, counts = frame.groupby('g').v.sum().values, frame.groupby('g').v.count().values
    draws = np.random.RandomState(seed).randint(0, len(sums), (n, len(sums)))
    return np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [2.5, 97.5])


def units(d):
    """One row per unit and method: datasets for confirmation, tasks or task-proxy pairs for TCGA."""
    out = []
    c = d[d.Study == 'confirmation'].groupby(['did', 'grouping', 'Method'])[['ARI_chosen', 'ARI_true_K', 'exact', 'abs_error']].mean() \
        .groupby(['did', 'Method']).mean().reset_index()
    c['Part'], c['Unit'], c['Group'] = 'Confirmation datasets', c.did, c.did
    out.append(c)
    h = d[d.Study == 'tcga_hierarchy'].groupby(['Level', 'Task', 'Method'])[['ARI_chosen', 'ARI_true_K', 'exact', 'abs_error']].mean().reset_index()
    h['Part'] = np.where(h.Level == 'A', 'TCGA organ systems', 'TCGA molecular subtypes')
    h['Unit'], h['Group'] = h.Task, h.Task
    out.append(h)
    w = d[d.Study == 'tcga_within'].groupby(['Task', 'Proxy', 'Method'])[['ARI_chosen', 'ARI_true_K', 'exact', 'abs_error']].mean().reset_index()
    w['Part'], w['Unit'], w['Group'] = 'TCGA within-cohort', w.Task + '/' + w.Proxy, w.Task
    out.append(w)
    return pd.concat(out, ignore_index=True)[['Part', 'Unit', 'Group', 'Method', 'ARI_chosen', 'ARI_true_K', 'exact', 'abs_error']]


def main():
    d = pd.read_csv(os.path.join(RES, 'unknown_k.csv'))
    d['exact'] = (d.K_chosen == d.K_true).astype(float)
    d['abs_error'] = (d.K_chosen - d.K_true).abs()
    beyond = d[d.Study == 'confirmation'].groupby('did').apply(lambda x: (x.K_true > x.K_max).any()).sum()
    print('Confirmation datasets with true K beyond the candidate range:', int(beyond))
    u = units(d)
    rows = []
    for part, p in u.groupby('Part'):
        wide = p.pivot_table(index=['Unit', 'Group'], columns='Method', values='ARI_chosen').reset_index()
        for m in SUPERVISED[1:] + ['K-Means raw', 'K-Means scaled']:
            x = wide['BTC'] - wide[m]
            lo, hi = bootstrap(x, wide.Group)
            rows.append({'Part': part, 'Question': 'U1', 'Comparison': f'BTC minus {m}', 'Difference': x.mean(), 'Lower': lo, 'Upper': hi})
        for m in SUPERVISED:
            for base in ['K-Means raw', 'K-Means scaled']:
                x = wide[m] - wide[base]
                lo, hi = bootstrap(x, wide.Group)
                rows.append({'Part': part, 'Question': 'U2', 'Comparison': f'{m} gain over {base}', 'Difference': x.mean(), 'Lower': lo, 'Upper': hi})
        for m, q in p.groupby('Method'):
            x = q.ARI_chosen - q.ARI_true_K
            lo, hi = bootstrap(x.values, q.Group.values)
            rows.append({'Part': part, 'Question': 'U3', 'Comparison': f'{m}: chosen K minus true K', 'Difference': x.mean(),
                         'Lower': lo, 'Upper': hi, 'Exact K share': q.exact.mean(), 'Mean absolute K error': q.abs_error.mean()})
    summary = pd.DataFrame(rows)
    summary['Verdict'] = np.where((summary.Lower > 0) | (summary.Upper < 0), 'difference', 'no claim')
    summary.to_csv(os.path.join(RES, 'unknown_k_summary.csv'), index=False)
    pd.set_option('display.width', 240)
    print(u.groupby(['Part', 'Method'])[['ARI_chosen', 'ARI_true_K', 'exact']].mean().round(3).to_string())
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
