"""Pre-specified analysis of the alignment study (experiments/run_alignment.py), written before any result existed."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'experiments', 'results')
SUPERVISED = ['BTC', 'TreeSHAP', 'RF leaves with SVD', 'RFC spectral']


def cluster_bootstrap(values, groups, n=20000, seed=0):
    frame = pd.DataFrame({'v': np.asarray(values, float), 'g': np.asarray(groups)}).dropna()
    sums, counts = frame.groupby('g').v.sum().values, frame.groupby('g').v.count().values
    draws = np.random.RandomState(seed).randint(0, len(sums), (n, len(sums)))
    return np.percentile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [2.5, 97.5])


def units(d):
    t = d.groupby(['Source', 'Task', 'Noise', 'Method']).ARI.mean().unstack()
    best = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    gains = pd.DataFrame({m: t[m] - best for m in SUPERVISED})
    gains['Alignment'] = d.groupby(['Source', 'Task', 'Noise']).Alignment_AMI.mean()
    return gains.reset_index()


def main():
    d = pd.read_csv(os.path.join(RES, 'alignment.csv'))
    g = units(d)
    g['Cohort'] = g.Source + '/' + g.Task
    rows = []
    for scope, part in [('Pooled', g), ('TCGA', g[g.Source == 'TCGA']), ('GTEx', g[g.Source == 'GTEx'])]:
        for m in SUPERVISED:
            wide = part.pivot_table(index='Cohort', columns='Noise', values=m)
            x = wide[0.0] - wide[0.5]
            lo, hi = cluster_bootstrap(x, x.index)
            rows.append({'Scope': scope, 'Question': 'A1', 'Method': m, 'Noise': np.nan, 'Units': len(x),
                         'Estimate': x.mean(), 'Lower': lo, 'Upper': hi})
            for q in sorted(part.Noise.unique()):
                v = wide[q]
                lo, hi = cluster_bootstrap(v, v.index)
                rows.append({'Scope': scope, 'Question': 'A2', 'Method': m, 'Noise': q, 'Units': len(v),
                             'Estimate': v.mean(), 'Lower': lo, 'Upper': hi})
            rho = spearmanr(part.Alignment, part[m]).statistic
            cohorts = part.Cohort.unique()
            rng = np.random.RandomState(0)
            draws = []
            for _ in range(2000):
                pick = rng.choice(cohorts, len(cohorts))
                sample = pd.concat([part[part.Cohort == c] for c in pick])
                draws.append(spearmanr(sample.Alignment, sample[m]).statistic)
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])
            rows.append({'Scope': scope, 'Question': 'A3', 'Method': m, 'Noise': np.nan, 'Units': len(part),
                         'Estimate': rho, 'Lower': lo, 'Upper': hi})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RES, 'alignment_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print('Mean alignment (AMI) by noise level:', g.groupby('Noise').Alignment.mean().round(3).to_dict())
    print(s.round(3).to_string(index=False))
    print('\nVerdicts (pooled):')
    for m in SUPERVISED:
        a1 = s[(s.Scope == 'Pooled') & (s.Question == 'A1') & (s.Method == m)].iloc[0]
        a2 = s[(s.Scope == 'Pooled') & (s.Question == 'A2') & (s.Method == m)]
        at0 = a2[a2.Noise == 0.0].iloc[0]
        above = a2[a2.Lower > 0].Noise
        largest = above.max() if len(above) else None
        verdict = 'supported' if a1.Lower > 0 and at0.Lower > 0 else 'contradicted' if a1.Upper <= 0 else 'partly supported' if a1.Lower > 0 else 'inconclusive'
        print(f'  {m}: A1 {a1.Estimate:.3f} [{a1.Lower:.3f}, {a1.Upper:.3f}], gain at q=0 {at0.Estimate:.3f} '
              f'[{at0.Lower:.3f}, {at0.Upper:.3f}], largest q with gain above zero: {largest}; {verdict}')


if __name__ == '__main__':
    main()
