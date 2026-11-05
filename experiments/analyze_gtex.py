"""Pre-specified analysis of the GTEx study, written before any expression data were downloaded.

Implements the plan in experiments/run_gtex.py. Gains are measured against the better of raw and
standardized K-Means on the same cohort. Intervals are 95% bootstrap intervals with 20,000
resamples over tissues (G1, G2, G3) or tissue-proxy pairs (alignment correlation).
"""
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


def verdict(lo, hi, question):
    if question == 'G2':
        return 'contradicts explanation' if lo > 0 else 'consistent'
    return 'replicated' if hi < 0 else 'contradicted' if lo > 0 else 'inconclusive'


def main():
    d = pd.read_csv(os.path.join(RES, 'gtex.csv'))
    t = d.groupby(['Design', 'Task', 'Proxy', 'Method']).ARI.mean().unstack()
    t['Best K-Means'] = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    for m in SUPERVISED:
        t[f'{m} gain'] = t[m] - t['Best K-Means']
    pd.set_option('display.width', 250)
    print(t[SUPERVISED + ['K-Means raw', 'K-Means scaled']].round(3).to_string())

    rows = []
    tissue = t.loc['tissue'].droplevel('Proxy')
    within = t.loc['within'].reset_index()
    for m in SUPERVISED:
        for question, frame, groups in [('G1', tissue, tissue.index), ('G2', within, within.Task)]:
            x = frame[f'{m} gain']
            lo, hi = cluster_bootstrap(x, groups)
            rows.append({'Question': question, 'Comparison': f'{m} gain over better K-Means', 'Units': len(x), 'Difference': x.mean(),
                         'Lower': lo, 'Upper': hi, 'Positive in': int((x > 5e-4).sum()), 'Verdict': verdict(lo, hi, question)})
    for m in SUPERVISED[1:]:
        x = tissue['BTC'] - tissue[m]
        lo, hi = cluster_bootstrap(x, tissue.index)
        rows.append({'Question': 'G1', 'Comparison': f'BTC minus {m}', 'Units': len(x), 'Difference': x.mean(), 'Lower': lo, 'Upper': hi,
                     'Positive in': int((x > 5e-4).sum()), 'Verdict': 'difference' if lo > 0 or hi < 0 else 'no claim'})
    for m in SUPERVISED:
        x = (within.groupby('Task')[f'{m} gain'].mean() - tissue[f'{m} gain']).dropna()
        lo, hi = cluster_bootstrap(x, x.index)
        rows.append({'Question': 'G3', 'Comparison': f'{m}: within-tissue minus tissue-versus-rest gain', 'Units': len(x),
                     'Difference': x.mean(), 'Lower': lo, 'Upper': hi, 'Positive in': int((x > 5e-4).sum()),
                     'Verdict': 'difference' if lo > 0 or hi < 0 else 'no claim'})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RES, 'gtex_summary.csv'), index=False)
    print('\nPre-specified comparisons')
    print(summary.round(3).to_string(index=False))

    btc = d[d.Method == 'BTC'].groupby(['Design', 'Task', 'Proxy'])[['Mean_homogeneity', 'Proxy_site_AMI']].mean()
    print('\nMean leaf homogeneity by design')
    print(btc.groupby('Design').Mean_homogeneity.agg(['mean', 'min', 'max']).round(3).to_string())
    w = btc.loc['within'].join(within.set_index(['Task', 'Proxy'])['BTC gain'])
    rho = spearmanr(w.Proxy_site_AMI, w['BTC gain']).statistic
    rng = np.random.RandomState(0)
    draws = [spearmanr(w.Proxy_site_AMI.values[i], w['BTC gain'].values[i]).statistic
             for i in (rng.randint(0, len(w), len(w)) for _ in range(20000))]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    print(f'\nWithin-tissue alignment: Spearman correlation of proxy-site AMI with BTC gain {rho:.3f} [{lo:.3f}, {hi:.3f}] over {len(w)} pairs')
    print(w.round(3).to_string())


if __name__ == '__main__':
    main()
