"""Pre-specified analysis of the model-size study (experiments/run_capacity.py), written before any result existed."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'experiments', 'results')
LARGER = {'BTC, 1000 stumps': 'BTC, 100 stumps', 'TreeSHAP, 300 depth-3 trees': 'TreeSHAP, 100 depth-1 trees'}


def bootstrap(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    return np.percentile(x[np.random.RandomState(seed).randint(0, len(x), (n, len(x)))].mean(axis=1), [2.5, 97.5])


def main():
    d = pd.read_csv(os.path.join(RES, 'capacity.csv'))
    t = d.groupby(['Source', 'Task', 'Method']).ARI.mean().unstack()
    t['Best K-Means'] = t[['K-Means raw', 'K-Means scaled']].max(axis=1)
    pd.set_option('display.width', 200)
    print(t.round(3).to_string())
    rows = []
    for source in ['TCGA', 'GTEx']:
        part = t.loc[source]
        for large, default in LARGER.items():
            for question, x in [('M1', part[large] - part['Best K-Means']), ('M2', part[large] - part[default])]:
                lo, hi = bootstrap(x)
                rows.append({'Source': source, 'Question': question, 'Method': large,
                             'Comparison': 'gain over better K-Means' if question == 'M1' else f'minus {default}',
                             'Units': len(x), 'Estimate': x.mean(), 'Lower': lo, 'Upper': hi, 'Positive in': int((x > 5e-4).sum())})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RES, 'capacity_summary.csv'), index=False)
    print(s.round(3).to_string(index=False))
    m1 = s[s.Question == 'M1']
    print('\nRobust to model size:', bool((m1.Upper < 0).all()))


if __name__ == '__main__':
    main()
