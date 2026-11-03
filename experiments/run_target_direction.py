"""Exploratory: what remains of boosting histories after the target direction is removed?

Runs on the 13 development benchmarks with 10 seeds. Not part of any pre-specified plan.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score as ari
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
from experiments.dataset_loader import ALL_DATASETS, load_dataset
from btc.core import extract_raw_init, extract_staged_decision_function

def km(Z, k, s):
    return KMeans(k, n_init=10, random_state=s).fit_predict(Z)

def remove_pc1(M):
    C = M - M.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    return C - np.outer(C @ vt[0], vt[0])

def regress_out(M, v):
    C = M - M.mean(axis=0); x = v - v.mean()
    return C - np.outer(x, (x @ C) / (x @ x))

def task(s, name, Xf, y, Xt, z, k):
    g = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=s).fit(Xf, y)
    F = extract_staged_decision_function(g, Xt)
    F0 = extract_raw_init(g, Xt)
    D = np.diff(np.column_stack([F0, F]), axis=1)
    Fprev = np.column_stack([F0, F])[:, :-1]
    R = 1.0 - 1.0 / (1.0 + np.exp(-Fprev))
    final = F[:, -1]
    reps = {
        'BTC tree updates': D,
        'Final margin only': final[:, None],
        'Cumulative margins': F,
        'Cumulative margins, final margin regressed out': regress_out(F, final),
        'Cumulative margins, first component removed': remove_pc1(F),
        'Residual history': R,
        'Residual history, final margin regressed out': regress_out(R, final),
        'Residual history, first component removed': remove_pc1(R),
        'BTC tree updates, final margin regressed out': regress_out(D, final),
    }
    return [{'Dataset': name, 'seed': s, 'Method': m, 'ARI': ari(z, km(Z, k, s))} for m, Z in reps.items()]

rows = []
for n in ALL_DATASETS:
    data = load_dataset(n)
    for r in Parallel(n_jobs=3)(delayed(task)(s, n, *data) for s in range(10)):
        rows.extend(r)
    print(n, flush=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "target_direction.csv")
d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
t = d.groupby(['Dataset', 'Method']).ARI.mean().unstack()
def boot(x, n=20000):
    x = np.asarray(x); r = np.random.RandomState(0)
    return np.percentile(x[r.randint(0, len(x), (n, len(x)))].mean(axis=1), [2.5, 97.5]).round(3).tolist()
pd.set_option('display.width', 220)
print(t.T.round(3).to_string())
print()
for m in t.mean().sort_values(ascending=False).index:
    x = t[m] - t['BTC tree updates']
    print(f'{m:48s} mean {t[m].mean():.3f}  minus BTC {x.mean():+.3f} {boot(x) if m != "BTC tree updates" else ""}')
