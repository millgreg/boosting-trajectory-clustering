"""Exploratory addition to the confirmation run: spectral Random Forest Clustering.

Not part of the frozen protocol, so results are reported as exploratory. It uses the same
datasets, groupings, seeds and forests as experiments/run_confirmation.py, and computes the
exact spectral RFC embedding without forming the N x N proximity matrix.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baselines.rfc_factorized import FactorizedRFC
from experiments.run_confirmation import LIST, N_SEEDS, groupings, load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'confirmation_rfc_exploratory.csv')


def run(did, name, grouping, background, seed, X_all, labels):
    y = (~np.isin(labels, background)).astype(int)
    X_target, target_labels = X_all[y == 1], labels[y == 1]
    z = np.searchsorted(np.unique(target_labels), target_labels)
    started = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_all, y)
    labels_ = FactorizedRFC(rf_model=forest, n_clusters=len(np.unique(z)), random_state=seed).fit_predict(X_target)
    return {'did': did, 'Dataset': name, 'grouping': grouping, 'seed': seed, 'Method': 'RFC spectral',
            'ARI': adjusted_rand_score(z, labels_), 'Time_s': time.perf_counter() - started}


def main():
    rows = []
    for _, d in pd.read_csv(LIST).iterrows():
        try:
            X_all, labels = load(d.did)
            tasks = [(int(d.did), d['name'], g, b, s, X_all, labels) for g, b in enumerate(groupings(d.did, labels)) for s in range(N_SEEDS)]
            rows.extend(Parallel(n_jobs=-1)(delayed(run)(*t) for t in tasks))
            status = 'ok'
        except Exception as e:
            status = 'FAILED ' + str(e)[:120]
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(d['name'], status, flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
