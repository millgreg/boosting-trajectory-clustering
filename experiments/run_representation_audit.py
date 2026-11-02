"""Measure the geometry of the delta and cumulative trajectory representations.

For each dataset and seed, reports the variance captured by the first principal
component and two effective-dimensionality measures: the participation ratio
(squared sum of eigenvalues over sum of squared eigenvalues) and the number of
components required to reach 90% of the variance.
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import ALL_DATASETS, load_dataset

N_SEEDS = 30
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'representation_audit.csv')


def geometry(R):
    centered = R - R.mean(axis=0)
    eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    ratio = eigenvalues / total
    return ratio[0], total ** 2 / (eigenvalues ** 2).sum(), int(np.argmax(np.cumsum(ratio) >= 0.90) + 1)


def run_seed(seed, name, X_full, y_binary, X_target):
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed)
    gb.fit(X_full, y_binary)
    cumulative = extract_staged_decision_function(gb, X_target)
    baseline = extract_raw_init(gb, X_target)
    delta = np.diff(np.column_stack([baseline, cumulative]), axis=1)
    rows = []
    for representation, R in (('per_tree_delta', delta), ('cumul_logodds', cumulative)):
        pc1, participation, n90 = geometry(R)
        rows.append({'Dataset': name, 'seed': seed, 'Representation': representation, 'PC1_variance': pc1,
                     'Participation_ratio': participation, 'Components_90pct': n90})
    return rows


def main():
    rows = []
    for name in ALL_DATASETS:
        X_full, y_binary, X_target, _, _ = load_dataset(name)
        results = Parallel(n_jobs=-1)(
            delayed(run_seed)(seed, name, X_full, y_binary, X_target) for seed in range(N_SEEDS))
        for seed_rows in results:
            rows.extend(seed_rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    summary = df.groupby(['Representation', 'Dataset'])[['PC1_variance', 'Participation_ratio', 'Components_90pct']].mean()
    print(summary.round(3).to_string())
    print(summary.groupby('Representation').agg(['min', 'max']).round(3).to_string())


if __name__ == '__main__':
    main()
