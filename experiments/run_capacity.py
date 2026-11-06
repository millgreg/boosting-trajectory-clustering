"""Exploratory: do larger boosted models change the TCGA and GTEx results?

Analysis plan, fixed before any result of this study existed. Added after the main studies because
the default BTC and TreeSHAP models use 100 depth-1 trees and can therefore use at most 100 of the
2,000 genes.

Tasks: the cohort-versus-rest designs, namely the 20 TCGA molecular-subtype tasks (Level B of
run_tcga_hierarchy.py) and the 7 GTEx tissue-versus-rest tasks (run_gtex.py), 3 seeds.

Methods: BTC with the default 100 stumps and with 1,000 stumps (learning rate 0.1, square-root
feature subsampling); standardized TreeSHAP with the default (100 depth-1 trees) and with 300
depth-3 trees (learning rate 0.1, column subsampling 0.8); K-Means on raw and standardized features.

Questions, with 95% bootstrap intervals resampling tasks (20,000 resamples), per data source:
  M1. Gain of each larger model over the better of raw and standardized K-Means.
  M2. Larger model minus its default.
The conclusion that supervised representations fall below K-Means is considered robust to model
size if the M1 interval lies below zero for both larger models in both data sources.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments import run_gtex, run_tcga_hierarchy

OUT = os.path.join(ROOT, 'experiments', 'results', 'capacity.csv')
N_SEEDS = 3


def tasks():
    frame = run_tcga_hierarchy.load()
    X = frame.drop(columns=['cancer_type', 'subtype']).values.astype(np.float64)
    cancer = frame.cancer_type.values.astype(str)
    text = frame.subtype.fillna('').astype(str)
    for prefix in sorted(frame.subtype.dropna().str.split('.').str[0].unique()):
        members = run_tcga_hierarchy.SUBTYPE_TYPES.get(prefix, [prefix])
        y = np.isin(cancer, members).astype(int)
        target = (y == 1) & text.str.startswith(prefix + '.').values & ~(text.str.endswith('.NA') | text.str.endswith('.-')).values
        idx = np.where(target)[0]
        labels = frame.subtype.values[idx].astype(str)
        counts = pd.Series(labels).value_counts()
        keep = np.isin(labels, counts[counts >= run_tcga_hierarchy.MIN_SUBTYPE].index)
        yield 'TCGA', prefix, X, y, idx[keep], labels[keep]
    frame = run_gtex.load()
    X = frame.drop(columns=['tissue', 'site']).values.astype(np.float64)
    for name in run_gtex.tissues(frame):
        y = (frame.tissue.values == name).astype(int)
        idx = np.where(y == 1)[0]
        yield 'GTEx', name, X, y, idx, frame.site.values[idx].astype(str)


def run(source, name, X, y, idx, labels, seed):
    import xgboost as xgb
    X_target = X[idx]
    z = np.unique(labels, return_inverse=True)[1]
    k = int(z.max()) + 1

    def km(Z):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    def btc(n_trees):
        booster = GradientBoostingClassifier(n_estimators=n_trees, max_depth=1, max_features='sqrt', random_state=seed).fit(X, y)
        return km(np.diff(np.column_stack([extract_raw_init(booster, X_target), extract_staged_decision_function(booster, X_target)]), axis=1))

    def shap(n_trees, depth):
        model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=0.1, colsample_bytree=0.8, n_jobs=1,
                                  random_state=seed, verbosity=0).fit(X, y)
        values = model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]
        return km(StandardScaler().fit_transform(values))

    scores = {
        'BTC, 100 stumps': btc(100),
        'BTC, 1000 stumps': btc(1000),
        'TreeSHAP, 100 depth-1 trees': shap(100, 1),
        'TreeSHAP, 300 depth-3 trees': shap(300, 3),
        'K-Means raw': km(X_target),
        'K-Means scaled': km(StandardScaler().fit_transform(X_target)),
    }
    base = {'Source': source, 'Task': name, 'seed': seed, 'K': k, 'N_target': len(idx)}
    return [{**base, 'Method': m, 'ARI': adjusted_rand_score(z, l)} for m, l in scores.items()]


def main():
    rows = []
    for source, name, X, y, idx, labels in tasks():
        for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(run)(source, name, X, y, idx, labels, s) for s in range(N_SEEDS)):
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(source, name, flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
