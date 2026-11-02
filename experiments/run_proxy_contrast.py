"""Does a meaningful coarse label help more than a random grouping of the same classes?

Analysis plan, fixed before running:
  * Outcome: gain = supervised ARI minus unsupervised ARI on the same target cohort,
    against both raw and scaled K-Means.
  * Real proxy: the benchmark's own grouping. Random proxies: up to three other groupings
    with the same number of background classes and at least two target classes.
  * Primary contrast, per dataset: mean gain under the real proxy minus mean gain under
    random proxies, for BTC and for the RF leaf embedding.
  * Primary datasets have a domain-motivated proxy: Arrhythmia, CTG, HAR, Leukemia, Mice,
    Paul15, Thyroid. Dermatology, Segment and Vehicle use an arbitrary first class, so
    they are reported separately. BRCA, GBMLGG and Diabetes lack a usable class grouping.
  * With seven primary datasets the result is descriptive, and no significance claim is made.
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import load_dataset

N_SEEDS, N_RANDOM = 10, 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'proxy_contrast.csv')
PRIMARY = ['arrhythmia', 'ctg', 'har', 'leukemia', 'mice', 'paul15', 'thyroid']
ARBITRARY = ['dermatology', 'segment', 'vehicle']


def class_labels(name):
    """Original class of every row, aligned with X_full from load_dataset."""
    if name in ('dermatology', 'vehicle', 'segment'):
        return fetch_openml(name=name, version=1, as_frame=True, parser='auto').target.astype(str).values
    if name == 'thyroid':
        return fetch_openml(data_id=57, as_frame=True, parser='auto').target.astype(str).values
    if name == 'arrhythmia':
        return fetch_openml('arrhythmia', version=1, parser='auto', as_frame=True).target.astype(str).values
    if name == 'leukemia':
        return fetch_openml(data_id=45092, parser='auto', as_frame=True).target.astype(str).values
    if name == 'har':
        return fetch_openml('har', version=1, parser='auto').target.astype(str).values
    if name == 'mice':
        return fetch_openml('miceprotein', version=4, parser='auto').target.astype(str).values
    if name == 'ctg':
        return fetch_openml(data_id=1560, as_frame=True, parser='auto').target.astype(str).values
    if name == 'paul15':
        import scanpy as sc
        return sc.datasets.paul15().obs['paul15_clusters'].astype(str).values
    raise ValueError(name)


def groupings(name, labels, y_real):
    """The real grouping plus random groupings with the same number of background classes."""
    classes = np.unique(labels)
    background = tuple(sorted(np.unique(labels[y_real == 0])))
    options = [c for c in itertools.combinations(classes, len(background))
               if tuple(sorted(c)) != background and len(classes) - len(c) >= 2]
    rng = np.random.RandomState(sum(map(ord, name)))
    picks = rng.permutation(len(options))[:N_RANDOM]
    yield 'real', 0, background
    for i, p in enumerate(picks):
        yield 'random', i + 1, tuple(options[p])


def run_seed(seed, name, kind, grouping, background, X_full, labels):
    y = (~np.isin(labels, background)).astype(int)
    X_target = X_full[y == 1]
    target_labels = labels[y == 1]
    z = np.searchsorted(np.unique(target_labels), target_labels)
    k = len(np.unique(z))

    def score(Z):
        return adjusted_rand_score(z, KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z))

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_full, y)
    margins = extract_staged_decision_function(booster, X_target)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X_target), margins]), axis=1)
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_full, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    components = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    features = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thresholds = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    fraction = (X_target[:, features] <= thresholds).mean(axis=0)
    return {
        'Dataset': name, 'Proxy': kind, 'grouping': grouping, 'seed': seed, 'K': k, 'N_target': len(X_target),
        'BTC': score(updates),
        'RF leaves with SVD': score(TruncatedSVD(n_components=components, random_state=seed).fit_transform(leaves)),
        'K-Means raw': score(X_target),
        'K-Means scaled': score(StandardScaler().fit_transform(X_target)),
        'Mean_homogeneity': np.maximum(fraction, 1 - fraction).mean(),
    }


def main():
    rows = []
    for name in PRIMARY + ARBITRARY:
        X_full, y_real, _, _, _ = load_dataset(name)
        labels = class_labels(name)
        assert len(labels) == len(X_full), name
        for kind, grouping, background in groupings(name, labels, y_real):
            if kind == 'real':
                assert np.array_equal((~np.isin(labels, background)).astype(int), y_real), name
            rows.extend(Parallel(n_jobs=-1)(
                delayed(run_seed)(seed, name, kind, grouping, background, X_full, labels) for seed in range(N_SEEDS)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(name, kind, grouping, background[:4], flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
