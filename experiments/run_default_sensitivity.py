"""Development-only sensitivity of the BTC default, one setting changed at a time.

Runs on the 13 benchmarks plus OpenML-CC18 multi-class datasets, which are never
used for confirmation. CC18 proxies group the classes at random, and those target
classes act as subtypes.
"""
import os
import sys
import warnings

import numpy as np
import openml
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import ALL_DATASETS, clean_features, load_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'default_sensitivity.csv')
N_PROXIES, N_SEEDS, MAX_ROWS, MAX_FEATURES, MIN_CLASS = 3, 5, 20000, 1000, 5
OVERLAP = {'har', 'vehicle', 'segment', 'miceprotein', 'mice'}
BASE = dict(trees=100, learning_rate=0.1, depth=1, max_features='sqrt', subsample=1.0, scale=False)
CONFIGS = {
    'Default': {},
    'Trees 50': dict(trees=50),
    'Trees 300': dict(trees=300),
    'Learning rate 0.05': dict(learning_rate=0.05),
    'Learning rate 0.3': dict(learning_rate=0.3),
    'Depth 2': dict(depth=2),
    'Depth 3': dict(depth=3),
    'Features all': dict(max_features=None),
    'Features 10%': dict(max_features=0.1),
    'Row subsample 0.5': dict(subsample=0.5),
    'Rescaled per tree': dict(scale=True),
}


def kmeans(Z, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def updates(booster, X):
    margins = extract_staged_decision_function(booster, X)
    return np.diff(np.column_stack([extract_raw_init(booster, X), margins]), axis=1)


def run_task(name, proxy, seed, X_full, y, X_target, z, k):
    rows = []
    for label, change in CONFIGS.items():
        c = {**BASE, **change}
        booster = GradientBoostingClassifier(
            n_estimators=c['trees'], learning_rate=c['learning_rate'], max_depth=c['depth'],
            max_features=c['max_features'], subsample=c['subsample'], random_state=seed).fit(X_full, y)
        R = updates(booster, X_target)
        if c['scale']:
            R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-8)
        rows.append((label, adjusted_rand_score(z, kmeans(R, k, seed))))
        if label == 'Default':
            rows.append(('Agglomerative backend', adjusted_rand_score(z, AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(R)) if len(R) <= 10000 else np.nan))
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_full, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    components = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    rows.append(('RF leaves with SVD', adjusted_rand_score(z, kmeans(TruncatedSVD(n_components=components, random_state=seed).fit_transform(leaves), k, seed))))
    rows.append(('Raw K-Means', adjusted_rand_score(z, kmeans(X_target, k, seed))))
    return [{'Dataset': name, 'proxy': proxy, 'seed': seed, 'Method': m, 'ARI': a, 'N_target': len(X_target), 'K': k} for m, a in rows]


def cc18_tasks():
    suite = openml.study.get_suite(99)
    meta = openml.datasets.list_datasets(data_id=suite.data, output_format='dataframe')
    meta = meta[(meta.NumberOfClasses >= 3) & (meta.NumberOfInstances <= MAX_ROWS) & (meta.NumberOfFeatures <= MAX_FEATURES)]
    meta = meta[~meta.name.str.lower().isin(OVERLAP)]
    for _, m in meta.iterrows():
        ds = openml.datasets.get_dataset(int(m.did), download_qualities=False, download_features_meta_data=False)
        X, y_raw, _, _ = ds.get_data(target=ds.default_target_attribute)
        labels = pd.Series(y_raw).astype(str).values
        counts = pd.Series(labels).value_counts()
        usable = counts[counts >= MIN_CLASS].index.values
        mask = np.isin(labels, usable)
        X_all = clean_features(X.loc[mask].reset_index(drop=True))
        labels = labels[mask]
        rng = np.random.RandomState(int(m.did))
        for proxy in range(N_PROXIES):
            classes = rng.permutation(np.sort(usable))
            n_background = rng.randint(1, len(classes) - 1)
            target = classes[n_background:]
            y = np.isin(labels, target).astype(int)
            target_labels = labels[y == 1]
            z = np.searchsorted(np.sort(target), target_labels)
            yield m['name'], proxy, X_all, y, X_all[y == 1], z, len(target)


def main():
    rows = []
    for name in ALL_DATASETS:
        X_full, y, X_target, z, k = load_dataset(name)
        for result in Parallel(n_jobs=-1)(delayed(run_task)(name, 0, s, X_full, y, X_target, z, k) for s in range(N_SEEDS)):
            rows.extend(result)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(name, flush=True)
    for name, proxy, X_full, y, X_target, z, k in cc18_tasks():
        for result in Parallel(n_jobs=-1)(delayed(run_task)('cc18_' + name, proxy, s, X_full, y, X_target, z, k) for s in range(N_SEEDS)):
            rows.extend(result)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(name, proxy, flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
