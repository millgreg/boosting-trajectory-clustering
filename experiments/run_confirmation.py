"""Confirmation run on datasets never used for development. See experiments/confirmation/PROTOCOL.md.

Each dataset contributes three random class groupings with five seeds each. Every method
receives the true K. Results are written incrementally so an interrupted run can be resumed.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import openml
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import clean_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIST = os.path.join(ROOT, 'experiments', 'confirmation', 'confirmation_datasets.csv')
OUT = os.path.join(ROOT, 'experiments', 'results', 'confirmation_results.csv')
N_GROUPINGS, N_SEEDS, MAX_ROWS, MIN_CLASS, SEED = 3, 5, 20000, 5, 20261103
# Frozen from development Part A; set before the protocol is frozen.
SHAP_DEPTH, SHAP_COLSAMPLE, SHAP_CLUSTERING = 1, 0.8, 'kmeans_scaled'


def load(did):
    ds = openml.datasets.get_dataset(int(did), download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True).loc[y.notna().values].reset_index(drop=True)
    y = y.dropna().astype(str).reset_index(drop=True)
    if len(X) > MAX_ROWS:
        keep = np.sort(np.random.RandomState(SEED).permutation(len(X))[:MAX_ROWS])
        X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    counts = y.value_counts()
    mask = y.isin(counts[counts >= MIN_CLASS].index).values
    return clean_features(X.loc[mask].reset_index(drop=True)), y.values[mask]


def groupings(did, labels):
    classes = np.sort(np.unique(labels))
    rng = np.random.RandomState(SEED + int(did))
    out = []
    while len(out) < N_GROUPINGS:
        order = rng.permutation(classes)
        n_background = rng.randint(1, len(classes) - 1)
        background = tuple(sorted(order[:n_background]))
        if background not in out:
            out.append(background)
    return out


def kmeans(Z, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def pseudo_residuals(baseline, updates, y):
    margins = np.column_stack([baseline, baseline[:, None] + np.cumsum(updates, axis=1)])[:, :-1]
    return y - 1.0 / (1.0 + np.exp(-margins))


def run(did, name, grouping, background, seed, X_all, labels):
    import umap
    import xgboost as xgb
    y = (~np.isin(labels, background)).astype(int)
    X_target, target_labels = X_all[y == 1], labels[y == 1]
    z = np.searchsorted(np.unique(target_labels), target_labels)
    k = len(np.unique(z))
    rng = np.random.RandomState(seed)
    scores, times = {}, {}

    def score(method, labels_, started=None):
        scores[method] = adjusted_rand_score(z, labels_)
        times[method] = np.nan if started is None else time.perf_counter() - started

    started = time.perf_counter()
    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_all, y)
    margins = extract_staged_decision_function(booster, X_target)
    baseline = extract_raw_init(booster, X_target)
    updates = np.diff(np.column_stack([baseline, margins]), axis=1)
    score('BTC', kmeans(updates, k, seed), started)
    score('Final margin', kmeans(margins[:, -1:], k, seed))
    residuals = pseudo_residuals(baseline, updates, 1.0)
    score('Historic residuals', kmeans(residuals, k, seed))
    score('Historic residuals, order shuffled', kmeans(pseudo_residuals(baseline, updates[:, rng.permutation(100)], 1.0), k, seed))
    features = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thresholds = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    goes_left = X_target[:, features] <= thresholds
    score('Per-feature sums', kmeans(np.column_stack([updates[:, features == f].sum(axis=1) for f in np.unique(features)]), k, seed))

    started = time.perf_counter()
    model = xgb.XGBClassifier(n_estimators=100, max_depth=SHAP_DEPTH, learning_rate=0.1, colsample_bytree=SHAP_COLSAMPLE,
                              n_jobs=1, random_state=seed, verbosity=0).fit(X_all, y)
    S = model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]
    if SHAP_CLUSTERING == 'kmeans_scaled':
        S = StandardScaler().fit_transform(S)
    score('TreeSHAP', kmeans(S, k, seed), started)

    started = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_all, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    components = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    score('RF leaves with SVD', kmeans(TruncatedSVD(n_components=components, random_state=seed).fit_transform(leaves), k, seed), started)

    score('K-Means raw', kmeans(X_target, k, seed))
    scaled = StandardScaler().fit_transform(X_target)
    score('K-Means scaled', kmeans(scaled, k, seed))
    if len(X_target) <= 20000:
        n = max(1, min(10, X_target.shape[1] - 1, X_target.shape[0] - 2))
        score('UMAP raw', kmeans(umap.UMAP(n_components=n, random_state=seed, n_jobs=1).fit_transform(X_target), k, seed))
        score('UMAP scaled', kmeans(umap.UMAP(n_components=n, random_state=seed, n_jobs=1).fit_transform(scaled), k, seed))

    fraction = goes_left.mean(axis=0)
    homogeneity = np.maximum(fraction, 1 - fraction)
    common = {'did': did, 'Dataset': name, 'grouping': grouping, 'seed': seed, 'K': k, 'N_full': len(X_all),
              'N_target': len(X_target), 'N_features': X_all.shape[1],
              'Proxy_AUC': roc_auc_score(y, booster.predict_proba(X_all)[:, 1]),
              'Mean_homogeneity': homogeneity.mean(), 'Trees_gt_95': int((homogeneity > 0.95).sum())}
    return [{**common, 'Method': m, 'ARI': a, 'Time_s': times[m]} for m, a in scores.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Run only the first datasets, for testing')
    parser.add_argument('--list', default=LIST, help='Dataset list; use a development list for testing')
    args = parser.parse_args()
    datasets = pd.read_csv(args.list)
    if args.limit:
        datasets = datasets.head(args.limit)
    done = pd.read_csv(OUT) if os.path.exists(OUT) and not args.limit else pd.DataFrame(columns=['did'])
    rows = done.to_dict('records')
    for _, d in datasets.iterrows():
        if int(d.did) in set(done.did.astype(int)):
            continue
        try:
            X_all, labels = load(d.did)
            tasks = [(int(d.did), d['name'], g, b, s, X_all, labels) for g, b in enumerate(groupings(d.did, labels)) for s in range(N_SEEDS)]
            for r in Parallel(n_jobs=-1)(delayed(run)(*t) for t in tasks):
                rows.extend(r)
            status = 'ok'
        except Exception as e:
            status = 'FAILED ' + str(e)[:120]
            rows.append({'did': int(d.did), 'Dataset': d['name'], 'Method': 'FAILED', 'ARI': np.nan, 'error': str(e)[:200]})
        if not args.limit:
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(d['name'], status, flush=True)
    print('ALL_DONE', flush=True)
    return rows


if __name__ == '__main__':
    main()
