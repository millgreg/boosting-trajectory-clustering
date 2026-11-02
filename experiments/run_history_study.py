"""What in a boosting history reveals subtypes? Representation comparison on a set of datasets.

Every representation is clustered with K-Means and the true K unless named otherwise,
so differences reflect the representation. Proxy diagnostics are stored with each run.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baselines.rfc import RFC
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import ALL_DATASETS, load_dataset

N_TREES = 100
N_RESTARTS = 10
SVD_COMPONENTS = 50
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def kmeans(Z, k, seed):
    return KMeans(n_clusters=k, n_init=N_RESTARTS, random_state=seed).fit_predict(Z)


def target_indices(X_full, X_target):
    positions = {}
    for i, row in enumerate(X_full):
        positions.setdefault(row.tobytes(), []).append(i)
    return np.array([positions[row.tobytes()].pop(0) for row in X_target])


def fit_stumps(X, y, seed):
    return GradientBoostingClassifier(n_estimators=N_TREES, max_depth=1, max_features='sqrt', random_state=seed).fit(X, y)


def tree_updates(booster, X):
    margins = extract_staged_decision_function(booster, X)
    baseline = extract_raw_init(booster, X)
    return baseline, margins, np.diff(np.column_stack([baseline, margins]), axis=1)


def pseudo_residuals(baseline, updates, y):
    """y - p_(t-1), recomputed from the updates in the order supplied."""
    margins = np.column_stack([baseline, baseline[:, None] + np.cumsum(updates, axis=1)])[:, :-1]
    return y[:, None] - 1.0 / (1.0 + np.exp(-margins))


def learning_speed(residuals):
    size = np.abs(residuals)
    halved = size <= size[:, :1] / 2
    half_time = np.where(halved.any(axis=1), halved.argmax(axis=1), residuals.shape[1]) / residuals.shape[1]
    features = np.column_stack([half_time, size.mean(axis=1), size[:, -1]])
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)


def tree_shap(X_full, y, X_target, seed):
    """Exact TreeSHAP from a depth-3 XGBoost model, the configuration common in applied work."""
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=N_TREES, max_depth=3, learning_rate=0.1, colsample_bytree=0.8,
                              n_jobs=1, random_state=seed, verbosity=0).fit(X_full, y)
    return model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]


def run_seed(seed, name, X_full, y, X_target, subtypes, k, idx):
    rows = []
    rng = np.random.RandomState(seed)

    def record(method, labels, started=None):
        rows.append({'Dataset': name, 'seed': seed, 'Method': method,
                     'ARI': adjusted_rand_score(subtypes, labels),
                     'NMI': normalized_mutual_info_score(subtypes, labels),
                     'Time_s': np.nan if started is None else time.perf_counter() - started})

    # Boosting representations
    started = time.perf_counter()
    booster = fit_stumps(X_full, y, seed)
    baseline, margins, updates = tree_updates(booster, X_target)
    record('BTC tree updates', kmeans(updates, k, seed), started)
    record('BTC tree updates, order shuffled', kmeans(updates[:, rng.permutation(N_TREES)], k, seed))
    record('Final margin', kmeans(margins[:, -1:], k, seed))

    y_target = y[idx].astype(float)
    residuals = pseudo_residuals(baseline, updates, y_target)
    shuffled = pseudo_residuals(baseline, updates[:, rng.permutation(N_TREES)], y_target)
    record('Historic residuals', kmeans(residuals, k, seed))
    record('Historic residuals, order shuffled', kmeans(shuffled, k, seed))
    record('Residual change', kmeans(np.diff(residuals, axis=1), k, seed))
    record('Residual change, order shuffled', kmeans(np.diff(shuffled, axis=1), k, seed))
    record('Learning-speed summaries', kmeans(learning_speed(residuals), k, seed))

    features = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thresholds = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    goes_left = X_target[:, features] <= thresholds
    record('Stump leaves, unweighted', kmeans(goes_left.astype(float), k, seed))
    per_feature = np.column_stack([updates[:, features == f].sum(axis=1) for f in np.unique(features)])
    record('Per-feature sums, SHAP-equivalent', kmeans(per_feature, k, seed))

    permuted = fit_stumps(X_full, rng.permutation(y), seed)
    record('BTC, permuted proxy labels', kmeans(tree_updates(permuted, X_target)[2], k, seed))

    record('TreeSHAP, XGBoost depth 3', kmeans(tree_shap(X_full, y, X_target, seed), k, seed))

    # Random forest representations
    started = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=1).fit(X_full, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    embed_time = time.perf_counter() - started
    started = time.perf_counter()
    labels = kmeans(leaves, k, seed)
    record('RF leaves', labels, started)
    rows[-1]['Time_s'] += embed_time
    started = time.perf_counter()
    components = min(SVD_COMPONENTS, leaves.shape[0] - 1, leaves.shape[1] - 1)
    labels = kmeans(TruncatedSVD(n_components=components, random_state=seed).fit_transform(leaves), k, seed)
    record('RF leaves with SVD', labels, started)
    rows[-1]['Time_s'] += embed_time
    started = time.perf_counter()
    labels = RFC(rf_model=forest, n_clusters=k, clustering_algo='spectral', random_state=seed).fit_predict(X_target)
    record('RFC spectral', labels, started)
    rows[-1]['Time_s'] += embed_time

    # Unsupervised baselines
    started = time.perf_counter()
    record('Raw K-Means', kmeans(X_target, k, seed), started)
    started = time.perf_counter()
    pca = PCA(n_components=min(X_target.shape), random_state=seed).fit(X_target)
    keep = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1)
    record('PCA + K-Means', kmeans(pca.transform(X_target)[:, :keep], k, seed), started)
    import umap
    started = time.perf_counter()
    n_umap = max(1, min(10, X_target.shape[1] - 1, X_target.shape[0] - 2))
    record('UMAP + K-Means', kmeans(umap.UMAP(n_components=n_umap, random_state=seed, n_jobs=1).fit_transform(X_target), k, seed), started)

    # Unsupervised baselines on standardized features
    scaled = StandardScaler().fit_transform(X_target)
    started = time.perf_counter()
    record('K-Means scaled', kmeans(scaled, k, seed), started)
    started = time.perf_counter()
    pca_scaled = PCA(n_components=min(scaled.shape), random_state=seed).fit(scaled)
    keep_scaled = int(np.argmax(np.cumsum(pca_scaled.explained_variance_ratio_) >= 0.90) + 1)
    record('PCA scaled + K-Means', kmeans(pca_scaled.transform(scaled)[:, :keep_scaled], k, seed), started)
    started = time.perf_counter()
    record('UMAP scaled + K-Means', kmeans(umap.UMAP(n_components=n_umap, random_state=seed, n_jobs=1).fit_transform(scaled), k, seed), started)

    # Proxy diagnostics that do not use the subtypes
    fraction = goes_left.mean(axis=0)
    homogeneity = np.maximum(fraction, 1.0 - fraction)
    diagnostics = {'Proxy_AUC': roc_auc_score(y, booster.predict_proba(X_full)[:, 1]),
                   'Mean_homogeneity': homogeneity.mean(),
                   'Trees_gt_95': int((homogeneity > 0.95).sum()),
                   'N_full': len(X_full), 'N_target': len(X_target), 'N_features': X_full.shape[1], 'K': k}
    for row in rows:
        row.update(diagnostics)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--output', default='history_study_development.csv')
    args = parser.parse_args()
    rows = []
    for name in ALL_DATASETS:
        X_full, y, X_target, subtypes, k = load_dataset(name)
        idx = target_indices(X_full, X_target)
        results = Parallel(n_jobs=-1)(
            delayed(run_seed)(seed, name, X_full, y, X_target, subtypes, k, idx) for seed in range(args.seeds))
        for seed_rows in results:
            rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, args.output), index=False)
        print(name, flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
