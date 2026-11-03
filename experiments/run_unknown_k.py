"""Exploratory: do the conclusions hold when the number of subtypes must be estimated?

Analysis plan, fixed before any result existed. Exploratory, added after the main studies.

Every method chooses K from 2 to 15, capped at one tenth of the cohort size, by the silhouette
score of its own representation (Euclidean, computed on up to 2,000 samples), then reports the ARI
of the partition at the chosen K. Spectral RFC uses the first k eigenvectors of its exact embedding
for each candidate k. Methods: BTC, standardized TreeSHAP (the frozen confirmation setup), RF
leaves with SVD, spectral RFC (Breiman), and K-Means on raw and on standardized features. The true K
is also scored for reference. Silhouette selection tends to favour small K, which is reported.

Data: the 40 confirmation datasets (3 groupings, 3 seeds), the 6 real TCGA organ-system groupings
and 20 TCGA molecular-subtype tasks (3 seeds), and the 86 TCGA within-cohort pairs (3 seeds).
Datasets whose true K exceeds the candidate range are kept and reported.

Questions, each with 95% bootstrap intervals (over datasets, tasks, or cancer types):
  U1. With K estimated, BTC minus each comparator.
  U2. With K estimated, each supervised method's gain over raw and over standardized K-Means,
      separately, since choosing the better of the two would require the true labels.
  U3. The cost of not knowing K: ARI at the chosen K minus ARI at the true K, and how often the
      chosen K equals the true K.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from baselines import rf_proximities
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.run_proximity_variants import confirmation_tasks, tcga_tasks

OUT = os.path.join(ROOT, 'experiments', 'results', 'unknown_k.csv')
K_MAX, N_SEEDS, SILHOUETTE_SAMPLE = 15, 3, 2000


def candidate_ks(n):
    return list(range(2, max(2, min(K_MAX, n // 10)) + 1))


def select(Z_for_k, ks, seed):
    """Return (chosen k, labels at chosen k, labels by k) using the silhouette score."""
    best, best_k, labels_by_k = -np.inf, ks[0], {}
    for k in ks:
        Z = Z_for_k(k)
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        labels_by_k[k] = labels
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(Z, labels, sample_size=min(SILHOUETTE_SAMPLE, len(Z)), random_state=seed)
        if s > best:
            best, best_k = s, k
    return best_k, labels_by_k


def run(X_train, y, target_rows, labels, seed, base):
    import xgboost as xgb
    z = np.searchsorted(np.unique(labels), labels)
    k_true = len(np.unique(z))
    X_target = X_train[target_rows]
    ks = candidate_ks(len(target_rows))

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_train, y)
    margins = extract_staged_decision_function(booster, X_target)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X_target), margins]), axis=1)
    shap_model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1, colsample_bytree=0.8,
                                   n_jobs=1, random_state=seed, verbosity=0).fit(X_train, y)
    shap = StandardScaler().fit_transform(shap_model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1])
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_train, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    svd = TruncatedSVD(n_components=min(50, leaves.shape[0] - 1, leaves.shape[1] - 1), random_state=seed).fit_transform(leaves)
    n_eig = min(max(ks + [k_true]), len(target_rows) - 2)
    spectral = rf_proximities.spectral_embedding(forest, X_train, target_rows, 'breiman', n_eig, random_state=seed)

    representations = {
        'BTC': lambda k: updates,
        'TreeSHAP': lambda k: shap,
        'RF leaves with SVD': lambda k: svd,
        'RFC spectral': lambda k: spectral[:, :min(k, n_eig)],
        'K-Means raw': lambda k: X_target,
        'K-Means scaled': (lambda S: (lambda k: S))(StandardScaler().fit_transform(X_target)),
    }
    rows = []
    for method, Z_for_k in representations.items():
        chosen, labels_by_k = select(Z_for_k, ks, seed)
        true_labels = labels_by_k.get(k_true)
        if true_labels is None:
            true_labels = KMeans(n_clusters=k_true, n_init=10, random_state=seed).fit_predict(Z_for_k(k_true))
        rows.append({**base, 'seed': seed, 'Method': method, 'K_true': k_true, 'K_chosen': chosen, 'K_max': max(ks),
                     'ARI_chosen': adjusted_rand_score(z, labels_by_k[chosen]), 'ARI_true_K': adjusted_rand_score(z, true_labels)})
    return rows


def tasks():
    for source in (confirmation_tasks, tcga_tasks):
        for X_train, y, target, labels, seed, base in source():
            if seed >= N_SEEDS:
                continue
            if base.get('Study') == 'tcga_hierarchy' and base.get('Proxy') == 'random':
                continue
            yield X_train, y, target, labels, seed, base


def main():
    rows, batch = [], []
    for task in tasks():
        batch.append(task)
        if len(batch) == 9:
            for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(run)(*t) for t in batch):
                rows.extend(r)
            batch = []
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(rows[-1]['Study'], rows[-1].get('Dataset', rows[-1].get('Task')), flush=True)
    if batch:
        for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(run)(*t) for t in batch):
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
