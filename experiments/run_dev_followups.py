"""Two development-only follow-ups, run before the confirmation protocol is frozen.

Part A, SHAP fairness: compares BTC with several common SHAP clustering setups on the
13 benchmarks. The best SHAP setup becomes the SHAP comparator for confirmation.

Part B, homogeneity on a fixed cohort: for each multi-class development dataset, the
target classes are held fixed while the background classes vary. This removes the
cohort confound and asks whether lower leaf homogeneity predicts higher BTC ARI.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import openml
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.dataset_loader import ALL_DATASETS, clean_features, load_dataset
from experiments.run_proxy_contrast import class_labels

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SHAP_SETUPS = [(depth, colsample, clustering) for depth in (1, 3, 6) for colsample in (0.8, 1.0)
               for clustering in ('kmeans', 'kmeans_scaled', 'umap', 'ward')]


def kmeans(Z, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def btc_updates(X_full, y, X_target, seed):
    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_full, y)
    margins = extract_staged_decision_function(booster, X_target)
    return booster, np.diff(np.column_stack([extract_raw_init(booster, X_target), margins]), axis=1)


def cluster(Z, k, seed, how):
    if how == 'kmeans':
        return kmeans(Z, k, seed)
    if how == 'kmeans_scaled':
        return kmeans(StandardScaler().fit_transform(Z), k, seed)
    if how == 'ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Z) if len(Z) <= 10000 else kmeans(Z, k, seed)
    import umap
    n = max(1, min(10, Z.shape[1] - 1, Z.shape[0] - 2))
    return kmeans(umap.UMAP(n_components=n, random_state=seed, n_jobs=1).fit_transform(Z), k, seed)


def shap_task(seed, name, X_full, y, X_target, z, k):
    import xgboost as xgb
    rows = [{'Dataset': name, 'seed': seed, 'Method': 'BTC', 'ARI': adjusted_rand_score(z, kmeans(btc_updates(X_full, y, X_target, seed)[1], k, seed))}]
    cache = {}
    for depth, colsample, how in SHAP_SETUPS:
        if (depth, colsample) not in cache:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=0.1, colsample_bytree=colsample,
                                      n_jobs=1, random_state=seed, verbosity=0).fit(X_full, y)
            cache[(depth, colsample)] = model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]
        S = cache[(depth, colsample)]
        if how == 'umap' and seed >= 5:
            continue
        rows.append({'Dataset': name, 'seed': seed, 'Method': f'TreeSHAP depth {depth}, columns {colsample}, {how}',
                     'ARI': adjusted_rand_score(z, cluster(S, k, seed, how))})
    return rows


def homogeneity_task(seed, name, background_id, X_all, labels, target, background):
    keep = np.isin(labels, list(target) + list(background))
    X_full, lab = X_all[keep], labels[keep]
    y = np.isin(lab, target).astype(int)
    X_target, t = X_full[y == 1], lab[y == 1]
    z = np.searchsorted(np.unique(t), t)
    booster, R = btc_updates(X_full, y, X_target, seed)
    feats = np.array([e.tree_.feature[0] for e in booster.estimators_.ravel()])
    thr = np.array([e.tree_.threshold[0] for e in booster.estimators_.ravel()])
    frac = (X_target[:, feats] <= thr).mean(axis=0)
    return {'Dataset': name, 'background': background_id, 'seed': seed, 'N_background': int((y == 0).sum()),
            'BTC': adjusted_rand_score(z, kmeans(R, len(np.unique(z)), seed)),
            'Mean_homogeneity': np.maximum(frac, 1 - frac).mean(),
            'Trees_gt_95': int((np.maximum(frac, 1 - frac) > 0.95).sum()),
            'Proxy_AUC': roc_auc_score(y, booster.predict_proba(X_full)[:, 1])}


def homogeneity_datasets():
    for name in ('har', 'mice', 'arrhythmia', 'dermatology', 'segment', 'paul15'):
        yield name, load_dataset(name)[0], class_labels(name)
    suite = openml.study.get_suite(99)
    meta = openml.datasets.list_datasets(data_id=suite.data, output_format='dataframe')
    meta = meta[(meta.NumberOfClasses >= 5) & (meta.NumberOfInstances <= 20000) & (meta.NumberOfFeatures <= 1000)]
    meta = meta[~meta.name.str.lower().isin({'har', 'vehicle', 'segment', 'miceprotein'})]
    for _, m in meta.iterrows():
        ds = openml.datasets.get_dataset(int(m.did), download_qualities=False, download_features_meta_data=False)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        yield 'cc18_' + m['name'], clean_features(X), pd.Series(y).astype(str).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['shap', 'homogeneity'], required=True)
    args = parser.parse_args()
    rows = []
    if args.part == 'shap':
        for name in ALL_DATASETS:
            data = load_dataset(name)
            for r in Parallel(n_jobs=-1)(delayed(shap_task)(s, name, *data) for s in range(10)):
                rows.extend(r)
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'dev_shap_setups.csv'), index=False)
            print(name, flush=True)
    else:
        for name, X_all, labels in homogeneity_datasets():
            counts = pd.Series(labels).value_counts()
            classes = np.sort(counts[counts >= 10].index.values)
            if len(classes) < 5:
                continue
            rng = np.random.RandomState(sum(map(ord, name)))
            target = tuple(rng.choice(classes, size=3 if len(classes) >= 7 else 2, replace=False))
            rest = [c for c in classes if c not in target]
            backgrounds = []
            while len(backgrounds) < 5:
                b = tuple(sorted(rng.choice(rest, size=rng.randint(1, len(rest) + 1), replace=False)))
                if b not in backgrounds:
                    backgrounds.append(b)
                if len(backgrounds) >= 2 ** len(rest) - 1:
                    break
            tasks = [(s, name, i, X_all, labels, target, b) for i, b in enumerate(backgrounds) for s in range(5)]
            rows.extend(Parallel(n_jobs=-1)(delayed(homogeneity_task)(*t) for t in tasks))
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'dev_homogeneity_fixed_cohort.csv'), index=False)
            print(name, target, len(backgrounds), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
