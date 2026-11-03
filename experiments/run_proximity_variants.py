"""Exploratory: do other Random Forest proximities change the conclusions?

Analysis plan, fixed before any result existed. This study is exploratory because it was added
after the confirmation protocol was frozen and after the TCGA results were known.

Proximities: Breiman (all trees), out-of-bag (separable approximation of Aumon et al., 2026), and
RF-GAP (Rhodes et al., 2023), each followed by spectral clustering with the true K, computed
exactly without the N x N matrix (baselines/rf_proximities.py). Forests use 100 trees and the same
seeds as the original runs, so Breiman results should closely reproduce the earlier spectral RFC.

Data: the 40 confirmation datasets (3 groupings x 5 seeds), the 44 TCGA hierarchy tasks, and the
86 TCGA within-cohort pairs (5 seeds each), with the same task definitions as the original studies.

Questions, each with 95% bootstrap intervals (over datasets, TCGA tasks, or cancer types):
  P1. Do OOB or RF-GAP differ from Breiman? Mean ARI difference per study.
  P2. Does any variant beat unsupervised clustering on TCGA? Gain over the better of raw and
      standardized K-Means from the original runs; "beats" requires the lower bound above 0.
  P3. On the confirmation datasets, each variant minus BTC from the original run.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from baselines import rf_proximities
from experiments import run_confirmation, run_tcga_hierarchy, run_tcga_within_cohort

OUT = os.path.join(ROOT, 'experiments', 'results', 'proximity_variants.csv')
VARIANTS = list(rf_proximities.VARIANTS)


def score_variants(X_train, y, target_rows, labels, seed, base, variants=None):
    z = np.searchsorted(np.unique(labels), labels)
    k = len(np.unique(z))
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_train, y)
    rows = []
    for variant in (variants or VARIANTS):
        predicted = rf_proximities.cluster(forest, X_train, target_rows, variant, k, random_state=seed)
        rows.append({**base, 'seed': seed, 'K': k, 'Method': variant, 'ARI': adjusted_rand_score(z, predicted)})
    return rows


def confirmation_tasks():
    for _, d in pd.read_csv(run_confirmation.LIST).iterrows():
        X_all, labels = run_confirmation.load(d.did)
        for g, background in enumerate(run_confirmation.groupings(d.did, labels)):
            y = (~np.isin(labels, background)).astype(int)
            target = np.where(y == 1)[0]
            for seed in range(run_confirmation.N_SEEDS):
                yield X_all, y, target, labels[target], seed, {'Study': 'confirmation', 'did': int(d.did), 'Dataset': d['name'], 'grouping': g}


def tcga_tasks():
    frame = run_tcga_hierarchy.load()
    X = frame.drop(columns=['cancer_type', 'subtype']).values.astype(np.float64)
    cancer = frame.cancer_type.values.astype(str)
    subtype = frame.subtype.values.astype(object)
    text = pd.Series(subtype).fillna('')
    unknown = (text.str.endswith('.NA') | text.str.endswith('.-')).values
    for level, name, kind, grouping, members, prefix in run_tcga_hierarchy.tasks(frame):
        y = np.isin(cancer, members).astype(int)
        if level == 'A':
            target = np.where(y == 1)[0]
            labels = cancer[target]
        else:
            mask = (y == 1) & text.str.startswith(prefix + '.').values & ~unknown
            idx = np.where(mask)[0]
            counts = pd.Series(subtype[idx]).value_counts()
            target = idx[np.isin(subtype[idx], counts[counts >= run_tcga_hierarchy.MIN_SUBTYPE].index)]
            labels = subtype[target].astype(str)
        for seed in range(run_tcga_hierarchy.N_SEEDS):
            yield X, y, target, labels, seed, {'Study': 'tcga_hierarchy', 'Level': level, 'Task': name, 'Proxy': kind, 'grouping': grouping}
    proxies = run_tcga_within_cohort.clinical_proxies()
    for prefix, proxy_name, idx, y, labels in run_tcga_within_cohort.tasks(frame, proxies):
        X_cohort = X[idx]
        for seed in range(run_tcga_hierarchy.N_SEEDS):
            yield X_cohort, y, np.arange(len(idx)), labels, seed, {'Study': 'tcga_within', 'Task': prefix, 'Proxy': proxy_name}


def main():
    global OUT, VARIANTS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=VARIANTS)
    parser.add_argument('--output', default=OUT)
    args = parser.parse_args()
    VARIANTS, OUT = args.variants, args.output
    rows = []
    for source in (confirmation_tasks, tcga_tasks):
        batch = []
        for task in source():
            batch.append(task)
            if len(batch) == 15:
                for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(score_variants)(*t, VARIANTS) for t in batch):
                    rows.extend(r)
                batch = []
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(rows[-1]['Study'], rows[-1].get('Dataset', rows[-1].get('Task')), flush=True)
        if batch:
            for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(score_variants)(*t, VARIANTS) for t in batch):
                rows.extend(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
