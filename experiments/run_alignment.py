"""Does supervision help when the proxy is aligned with the subtypes? A controlled test on TCGA and GTEx.

Analysis plan, fixed before any result of this study existed. Added after the main studies to test
the explanation offered in the manuscript: a supervised representation helps only when the proxy is
aligned with the subtypes. This is a mechanism test, not a realistic use case, because the proxies
are built from the subtype labels.

Cohorts: the 20 TCGA cancer types with molecular subtypes (as in run_tcga_hierarchy.py, Level B,
subtypes with fewer than 10 samples dropped) and the GTEx tissues with several sampling sites (as in
run_gtex.py). Each cohort is clustered on its own, as in the within-cohort studies.

Proxies: the subtypes of a cohort are split into two groups by a fixed rule (subtypes sorted by
size, largest first, each assigned to the currently smaller group). The proxy marks the samples of
one group. A share q of the samples then has its proxy label flipped at random, with
q in {0, 0.1, 0.2, 0.3, 0.4, 0.5}; q = 0 is a perfectly aligned coarse label and q = 0.5 is a random
proxy. A cohort is used when both groups have at least 20 samples and 15% of the cohort at q = 0.
Alignment is measured as the adjusted mutual information between the proxy and the subtypes.

Methods and settings match the TCGA studies: BTC, standardized TreeSHAP, RF leaves with SVD,
spectral RFC (Breiman), K-Means on raw and standardized features, the true K, 3 seeds.
Outcome: gain = method ARI minus the better of raw and standardized K-Means on the same cohort.

Questions, with 95% bootstrap intervals resampling cohorts (20,000 resamples), pooled over TCGA and
GTEx cohorts and reported per data source as description:
  A1 (primary). For each method, gain at q = 0 minus gain at q = 0.5. The explanation predicts a
      positive difference.
  A2. For each method and noise level, the mean gain. Reported: whether the gain at q = 0 exceeds
      zero, and the largest q at which the interval for the gain still lies above zero.
  A3. Spearman correlation between measured alignment and gain over all cohort-level units, per
      method, with a bootstrap interval resampling cohorts.
The explanation is supported for a method if A1 lies above zero and the gain at q = 0 lies above
zero; it is contradicted if A1 lies at or below zero.
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
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from baselines.rfc_factorized import FactorizedRFC
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments import run_gtex, run_tcga_hierarchy

OUT = os.path.join(ROOT, 'experiments', 'results', 'alignment.csv')
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS, MIN_CLASS_SIZE, MIN_CLASS_SHARE = 3, 20, 0.15


def tcga_cohorts():
    frame = run_tcga_hierarchy.load()
    X = frame.drop(columns=['cancer_type', 'subtype']).values.astype(np.float64)
    cancer = frame.cancer_type.values.astype(str)
    text = frame.subtype.fillna('').astype(str)
    for prefix in sorted(frame.subtype.dropna().str.split('.').str[0].unique()):
        members = run_tcga_hierarchy.SUBTYPE_TYPES.get(prefix, [prefix])
        cohort = np.isin(cancer, members) & text.str.startswith(prefix + '.').values & \
            ~(text.str.endswith('.NA') | text.str.endswith('.-')).values
        idx = np.where(cohort)[0]
        labels = frame.subtype.values[idx].astype(str)
        counts = pd.Series(labels).value_counts()
        keep = np.isin(labels, counts[counts >= run_tcga_hierarchy.MIN_SUBTYPE].index)
        yield 'TCGA', prefix, X[idx[keep]], labels[keep]


def gtex_cohorts():
    frame = run_gtex.load()
    X = frame.drop(columns=['tissue', 'site']).values.astype(np.float64)
    for name in run_gtex.tissues(frame):
        idx = np.where(frame.tissue.values == name)[0]
        yield 'GTEx', name, X[idx], frame.site.values[idx].astype(str)


def aligned_proxy(labels):
    sizes = pd.Series(labels).value_counts()
    groups, totals = ([], []), [0, 0]
    for subtype, size in sizes.items():
        g = int(totals[1] < totals[0])
        groups[g].append(subtype)
        totals[g] += size
    return np.isin(labels, groups[1]).astype(int)


def run(source, name, X, labels, y0, q, seed):
    import xgboost as xgb
    rng = np.random.RandomState(seed * 1000 + int(round(q * 100)))
    y = y0.copy()
    flip = rng.choice(len(y), int(round(q * len(y))), replace=False)
    y[flip] = 1 - y[flip]
    z = np.unique(labels, return_inverse=True)[1]
    k = int(z.max()) + 1

    def km(Z):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X, y)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X), extract_staged_decision_function(booster, X)]), axis=1)
    shap_model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1, colsample_bytree=0.8, n_jobs=1,
                                   random_state=seed, verbosity=0).fit(X, y)
    shap = shap_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X))
    comps = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    scores = {
        'BTC': km(updates),
        'TreeSHAP': km(StandardScaler().fit_transform(shap)),
        'RF leaves with SVD': km(TruncatedSVD(n_components=comps, random_state=seed).fit_transform(leaves)),
        'RFC spectral': FactorizedRFC(rf_model=forest, n_clusters=k, random_state=seed).fit_predict(X),
        'K-Means raw': km(X),
        'K-Means scaled': km(StandardScaler().fit_transform(X)),
    }
    base = {'Source': source, 'Task': name, 'Noise': q, 'seed': seed, 'K': k, 'N': len(y),
            'Alignment_AMI': adjusted_mutual_info_score(z, y)}
    return [{**base, 'Method': m, 'ARI': adjusted_rand_score(z, l)} for m, l in scores.items()]


def main():
    rows, skipped = [], []
    for source, name, X, labels in list(tcga_cohorts()) + list(gtex_cohorts()):
        y0 = aligned_proxy(labels)
        smallest = min(y0.sum(), len(y0) - y0.sum())
        if smallest < MIN_CLASS_SIZE or smallest / len(y0) < MIN_CLASS_SHARE:
            skipped.append(f'{source}/{name}')
            continue
        jobs = [delayed(run)(source, name, X, labels, y0, q, s) for q in NOISE for s in range(N_SEEDS)]
        for r in Parallel(n_jobs=-1, max_nbytes='1M')(jobs):
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(source, name, len(y0), flush=True)
    print('Skipped:', ', '.join(skipped) or 'none', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
