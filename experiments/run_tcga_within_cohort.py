"""Do proxies that vary within the target cohort help subtype discovery? Pre-specified TCGA study.

Analysis plan, fixed before any result existed:

Design: for each cancer-type cohort with published molecular subtypes (the 20 Level B tasks of
run_tcga_hierarchy.py), the proxy is a binary clinical label measured within that cohort, and the
whole cohort is clustered. This is the cross-cutting design, in contrast to the cohort-versus-rest
proxies of the hierarchy study.

Proxies, defined mechanically:
  stage       pathologic stage I-II versus III-IV
  grade       G1, G2 or Low Grade versus G3, G4 or High Grade
  histology   most common histological type in the cohort versus all others
  progression progression-free interval event versus none
  survival    overall survival event versus none
  age         at or above the cohort median versus below
  sex         male versus female, included as a proxy expected to carry little subtype information
  er, pr      estrogen and progesterone receptor status, breast cancer only
A proxy is used when both classes have at least 20 samples and at least 15% of the cohort, after
restricting to samples with a known proxy and a known subtype. Subtypes with fewer than 10 samples
are dropped.

Outcome: gain = method ARI minus the better of raw and scaled K-Means on the same cohort.
Primary questions, with 95% bootstrap intervals that resample cancer types (20,000 resamples):
  Q1. Do within-cohort proxies help? Mean gain over all task-proxy pairs, per method.
      "Helps" requires the lower bound above 0.
  Q2. Are they better than cohort-versus-rest proxies? Per cancer type, the mean gain over its
      within-cohort proxies minus its gain in the hierarchy study, averaged over types.
      "Better" requires the lower bound above 0.
Descriptive: mean gain by proxy type, and the share of task-proxy pairs with positive BTC gain.
Methods match the hierarchy study: BTC, TreeSHAP, RF leaves with SVD, spectral RFC, and K-Means.
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
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from baselines.rfc_factorized import FactorizedRFC
from btc.core import extract_raw_init, extract_staged_decision_function
from experiments.run_tcga_hierarchy import DATA, MIN_SUBTYPE, N_SEEDS, SUBTYPE_TYPES, load

OUT = os.path.join(ROOT, 'experiments', 'results', 'tcga_within_cohort.csv')
MIN_CLASS_SIZE, MIN_CLASS_SHARE = 20, 0.15


def clinical_proxies():
    c = pd.read_csv(os.path.join(DATA, 'Survival_SupplementalTable_S1_20171025_xena_sp'), sep='\t').set_index('sample')
    stage = c.ajcc_pathologic_tumor_stage.astype(str).str.extract(r'Stage (IV|III|II|I)')[0]
    grade = c.histological_grade.astype(str)
    p = pd.DataFrame(index=c.index)
    p['stage'] = stage.map({'I': 0, 'II': 0, 'III': 1, 'IV': 1})
    p['grade'] = grade.map({'G1': 0, 'G2': 0, 'Low Grade': 0, 'G3': 1, 'G4': 1, 'High Grade': 1})
    p['progression'] = c.PFI
    p['survival'] = c.OS
    p['sex'] = c.gender.map({'FEMALE': 0, 'MALE': 1})
    p['histology_label'] = c.histological_type
    p['age_value'] = c.age_at_initial_pathologic_diagnosis
    brca = pd.read_csv(os.path.join(ROOT, 'data', 'tcga', 'BRCA_clinicalMatrix'), sep='\t').set_index('sampleID')
    p['er'] = brca.breast_carcinoma_estrogen_receptor_status.map({'Negative': 0, 'Positive': 1}).reindex(p.index)
    p['pr'] = brca.breast_carcinoma_progesterone_receptor_status.map({'Negative': 0, 'Positive': 1}).reindex(p.index)
    return p


def tasks(frame, proxies):
    cancer = frame.cancer_type.values.astype(str)
    text = frame.subtype.fillna('').astype(str)
    for prefix in sorted(frame.subtype.dropna().str.split('.').str[0].unique()):
        members = SUBTYPE_TYPES.get(prefix, [prefix])
        cohort = np.isin(cancer, members) & text.str.startswith(prefix + '.').values & ~(text.str.endswith('.NA') | text.str.endswith('.-')).values
        samples = frame.index[cohort]
        clin = proxies.reindex(samples)
        candidates = {name: clin[name] for name in ['stage', 'grade', 'progression', 'survival', 'sex', 'er', 'pr']}
        histology = clin.histology_label
        if histology.notna().sum():
            candidates['histology'] = (histology != histology.value_counts().index[0]).astype(float).where(histology.notna())
        age = clin.age_value
        candidates['age'] = (age >= age.median()).astype(float).where(age.notna())
        for proxy_name, values in candidates.items():
            known = values.notna().values
            idx = np.where(cohort)[0][known]
            if len(idx) == 0:
                continue
            labels = frame.subtype.values[idx].astype(str)
            counts = pd.Series(labels).value_counts()
            keep = np.isin(labels, counts[counts >= MIN_SUBTYPE].index)
            idx, labels = idx[keep], labels[keep]
            y = values.values[known][keep].astype(int)
            if len(np.unique(labels)) < 2:
                continue
            smallest = min((y == 0).sum(), (y == 1).sum())
            if smallest < MIN_CLASS_SIZE or smallest / len(y) < MIN_CLASS_SHARE:
                continue
            yield prefix, proxy_name, idx, y, labels


def run(prefix, proxy_name, idx, y, labels, seed, X):
    import xgboost as xgb
    X_cohort = X[idx]
    z = np.searchsorted(np.unique(labels), labels)
    k = len(np.unique(z))

    def km(Z):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_cohort, y)
    margins = extract_staged_decision_function(booster, X_cohort)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X_cohort), margins]), axis=1)
    feats = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thr = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    frac = (X_cohort[:, feats] <= thr).mean(axis=0)
    shap_model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1, colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=0).fit(X_cohort, y)
    shap = shap_model.get_booster().predict(xgb.DMatrix(X_cohort), pred_contribs=True)[:, :-1]
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_cohort, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_cohort))
    comps = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    scores = {
        'BTC': km(updates),
        'TreeSHAP': km(StandardScaler().fit_transform(shap)),
        'RF leaves with SVD': km(TruncatedSVD(n_components=comps, random_state=seed).fit_transform(leaves)),
        'RFC spectral': FactorizedRFC(rf_model=forest, n_clusters=k, random_state=seed).fit_predict(X_cohort),
        'K-Means raw': km(X_cohort),
        'K-Means scaled': km(StandardScaler().fit_transform(X_cohort)),
    }
    base = {'Task': prefix, 'Proxy': proxy_name, 'seed': seed, 'K': k, 'N': len(idx), 'N_positive': int(y.sum()),
            'Mean_homogeneity': np.maximum(frac, 1 - frac).mean()}
    return [{**base, 'Method': m, 'ARI': adjusted_rand_score(z, l)} for m, l in scores.items()]


def main():
    frame = load()
    X = frame.drop(columns=['cancer_type', 'subtype']).values.astype(np.float64)
    proxies = clinical_proxies()
    rows = []
    for prefix, proxy_name, idx, y, labels in tasks(frame, proxies):
        for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(run)(prefix, proxy_name, idx, y, labels, s, X) for s in range(N_SEEDS)):
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(prefix, proxy_name, len(idx), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
