"""Do meaningful coarse labels reveal subtypes? A pre-specified study on TCGA PanCanAtlas data.

Analysis plan, fixed before any result existed:

Data: batch-corrected PanCanAtlas RNA-seq, primary tumors (sample type 01) plus blood-derived
AML (03), restricted to the 2,000 genes with the highest variance across all included samples.
Gene selection uses no labels.

Level A, organ system to cancer type. Proxies are PanCanAtlas groupings: Pan-GI (ESCA, STAD,
COAD, READ), Pan-Gyn (BRCA, OV, UCEC, CESC, UCS), Pan-Kidney (KIRC, KIRP, KICH), Pan-Squamous
(BLCA, CESC, ESCA, HNSC, LUSC), Lung (LUAD, LUSC) and Glioma (GBM, LGG). The target cohort is
the group's cancer types, which are the subtypes. Each real grouping is compared with three
random groupings with the same number of cancer types.
  Primary outcome: gain = supervised ARI minus the better of raw and scaled K-Means on the same
  cohort; contrast = real-grouping gain minus mean random-grouping gain, averaged over the six
  groupings, for BTC and for the RF leaf embedding. Six groupings are descriptive only.

Level B, cancer type to molecular subtype. The proxy is one cancer type (or a PanCanAtlas
combined type) versus all other cancers; the target cohort is that type and the subtypes are
Subtype_Selected, excluding unknown labels (NA or "-") and subtypes with fewer than 10 samples.
  Outcomes: for each method, mean ARI and gain over the better K-Means baseline across tasks,
  with 95% bootstrap intervals over tasks. Pre-specified comparisons: BTC minus TreeSHAP, BTC
  minus RF leaves with SVD, BTC minus spectral RFC, and BTC gain over unsupervised.
  Diagnostic: Spearman correlation across tasks between mean target leaf homogeneity and BTC
  gain, reported with a bootstrap interval.

Caveats stated in advance: many molecular subtypes were defined by clustering expression data,
which favours unsupervised clustering on expression; cancer types are strongly separated by
tissue of origin.
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

DATA = os.path.join(ROOT, 'data', 'tcga_pancan')
OUT = os.path.join(ROOT, 'experiments', 'results', 'tcga_hierarchy.csv')
N_SEEDS, N_RANDOM, N_GENES, MIN_SUBTYPE, SEED = 5, 3, 2000, 10, 20261103
GROUPS = {
    'Pan-GI': ['ESCA', 'STAD', 'COAD', 'READ'],
    'Pan-Gyn': ['BRCA', 'OV', 'UCEC', 'CESC', 'UCS'],
    'Pan-Kidney': ['KIRC', 'KIRP', 'KICH'],
    'Pan-Squamous': ['BLCA', 'CESC', 'ESCA', 'HNSC', 'LUSC'],
    'Lung': ['LUAD', 'LUSC'],
    'Glioma': ['GBM', 'LGG'],
}
SUBTYPE_TYPES = {'GBM_LGG': ['GBM', 'LGG'], 'GI': ['COAD', 'READ', 'STAD', 'ESCA'], 'OVCA': ['OV'], 'AML': ['LAML']}


def load():
    cache = os.path.join(DATA, 'pancan_top_genes.pkl')
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    clinical = pd.read_csv(os.path.join(DATA, 'Survival_SupplementalTable_S1_20171025_xena_sp'), sep='\t')
    cancer = clinical.set_index('sample')['cancer type abbreviation']
    expression = pd.read_csv(os.path.join(DATA, 'EB_AdjustPANCAN_geneExp.xena.gz'), sep='\t', index_col=0).T
    keep = [s for s in expression.index if s[13:15] in ('01', '03') and s in cancer.index]
    expression = expression.loc[keep].dropna(axis=1)
    expression = expression[~expression.index.duplicated()]
    top = expression.var().sort_values(ascending=False).index[:N_GENES]
    frame = expression[top].astype('float32')
    frame.insert(0, 'cancer_type', cancer.loc[frame.index].values)
    subtypes = pd.read_csv(os.path.join(DATA, 'TCGASubtype.20170308.tsv.gz'), sep='\t').set_index('sampleID').Subtype_Selected
    frame.insert(1, 'subtype', subtypes.reindex(frame.index).values)
    frame.to_pickle(cache)
    return frame


def tasks(frame):
    cancers = np.sort(frame.cancer_type.unique())
    rng = np.random.RandomState(SEED)
    for name, members in GROUPS.items():
        yield 'A', name, 'real', 0, members, None
        for i in range(N_RANDOM):
            yield 'A', name, 'random', i + 1, sorted(rng.choice(cancers, size=len(members), replace=False)), None
    for prefix in sorted(frame.subtype.dropna().str.split('.').str[0].unique()):
        members = SUBTYPE_TYPES.get(prefix, [prefix])
        yield 'B', prefix, 'real', 0, members, prefix


def run(level, name, kind, grouping, members, prefix, seed, X, cancer, subtype):
    import xgboost as xgb
    y = np.isin(cancer, members).astype(int)
    if level == 'A':
        target_mask = y == 1
        labels = cancer[target_mask]
    else:
        text = pd.Series(subtype).fillna('')
        unknown = text.str.endswith('.NA') | text.str.endswith('.-')
        target_mask = (y == 1) & text.str.startswith(prefix + '.').values & ~unknown.values
        labels = subtype[target_mask]
        counts = pd.Series(labels).value_counts()
        keep = np.isin(labels, counts[counts >= MIN_SUBTYPE].index)
        idx = np.where(target_mask)[0][keep]
        target_mask = np.zeros(len(y), bool); target_mask[idx] = True
        labels = subtype[target_mask]
    X_target = X[target_mask]
    z = np.searchsorted(np.unique(labels), labels)
    k = len(np.unique(z))

    def km(Z):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X, y)
    margins = extract_staged_decision_function(booster, X_target)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X_target), margins]), axis=1)
    feats = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thr = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    frac = (X_target[:, feats] <= thr).mean(axis=0)
    shap_model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1, colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=0).fit(X, y)
    shap = shap_model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X, y)
    leaves = OneHotEncoder(handle_unknown='ignore').fit_transform(forest.apply(X_target))
    comps = min(50, leaves.shape[0] - 1, leaves.shape[1] - 1)
    scores = {
        'BTC': km(updates),
        'TreeSHAP': km(StandardScaler().fit_transform(shap)),
        'RF leaves with SVD': km(TruncatedSVD(n_components=comps, random_state=seed).fit_transform(leaves)),
        'RFC spectral': FactorizedRFC(rf_model=forest, n_clusters=k, random_state=seed).fit_predict(X_target),
        'K-Means raw': km(X_target),
        'K-Means scaled': km(StandardScaler().fit_transform(X_target)),
    }
    base = {'Level': level, 'Task': name, 'Proxy': kind, 'grouping': grouping, 'members': ','.join(members), 'seed': seed,
            'K': k, 'N_target': int(target_mask.sum()), 'N_background': int((y == 0).sum()),
            'Mean_homogeneity': np.maximum(frac, 1 - frac).mean()}
    return [{**base, 'Method': m, 'ARI': adjusted_rand_score(z, l)} for m, l in scores.items()]


def main():
    frame = load()
    X = frame.drop(columns=['cancer_type', 'subtype']).values.astype(np.float64)
    cancer = frame.cancer_type.values.astype(str)
    subtype = frame.subtype.values.astype(object)
    rows = []
    for level, name, kind, grouping, members, prefix in tasks(frame):
        results = Parallel(n_jobs=-1, max_nbytes='1M')(
            delayed(run)(level, name, kind, grouping, members, prefix, s, X, cancer, subtype) for s in range(N_SEEDS))
        for r in results:
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(level, name, kind, grouping, flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
