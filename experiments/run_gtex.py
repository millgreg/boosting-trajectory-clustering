"""Do coarse labels reveal finer groups when the fine labels were not derived by clustering? GTEx study.

Analysis plan, fixed before any expression data were downloaded. Only the tissue labels and the
donor and sample attribute tables had been inspected, to check which designs are possible.

Motivation: some TCGA molecular subtypes were defined by clustering expression data, which may
favour K-Means. GTEx sampling sites are anatomical labels recorded at collection, so they carry no
such bias. The study repeats the TCGA designs on GTEx with the same methods and settings.

Data: GTEx RSEM gene TPM from the UCSC Xena Toil hub (stored as log2(TPM + 0.001)), converted to
log2(TPM + 1). Included samples have a primary site and a detailed site; cell-culture sites
(detailed site starting with "Cells -") are excluded, and so are detailed sites with fewer than
30 samples with expression data. Genes with a mean log2(TPM + 1) of at least 1 over the included
samples are kept, and of these the 2,000 with the highest variance. Gene selection uses no labels.

Design 1, tissue versus rest (analogue of TCGA Level B). For every primary site with at least two
eligible detailed sites, the proxy is that primary site versus all other included samples, the
target cohort is the site's samples, and the subtypes are its detailed sites. From the labels,
this is expected to give Adipose Tissue, Blood Vessel, Brain, Colon, Esophagus, Heart and Skin.

Design 1 mirrors how a user would apply supervised clustering: a cohort of interest, a background,
and the contrast between them as the proxy.

Design 2, within-tissue proxies, a negative control and test of the explanation, not a realistic
use case. None of these attributes is expected to be related to the sampling site, and no user
would choose them to find sites. They test whether proxies without alignment to the subtypes fail,
as the explanation predicts. For the same cohorts, the proxy is a binary donor or sample attribute
measured within the cohort:
  sex        male versus female
  age        age bracket at or above the cohort's median bracket versus below
  death      ventilator case (Hardy scale 0) versus all other circumstances of death
  ischemic   ischemic time at or above the cohort median versus below
A proxy is used when both classes have at least 20 samples and at least 15% of the cohort, after
restricting to samples with a known proxy. The whole cohort is clustered.

Methods and settings match the TCGA studies: BTC, standardized TreeSHAP, RF leaves with SVD,
spectral RFC (Breiman), K-Means on raw and on standardized features, the true K, 5 seeds.
Outcome: gain = method ARI minus the better of raw and standardized K-Means on the same cohort.

Questions, with 95% bootstrap intervals (20,000 resamples):
  G1 (primary). Tissue versus rest: mean gain per method over tissues, resampling tissues. Also
      BTC minus TreeSHAP, minus RF leaves with SVD, and minus spectral RFC.
  G2 (negative control). Within-tissue proxies: mean gain per method over tissue-proxy pairs,
      resampling tissues. The expectation is no gain.
  G3 (secondary). Within-tissue minus tissue-versus-rest gain per tissue, averaged over tissues.
For G1, the TCGA finding is replicated for a method if the interval lies below zero, contradicted
if it lies above zero, and otherwise the result is inconclusive. For G2, an interval above zero
would contradict the explanation that unaligned proxies do not help.

Descriptive, to check the explanation that a proxy helps only when aligned with the subtypes:
mean leaf homogeneity of the BTC stumps on the target cohort for both designs, and, for Design 2,
the Spearman correlation across tissue-proxy pairs between BTC gain and the adjusted mutual
information of the proxy with the detailed sites, with a bootstrap interval over pairs. Because
these proxies are chosen to be unaligned, the range of alignment is expected to be narrow and the
correlation weakly informative.

Caveats stated in advance: there are only about seven tissues, so G1 intervals will be wide.
Donors contribute samples to several detailed sites of a cohort, and no donor correction is made.
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

DATA = os.path.join(ROOT, 'data', 'gtex')
OUT = os.path.join(ROOT, 'experiments', 'results', 'gtex.csv')
N_SEEDS, N_GENES, MIN_SITE, MIN_MEAN = 5, 2000, 30, 1.0
MIN_CLASS_SIZE, MIN_CLASS_SHARE = 20, 0.15
STORED_FLOOR = np.log2(0.001)


def labels():
    p = pd.read_csv(os.path.join(DATA, 'GTEX_phenotype.gz'), sep='\t', encoding='latin1').set_index('Sample')
    p = p.rename(columns={'body_site_detail (SMTSD)': 'site', '_primary_site': 'tissue'})
    p = p[p.tissue.notna() & (p.tissue != '<not provided>') & p.site.notna()]
    return p[~p.site.str.startswith('Cells -')][['tissue', 'site']]


def load():
    cache = os.path.join(DATA, 'gtex_top_genes.pkl')
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    lab = labels()
    path = os.path.join(DATA, 'gtex_RSEM_gene_tpm.gz')
    header = pd.read_csv(path, sep='\t', nrows=0).columns
    samples = [s for s in header[1:] if s in lab.index]
    counts = lab.loc[samples].site.value_counts()
    samples = [s for s in samples if counts[lab.site[s]] >= MIN_SITE]
    top, floor = None, np.inf
    for chunk in pd.read_csv(path, sep='\t', index_col=0, usecols=[header[0]] + samples, chunksize=2000):
        values = chunk[samples].values.astype(np.float64)
        floor = min(floor, values.min())
        x = np.log2(np.clip(np.exp2(values) - 0.001, 0, None) + 1)
        frame = pd.DataFrame(x, index=chunk.index, columns=samples)
        frame = frame[frame.mean(axis=1) >= MIN_MEAN]
        frame = frame.assign(_var=frame.var(axis=1))
        top = (frame if top is None else pd.concat([top, frame])).nlargest(N_GENES, '_var')
    assert abs(floor - STORED_FLOOR) < 0.01, f'unexpected storage format, minimum {floor:.3f}'
    top = top.drop(columns='_var')
    frame = top.T.astype('float32')
    frame.insert(0, 'tissue', lab.tissue.loc[frame.index].values)
    frame.insert(1, 'site', lab.site.loc[frame.index].values)
    frame.to_pickle(cache)
    return frame


def attributes(index):
    subj = pd.read_csv(os.path.join(DATA, 'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'), sep='\t').set_index('SUBJID')
    attr = pd.read_csv(os.path.join(DATA, 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'), sep='\t', low_memory=False).set_index('SAMPID')
    donor = pd.Index(['-'.join(s.split('-')[:2]) for s in index])
    a = pd.DataFrame(index=index)
    a['sex'] = subj.SEX.reindex(donor).map({2: 0, 1: 1}).values
    a['age_bracket'] = subj.AGE.reindex(donor).str[:2].astype(float).values
    a['death'] = (subj.DTHHRDY.reindex(donor) == 0).astype(float).where(subj.DTHHRDY.reindex(donor).notna()).values
    a['ischemic_value'] = attr.SMTSISCH.reindex(index).values
    return a


def tissues(frame):
    sites = frame.groupby('tissue').site.nunique()
    return sorted(sites[sites >= 2].index)


def tasks(frame):
    attr = attributes(frame.index)
    tissue = frame.tissue.values
    for name in tissues(frame):
        cohort = np.where(tissue == name)[0]
        yield 'tissue', name, 'tissue versus rest', cohort, (tissue == name).astype(int)
        a = attr.iloc[cohort]
        candidates = {'sex': a.sex, 'death': a.death}
        candidates['age'] = (a.age_bracket >= a.age_bracket.median()).astype(float).where(a.age_bracket.notna())
        candidates['ischemic'] = (a.ischemic_value >= a.ischemic_value.median()).astype(float).where(a.ischemic_value.notna())
        for proxy, values in candidates.items():
            known = values.notna().values
            idx, y = cohort[known], values.values[known].astype(int)
            smallest = min((y == 0).sum(), (y == 1).sum())
            if smallest < MIN_CLASS_SIZE or smallest / len(y) < MIN_CLASS_SHARE:
                continue
            yield 'within', name, proxy, idx, y


def run(design, name, proxy, idx, y, seed, X, site):
    import xgboost as xgb
    X_target = X[idx]
    X_fit = X if design == 'tissue' else X_target
    z = np.unique(site[idx], return_inverse=True)[1]
    k = int(z.max()) + 1

    def km(Z):
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)

    booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed).fit(X_fit, y)
    margins = extract_staged_decision_function(booster, X_target)
    updates = np.diff(np.column_stack([extract_raw_init(booster, X_target), margins]), axis=1)
    feats = np.array([t.tree_.feature[0] for t in booster.estimators_.ravel()])
    thr = np.array([t.tree_.threshold[0] for t in booster.estimators_.ravel()])
    frac = (X_target[:, feats] <= thr).mean(axis=0)
    shap_model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1, colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=0).fit(X_fit, y)
    shap = shap_model.get_booster().predict(xgb.DMatrix(X_target), pred_contribs=True)[:, :-1]
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X_fit, y)
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
    alignment = np.nan if design == 'tissue' else adjusted_mutual_info_score(z, y)
    base = {'Design': design, 'Task': name, 'Proxy': proxy, 'seed': seed, 'K': k, 'N_target': len(idx),
            'N_positive': int(y.sum()), 'Proxy_site_AMI': alignment, 'Mean_homogeneity': np.maximum(frac, 1 - frac).mean()}
    return [{**base, 'Method': m, 'ARI': adjusted_rand_score(z, l)} for m, l in scores.items()]


def main():
    frame = load()
    X = frame.drop(columns=['tissue', 'site']).values.astype(np.float64)
    site = frame.site.values.astype(str)
    print(f'{len(frame)} samples, {X.shape[1]} genes, tissues: {", ".join(tissues(frame))}', flush=True)
    rows = []
    for design, name, proxy, idx, y in tasks(frame):
        for r in Parallel(n_jobs=-1, max_nbytes='1M')(delayed(run)(design, name, proxy, idx, y, s, X, site) for s in range(N_SEEDS)):
            rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(design, name, proxy, len(idx), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
