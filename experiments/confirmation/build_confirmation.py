"""Build and audit the confirmation dataset list from OpenML. No clustering or boosting model is run.

Stage 1 selects candidates from metadata. Stage 2 loads each dataset and applies
mechanical validity rules. Stage 3 applies a documented manual exclusion list for
targets that are ordered levels or scores rather than kinds. All three stages are
fixed before any method is evaluated on these datasets.
"""
import os
import re
import sys

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from experiments.dataset_loader import clean_features

SEED = 20261103
MAX_ROWS_USED = 20000
HERE = os.path.dirname(os.path.abspath(__file__))

DEVELOPMENT = r'arrhythm|dermatol|cardioto|fetal|ctg|mice|diabetes|thyroid|allbp|allrep|allhyp|allhypo|sick|vehicle|segment|leukemia|golub|^har$|human.activity|walking|paul|brca|gbm|lgg|tcga'
CC18_FAMILY = r'^car[-_]|car_eval|jungle|kr-vs|kropt|king-rook|connect|mfeat|letter|optdigits|pendigits|satimage|texture|^dna|splice|isolet|semeion|vowel|mnist|fashion|cifar|devnagari|cnae|eucalyptus|balance|cmc|steel|wall-robot|gesture|first-order|authorship|dmft'
SYNTHETIC = r'bng|fri_|synth|artificial|random|rbf|hyperplane|waveform|^led|autouniv|openml|_seed|test|tmp|dummy|meta|mabbob|run_|task_|dgf_'

# Targets that are ordered levels, scores, or binned measurements. Judged from names and
# OpenML descriptions only, before any result on these datasets existed.
ORDINAL_TARGETS = {
    'abalone': 'binned age (rings)', 'wine-quality-red': 'quality score', 'wine-quality-white': 'quality score',
    'Credit_Score': 'credit level', 'Interest_Rate': 'binned rate', 'Mobile_Price': 'price range',
    'student_lifestyle_dataset': 'stress level', 'maternal_health_risk': 'risk level', 'marketing-income': 'income band',
    'Financial-Risk-Assessment': 'risk rating', 'company_quality_and_valuation_finance': 'total score',
    'Multiclass_Classification_for_Corporate_Credit_Ratings': 'credit rating', 'Estimation_of_Obesity_Levels': 'obesity level',
    'air-quality-and-pollution-assessment': 'air quality level', 'hepatitis_c_virus_hcv_for_egyptian_patients': 'histological stage',
    'user-knowledge': 'knowledge level', 'baseball': 'nomination level', 'Apple_Stock_Price_Trends': 'price movement',
    'product_sentiment_machine_hack': 'sentiment score', 'mental_health_detection': 'depression state',
    'diggle_table_a2': 'unnamed target column', 'Cervical_Cancer_Risk_Factors': 'combined test positives',
}
# Second review of included targets, from target names and class levels only, before any
# result on these datasets existed.
REVIEW_EXCLUSIONS = {
    'analcatdata_halloffame': 'ordinal nomination level, same data as baseball',
    'cars1': 'duplicate of cars',
    'Apple_Stock_Price_Trends_(2014-2023)': 'ordinal price movement',
    'Credit_Score_Classification': 'ordinal credit level',
    'total_score': 'ordinal score',
    'Student_Performance_on_an_Entrance_Examination': 'ordinal grade',
    'HCV_data': 'disease stage',
    'OSMI_Mental_Health_in_Tech_Survey': 'ordinal frequency answer',
    'eye_movements': 'ordinal relevance level',
    'Indicators_of_Anxiety_or_Depression': 'overlapping classes of aggregated survey rows',
    'fars': 'ordinal injury severity',
}
# Rules applied in stage 2.
MAX_LABEL_AUC = 0.999
MAX_DUPLICATE_PCT = 20.0
MAX_MISSING_PCT = 50.0
MIN_FEATURES = 4


def family(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_').split('_')[0][:8]


def candidates():
    cc18 = openml.study.get_suite(99)
    t = openml.datasets.list_datasets(output_format='dataframe')
    cc_names = t[t.did.isin(cc18.data)].name.str.lower().tolist()
    f = t[(t.status == 'active') & t.NumberOfClasses.between(3, 30) & t.NumberOfInstances.between(300, 200000)
          & t.NumberOfFeatures.between(4, 3000) & (t.MinorityClassSize >= 5) & (~t.did.isin(cc18.data))].copy()
    names = f.name.str.lower()
    f = f[~names.str.contains(DEVELOPMENT) & ~names.str.contains(CC18_FAMILY) & ~names.str.contains(SYNTHETIC) & ~names.isin(cc_names)].copy()
    f['family'] = f.name.map(family)
    f = f.sample(frac=1, random_state=SEED).sort_values('version', kind='stable').drop_duplicates('family', keep='first')
    return f.sort_values('did')


def audit(did):
    ds = openml.datasets.get_dataset(int(did), download_data=True, download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True).loc[y.notna().values]
    y = y.dropna().astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if len(X) > MAX_ROWS_USED:
        keep = np.sort(np.random.RandomState(SEED).permutation(len(X))[:MAX_ROWS_USED])
        X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    row = {'target': ds.default_target_attribute, 'N_used': len(X), 'missing_pct': round(100 * X.isna().mean().mean(), 1)}
    Z = clean_features(X)
    row['features'] = Z.shape[1]
    row['duplicate_pct'] = round(100 * (1 - len(np.unique(Z.round(8), axis=0)) / len(Z)), 1)
    counts = y.value_counts()
    row['K'] = int((counts >= 5).sum())
    pick = np.random.RandomState(0).permutation(len(Z))[:3000]
    Zs, ys = Z[pick], y.values[pick]
    best = 0.5
    for c in counts.index:
        b = (ys == c).astype(int)
        if 0 < b.sum() < len(b):
            for j in np.where(Zs.std(axis=0) > 0)[0]:
                a = roc_auc_score(b, Zs[:, j]); best = max(best, a, 1 - a)
    row['max_single_feature_auc'] = round(best, 4)
    return row


def main():
    f = candidates()
    rows = []
    for _, m in f.iterrows():
        r = {'did': int(m.did), 'name': m['name'], 'version': int(m.version)}
        try:
            r.update(audit(m.did))
            reasons = []
            if r['max_single_feature_auc'] >= MAX_LABEL_AUC: reasons.append('single feature determines a class')
            if r['duplicate_pct'] >= MAX_DUPLICATE_PCT: reasons.append('duplicated rows')
            if r['missing_pct'] >= MAX_MISSING_PCT: reasons.append('mostly missing')
            if r['features'] < MIN_FEATURES: reasons.append('too few features')
            if r['K'] < 3: reasons.append('fewer than 3 usable classes')
            if m['name'] in ORDINAL_TARGETS: reasons.append('ordinal target: ' + ORDINAL_TARGETS[m['name']])
            if m['name'] in REVIEW_EXCLUSIONS: reasons.append('review: ' + REVIEW_EXCLUSIONS[m['name']])
            r['excluded'] = '; '.join(reasons)
        except Exception as e:
            r['excluded'] = 'load failure: ' + str(e)[:80]
        rows.append(r)
        print(r['name'][:40], '|', r['excluded'] or 'included', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'confirmation_audit.csv'), index=False)
    df = pd.DataFrame(rows)
    df[df.excluded == ''].drop(columns='excluded').to_csv(os.path.join(HERE, 'confirmation_datasets.csv'), index=False)
    print('candidates', len(df), 'included', int((df.excluded == '').sum()))


if __name__ == '__main__':
    main()
