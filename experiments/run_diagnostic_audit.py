"""Diagnostic audit of the HAR failure mode, averaged over the benchmark seeds."""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import BTC
from experiments.dataset_loader import load_dataset

N_SEEDS = 30
CASE_DATASETS = [
    ('har', 'HAR'),
    ('thyroid', 'Thyroid'),
    ('brca', 'BRCA'),
    ('leukemia', 'Leukemia'),
]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TABLE_PATH = os.path.join(BASE_DIR, 'paper', 'tables', 'diagnostic_audit.tex')
CSV_PATH = os.path.join(BASE_DIR, 'experiments', 'results', 'diagnostic_audit.csv')


def run_seed(seed, key, X_full, y_binary, X_target, subtypes, k):
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed)
    gb.fit(X_full, y_binary)
    auc = roc_auc_score(y_binary, gb.predict_proba(X_full)[:, 1])

    proportions, features = [], []
    for tree in gb.estimators_.ravel():
        feature = tree.tree_.feature[0]
        goes_left = X_target[:, feature] <= tree.tree_.threshold[0]
        proportions.append(max(goes_left.mean(), 1.0 - goes_left.mean()))
        features.append(feature)
    unique_features = np.unique(features)

    km_all = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_target)
    km_selected = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_target[:, unique_features])
    btc = BTC(booster=gb, n_clusters=k, representation='per_tree_delta', normalization='none', random_state=seed)
    return {
        'Dataset': key, 'seed': seed, 'Proxy_AUC': auc,
        'KM_all_ARI': adjusted_rand_score(subtypes, km_all),
        'KM_selected_ARI': adjusted_rand_score(subtypes, km_selected),
        'N_selected_features': len(unique_features),
        'Mean_homogeneity': 100.0 * np.mean(proportions),
        'Trees_gt_95': int(np.sum(np.array(proportions) > 0.95)),
        'BTC_ARI': adjusted_rand_score(subtypes, btc.fit_predict(X_target)),
    }


def main():
    rows = []
    for key, _ in CASE_DATASETS:
        data = load_dataset(key)
        rows.extend(Parallel(n_jobs=-1)(delayed(run_seed)(seed, key, *data) for seed in range(N_SEEDS)))
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    mean = df.drop(columns='seed').groupby('Dataset').mean()
    std = df.drop(columns='seed').groupby('Dataset').std()
    keys = [key for key, _ in CASE_DATASETS]

    def ari_row(column):
        return ' & '.join(f"${mean.loc[k, column]:.3f} \\pm {std.loc[k, column]:.3f}$" for k in keys)

    har = mean.loc['har']
    tex = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        (r'\caption{Diagnostic case study comparing the failure case (HAR) against datasets where supervised clustering recovers subtypes. '
         r'Values are means over 30 seeds, with standard deviations for ARI. On HAR, booster-selected features retain subtype information '
         f"(${har['KM_selected_ARI']:.3f}$ ARI with raw $K$-Means), yet ${har['Trees_gt_95']:.1f}$ of $100$ trees place $>95\\%$ of target samples into one leaf "
         f"(${har['Mean_homogeneity']:.1f}\\%$ mean homogeneity), collapsing trajectory variance.}}"),
        r'\label{tab:app_diagnostic}', r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Diagnostic Metric} & ' + ' & '.join(f'\\textbf{{{label}}}' for _, label in CASE_DATASETS) + r' \\', r'\midrule',
        r'Proxy ROC-AUC & ' + ' & '.join(f"${mean.loc[k, 'Proxy_AUC']:.3f}$" for k in keys) + r' \\', r'\midrule',
        r'\multicolumn{5}{l}{\textit{Hypothesis 1: Feature Incoherence}} \\',
        r'$K$-Means ARI (all features) & ' + ari_row('KM_all_ARI') + r' \\',
        r'$K$-Means ARI (selected features) & ' + ari_row('KM_selected_ARI') + r' \\',
        r'Selected features & ' + ' & '.join(f"${mean.loc[k, 'N_selected_features']:.1f}$" for k in keys) + r' \\', r'\midrule',
        r'\multicolumn{5}{l}{\textit{Hypothesis 2: Intra-Cohort Threshold Collapse}} \\',
        r'Mean target leaf homogeneity & ' + ' & '.join(f"${mean.loc[k, 'Mean_homogeneity']:.1f}\\%$" for k in keys) + r' \\',
        r'Trees with $>95\%$ homogeneity & ' + ' & '.join(f"${mean.loc[k, 'Trees_gt_95']:.1f}$" for k in keys) + r' \\', r'\midrule',
        r'BTC trajectory ARI & ' + ari_row('BTC_ARI') + r' \\',
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ]
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(mean.round(3).T.to_string())
    print(std.round(3).T.to_string())


if __name__ == '__main__':
    main()
