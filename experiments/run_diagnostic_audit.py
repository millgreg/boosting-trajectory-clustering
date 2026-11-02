import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.dataset_loader import load_dataset
from btc.core import BTC

CASE_DATASETS = [
    ('har', 'HAR (Sensor)'),
    ('thyroid', 'Thyroid'),
    ('brca', 'BRCA')
]

def run_diagnostic():
    rows = []
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_table_path = os.path.join(base_dir, 'paper', 'tables', 'diagnostic_audit.tex')
    
    for key, display_name in CASE_DATASETS:
        X_full, y_binary, X_target, subtypes, k = load_dataset(key)
        
        # 1. Fit canonical GBDT proxy model
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=42)
        gb.fit(X_full, y_binary)
        
        # 2. Proxy ROC-AUC
        if len(np.unique(y_binary)) > 1:
            auc = roc_auc_score(y_binary, gb.predict_proba(X_full)[:, 1])
        else:
            auc = np.nan
            
        # 3. Leaf homogeneity on target cohort
        homogeneity_props = []
        selected_feats = []
        for tree in gb.estimators_.ravel():
            feat = tree.tree_.feature[0]
            thresh = tree.tree_.threshold[0]
            selected_feats.append(feat)
            goes_left = (X_target[:, feat] <= thresh)
            prop = max(np.mean(goes_left), 1.0 - np.mean(goes_left))
            homogeneity_props.append(prop)
            
        mean_homogeneity = np.mean(homogeneity_props) * 100.0
        trees_gt_95 = sum(p > 0.95 for p in homogeneity_props)
        
        # 4. K-Means on all raw features
        km_all = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_target)
        ari_all = adjusted_rand_score(subtypes, km_all)
        
        # 5. K-Means on raw booster-selected features
        unique_feats = np.unique(selected_feats)
        km_sel = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_target[:, unique_feats])
        ari_sel = adjusted_rand_score(subtypes, km_sel)
        
        # 6. BTC on trajectory
        btc = BTC(booster=gb, n_clusters=k, representation='per_tree_delta', normalization='none', random_state=42)
        btc_labels = btc.fit_predict(X_target)
        ari_btc = adjusted_rand_score(subtypes, btc_labels)
        
        rows.append({
            'Dataset': display_name,
            'Proxy_AUC': f"${auc:.3f}$",
            'Raw_KM_All': f"${ari_all:.3f}$",
            'Raw_KM_Booster': f"${ari_sel:.3f}$",
            'Mean_Homogeneity': f"${mean_homogeneity:.1f}\\%$",
            'Trees_gt_95': f"${trees_gt_95} / 100$",
            'BTC_ARI': f"${ari_btc:.3f}$"
        })
        
    df = pd.DataFrame(rows)
    
    # Generate Transposed LaTeX table
    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{Diagnostic case study comparing the failure case (HAR) against representative successful datasets (Thyroid and BRCA). While booster-selected features on HAR retain subtype information ($0.142$ ARI with raw $K$-Means), $100\%$ of trees place $>95\%$ of target samples into identical leaves ($99.4\%$ mean homogeneity), causing trajectory variance collapse.}',
        r'\label{tab:app_diagnostic}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'\textbf{Diagnostic Metric} & ' + ' & '.join([f"\\textbf{{{r['Dataset']}}}" for r in rows]) + r' \\',
        r'\midrule',
        r'Proxy ROC-AUC & ' + ' & '.join([r['Proxy_AUC'] for r in rows]) + r' \\',
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Hypothesis 1: Feature Incoherence (Subtype Signal in Selected Features)}} \\',
        r'Raw $K$-Means ARI (All features) & ' + ' & '.join([r['Raw_KM_All'] for r in rows]) + r' \\',
        r'$K$-Means ARI (Booster-selected features only) & ' + ' & '.join([r['Raw_KM_Booster'] for r in rows]) + r' \\',
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Hypothesis 2: Intra-Cohort Threshold Collapse (Partition Diversity)}} \\',
        r'Mean Target Leaf Homogeneity & ' + ' & '.join([r['Mean_Homogeneity'] for r in rows]) + r' \\',
        r'Trees with $>95\%$ Target Homogeneity & ' + ' & '.join([r['Trees_gt_95'] for r in rows]) + r' \\',
        r'\midrule',
        r'\multicolumn{4}{l}{\textit{Downstream Representation Quality}} \\',
        r'BTC Trajectory ARI & ' + ' & '.join([r['BTC_ARI'] for r in rows]) + r' \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ]
    
    os.makedirs(os.path.dirname(out_table_path), exist_ok=True)
    with open(out_table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"Generated diagnostic audit table: {out_table_path}")

if __name__ == '__main__':
    run_diagnostic()
