import os
import numpy as np
import pandas as pd
from dataset_loader import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_FILE = os.path.join(PAPER_DIR, 'tables', 'main_results.tex')
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

DATASET_ORDER = [
    'arrhythmia',
    'brca',
    'ctg',
    'dermatology',
    'diabetes',
    'gbmlgg',
    'har',
    'leukemia',
    'mice',
    'paul15',
    'segment',
    'thyroid',
    'vehicle'
]

DATASET_METADATA = {
    'arrhythmia': {'display': 'Arrhythmia', 'domain': 'Medical'},
    'brca': {'display': 'BRCA', 'domain': 'Genomics'},
    'ctg': {'display': 'CTG', 'domain': 'Medical'},
    'dermatology': {'display': 'Dermatology', 'domain': 'Medical'},
    'diabetes': {'display': 'Diabetes', 'domain': 'Medical'},
    'gbmlgg': {'display': 'GBMLGG', 'domain': 'Genomics'},
    'har': {'display': 'HAR', 'domain': 'Sensor'},
    'leukemia': {'display': 'Leukemia', 'domain': 'Genomics'},
    'mice': {'display': 'Mice', 'domain': 'Proteomics'},
    'paul15': {'display': 'Paul15', 'domain': 'scRNA-seq'},
    'segment': {'display': 'Segment', 'domain': 'ML benchmark'},
    'thyroid': {'display': 'Thyroid', 'domain': 'Medical'},
    'vehicle': {'display': 'Vehicle', 'domain': 'ML benchmark'}
}

df = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
subset = df[df['Method'].isin(['BTC_Stochastic_Delta_sqrt_none', 'Baseline_RFC'])].copy()
subset['Method'] = subset['Method'].replace({'BTC_Stochastic_Delta_sqrt_none': 'BTC (Ours)', 'Baseline_RFC': 'RFC'})

summary = subset.groupby(['Dataset', 'Method'])['ARI'].agg(['mean', 'std']).reset_index()

km = pd.read_csv(os.path.join(RESULTS_DIR, 'kmeans_baselines.csv'))
km_sub = km[km['Method'].isin(['Baseline_KMeans', 'Baseline_PCA_KMeans', 'Baseline_UMAP_KMeans'])]
km_sum = km_sub.groupby(['Dataset', 'Method'])['ARI'].mean().reset_index()

tex = []
tex.append('\\begin{table}[htbp]')
tex.append('\\centering')
tex.append('\\caption{Clustering performance (Adjusted Rand Index) across benchmarks. Means and standard deviations computed over 30 random seeds.}')
tex.append('\\label{tab:main_results}')
tex.append('\\begin{tabular}{lccccc}')
tex.append('\\toprule')
tex.append('\\textbf{Dataset} & \\textbf{BTC (Ours)} & \\textbf{RFC} & \\textbf{K-Means} & \\textbf{PCA + K-Means} & \\textbf{UMAP + K-Means} \\\\')
tex.append('\\midrule')

for ds in DATASET_ORDER:
    if ds not in df['Dataset'].values: continue
    btc_row = summary[(summary['Dataset']==ds) & (summary['Method']=='BTC (Ours)')]
    rfc_row = summary[(summary['Dataset']==ds) & (summary['Method']=='RFC')]
    k_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_KMeans')]
    pk_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_PCA_KMeans')]
    u_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_UMAP_KMeans')]
    
    if btc_row.empty: continue
    
    btc_str = f"{btc_row['mean'].values[0]:.3f} \\pm {btc_row['std'].values[0]:.3f}"
    rfc_str = f"{rfc_row['mean'].values[0]:.3f} \\pm {rfc_row['std'].values[0]:.3f}" if not rfc_row.empty else '-'
    
    k_val = k_row['ARI'].values[0] if not k_row.empty else 0.0
    pk_val = pk_row['ARI'].values[0] if not pk_row.empty else 0.0
    u_val = u_row['ARI'].values[0] if not u_row.empty else 0.0
    
    if not rfc_row.empty and btc_row['mean'].values[0] > rfc_row['mean'].values[0]:
        btc_str = f"\\mathbf{{{btc_str}}}"
    elif not rfc_row.empty and rfc_row['mean'].values[0] > btc_row['mean'].values[0]:
        rfc_str = f"\\mathbf{{{rfc_str}}}"
        
    disp_name = DATASET_METADATA[ds]['display']
    tex.append(f"{disp_name} & ${btc_str}$ & ${rfc_str}$ & {k_val:.3f} & {pk_val:.3f} & {u_val:.3f} \\\\")

tex.append('\\bottomrule')
tex.append('\\end{tabular}')
tex.append('\\end{table}')

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(tex))
print(f'Table generated in {OUT_FILE}')

# Generate dataset summary table (Table 1)
ds_meta_df = df[['Dataset', 'N_Samples', 'N_Target', 'N_Features', 'K_Clusters']].drop_duplicates().set_index('Dataset')

tex_summary = [
    '\\begin{table}[htbp]',
    '\\centering',
    '\\caption{Summary of benchmark datasets. $N$ denotes the full training set size, $N_{\\text{target}}$ the target cohort size, $P$ the number of features, and $K$ the number of ground-truth subtypes.}',
    '\\label{tab:datasets}',
    '\\begin{tabular}{lrrrrl}',
    '\\toprule',
    '\\textbf{Dataset} & $N$ & $N_{\\text{target}}$ & $P$ & $K$ & \\textbf{Domain} \\\\',
    '\\midrule'
]

for ds_key in DATASET_ORDER:
    if ds_key in ds_meta_df.index:
        meta = DATASET_METADATA[ds_key]
        n_val = f"{ds_meta_df.loc[ds_key, 'N_Samples']:,}".replace(',', '{,}')
        nt_val = f"{ds_meta_df.loc[ds_key, 'N_Target']:,}".replace(',', '{,}')
        p_val = f"{ds_meta_df.loc[ds_key, 'N_Features']:,}".replace(',', '{,}')
        k_val = ds_meta_df.loc[ds_key, 'K_Clusters']
        tex_summary.append(f"{meta['display']:<14} & {n_val:<8} & {nt_val:<7} & {p_val:<7} & {k_val:<2} & {meta['domain']} \\\\")

tex_summary.extend([
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}'
])

SUMMARY_OUT_FILE = os.path.join(PAPER_DIR, 'tables', 'dataset_summary.tex')
with open(SUMMARY_OUT_FILE, 'w') as f:
    f.write('\n'.join(tex_summary) + '\n')
print(f'Table generated in {SUMMARY_OUT_FILE}')

# -------------------------------------------------------------
# Generate Detailed Dataset Cohort Breakdown Table (Appendix)
# -------------------------------------------------------------

DATASET_DESCRIPTIONS = {
    'thyroid': {
        'proxy_0': 'Normal thyroid',
        'proxy_1': 'Abnormal thyroid',
        'subtypes': ['Compensated Hypo', 'Primary Hypo', 'Secondary Hypo']
    },
    'brca': {
        'proxy_0': 'ER-Negative',
        'proxy_1': 'ER-Positive',
        'subtypes': ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B']
    },
    'paul15': {
        'proxy_0': 'Non-myeloid lineages',
        'proxy_1': 'Myeloid progenitors',
        'subtypes': ['Erythrocyte', 'Monocyte', 'Granulocyte', 'Megakaryocyte', 'Neutrophil', 'Progenitor']
    },
    'gbmlgg': {
        'proxy_0': 'Alive at follow-up',
        'proxy_1': 'Deceased',
        'subtypes': ['Grade II (LGG)', 'Grade III (LGG)', 'Grade IV (GBM)']
    },
    'ctg': {
        'proxy_0': 'Normal fetal state',
        'proxy_1': 'Abnormal / Pathologic',
        'subtypes': ['Suspect', 'Pathologic']
    },
    'mice': {
        'proxy_0': 'Control genotype',
        'proxy_1': 'Trisomic (Ts65Dn)',
        'subtypes': ['Trisomy + Saline', 'Trisomy + Memantine', 'Control + Saline', 'Control + Memantine']
    },
    'har': {
        'proxy_0': 'Static activities (stand/sit/lie)',
        'proxy_1': 'Dynamic activities (walking)',
        'subtypes': ['Walking', 'Walking Upstairs', 'Walking Downstairs']
    },
    'diabetes': {
        'proxy_0': 'No hospital readmission',
        'proxy_1': 'Hospital readmission',
        'subtypes': ['Circulatory (ICD 390--459)', 'Respiratory (ICD 460--519)', 'Digestive (ICD 520--579)', 'Diabetes complications (ICD 250)']
    },
    'dermatology': {
        'proxy_0': 'Psoriasis (majority)',
        'proxy_1': 'Other skin diseases',
        'subtypes': ['Seborrheic Keratosis', 'Lichen Planus', 'Pityriasis Rosea', 'Chronic Dermatitis', 'Pityriasis Rubra']
    },
    'vehicle': {
        'proxy_0': 'Opel silhouette',
        'proxy_1': 'Other vehicle silhouettes',
        'subtypes': ['Saab', 'Bus', 'Van']
    },
    'segment': {
        'proxy_0': 'Brickface segment',
        'proxy_1': 'Other image segments',
        'subtypes': ['Sky', 'Foliage', 'Cement', 'Window', 'Path', 'Grass']
    },
    'arrhythmia': {
        'proxy_0': 'Normal ECG (Class 1)',
        'proxy_1': 'Cardiac arrhythmia',
        'subtypes': [f'Type {i}' for i in range(1, 13)]
    },
    'leukemia': {
        'proxy_0': 'AML (non-lymphoblastic)',
        'proxy_1': 'ALL (lymphoblastic)',
        'subtypes': ['B-cell ALL', 'T-cell ALL']
    }
}

tex_breakdown = [
    '\\begin{table}[htbp]',
    '\\centering',
    '\\caption{Detailed architectural breakdown of benchmark datasets, proxy classification tasks, and ground-truth cluster distributions across proxy classes.}',
    '\\label{tab:app_datasets}',
    '\\small',
    '\\begin{tabular}{lp{2.2cm}p{4.5cm}p{5.5cm}}',
    '\\toprule',
    '\\textbf{Dataset} & \\textbf{Paradigm} & \\textbf{Proxy Classes ($y=0$ vs.\\ $y=1$)} & \\textbf{Target Clusters \\& Class Membership} \\\\',
    '\\midrule'
]

print("Generating detailed cohort breakdown table...")
for ds_key in DATASET_ORDER:
    meta = DATASET_METADATA[ds_key]
    try:
        X_full, y_binary, X_target, subtypes, k = load_dataset(ds_key)
        is_hier = (len(X_target) < len(X_full))
        paradigm = "Hierarchical" if is_hier else "Cross-Cutting"
        
        n0 = np.sum(y_binary == 0)
        n1 = np.sum(y_binary == 1)
        
        info = DATASET_DESCRIPTIONS.get(ds_key, {})
        p0_desc = info.get('proxy_0', 'Class 0')
        p1_desc = info.get('proxy_1', 'Class 1')
        sub_names = info.get('subtypes', [f'Subtype {i}' for i in range(k)])
        
        proxy_str = f"\\textbf{{y=0:}} {p0_desc} ({n0:,}) \\newline \\textbf{{y=1:}} {p1_desc} ({n1:,})".replace(',', '{,}')
        
        cluster_parts = []
        if is_hier:
            u_subs, counts = np.unique(subtypes, return_counts=True)
            for idx_sub, cnt in zip(u_subs, counts):
                s_name = sub_names[idx_sub] if idx_sub < len(sub_names) else f"Cluster {idx_sub}"
                cluster_parts.append(f"{s_name} ({cnt})")
            cluster_str = "\\textbf{100\\% in y=1:} " + ", ".join(cluster_parts) + " \\newline \\textit{(y=0 contains 0 clusters)}"
        else:
            ct = pd.crosstab(y_binary, subtypes)
            for idx_sub in range(k):
                s_name = sub_names[idx_sub] if idx_sub < len(sub_names) else f"Cluster {idx_sub}"
                c0 = ct.loc[0, idx_sub] if (0 in ct.index and idx_sub in ct.columns) else 0
                c1 = ct.loc[1, idx_sub] if (1 in ct.index and idx_sub in ct.columns) else 0
                cluster_parts.append(f"\\textbf{{{s_name}}}: {c0} in $y=0$, {c1} in $y=1$")
            cluster_str = "; \\newline ".join(cluster_parts)
            
        tex_breakdown.append(f"{meta['display']} & {paradigm} & {proxy_str} & {cluster_str} \\\\")
        tex_breakdown.append('\\midrule')
    except Exception as e:
        print(f"Error generating breakdown for {ds_key}: {e}")

if tex_breakdown[-1] == '\\midrule':
    tex_breakdown.pop()

tex_breakdown.extend([
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}'
])

BREAKDOWN_OUT_FILE = os.path.join(PAPER_DIR, 'tables', 'dataset_cohort_breakdown.tex')
with open(BREAKDOWN_OUT_FILE, 'w') as f:
    f.write('\n'.join(tex_breakdown) + '\n')
print(f'Table generated in {BREAKDOWN_OUT_FILE}')
