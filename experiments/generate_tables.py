import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_FILE = os.path.join(PAPER_DIR, 'tables', 'main_results.tex')
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
subset = df[df['Method'].isin(['RTC_Stochastic_Delta_sqrt_none', 'Baseline_RFC'])].copy()
subset['Method'] = subset['Method'].replace({'RTC_Stochastic_Delta_sqrt_none': 'RTC (Ours)', 'Baseline_RFC': 'RFC'})

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
tex.append('\\textbf{Dataset} & \\textbf{RTC (Ours)} & \\textbf{RFC} & \\textbf{K-Means} & \\textbf{PCA + K-Means} & \\textbf{UMAP + K-Means} \\\\')
tex.append('\\midrule')

for ds in df['Dataset'].unique():
    rtc_row = summary[(summary['Dataset']==ds) & (summary['Method']=='RTC (Ours)')]
    rfc_row = summary[(summary['Dataset']==ds) & (summary['Method']=='RFC')]
    k_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_KMeans')]
    pk_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_PCA_KMeans')]
    u_row = km_sum[(km_sum['Dataset']==ds) & (km_sum['Method']=='Baseline_UMAP_KMeans')]
    
    if rtc_row.empty: continue
    
    rtc_str = f"{rtc_row['mean'].values[0]:.3f} \\pm {rtc_row['std'].values[0]:.3f}"
    rfc_str = f"{rfc_row['mean'].values[0]:.3f} \\pm {rfc_row['std'].values[0]:.3f}" if not rfc_row.empty else '-'
    
    k_val = k_row['ARI'].values[0] if not k_row.empty else 0.0
    pk_val = pk_row['ARI'].values[0] if not pk_row.empty else 0.0
    u_val = u_row['ARI'].values[0] if not u_row.empty else 0.0
    
    if not rfc_row.empty and rtc_row['mean'].values[0] > rfc_row['mean'].values[0]:
        rtc_str = f"\\mathbf{{{rtc_str}}}"
    elif not rfc_row.empty and rfc_row['mean'].values[0] > rtc_row['mean'].values[0]:
        rfc_str = f"\\mathbf{{{rfc_str}}}"
        
    tex.append(f"{ds.capitalize()} & ${rtc_str}$ & ${rfc_str}$ & {k_val:.3f} & {pk_val:.3f} & {u_val:.3f} \\\\")

tex.append('\\bottomrule')
tex.append('\\end{tabular}')
tex.append('\\end{table}')

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(tex))
print(f'Table generated in {OUT_FILE}')
