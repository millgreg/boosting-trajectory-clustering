import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_DIR = os.path.join(PAPER_DIR, 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

tex_appendix = []
tex_appendix.append('\\appendix')
tex_appendix.append('\\section{Detailed Ablation Results}')
tex_appendix.append('This appendix provides the full tabular results for the ablation studies summarized in Section 5.3.')

# 1. Clustering Backend Ablation
try:
    df_adv = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_advanced_summary.csv'))
    # Filter to relevant backends
    # Method names: RTC_Base_per_tree_delta, RTC_GMM_per_tree_delta, RTC_UMAP_2, etc.
    df_adv['Method_Clean'] = df_adv['Method'].replace({
        'RTC_Base_per_tree_delta': 'K-Means (Default)',
        'RTC_GMM_per_tree_delta': 'GMM (Full Covariance)',
        'RTC_UMAP_2': 'UMAP (2D) + K-Means',
        'RTC_UMAP_5': 'UMAP (5D) + K-Means',
        'RTC_Base_cumul_logodds': 'K-Means (Log-Odds)',
        'RTC_GMM_cumul_logodds': 'GMM (Log-Odds)'
    })
    
    summary = df_adv.groupby('Method_Clean')['ARI_mean'].mean().reset_index().sort_values('ARI_mean', ascending=False)
    
    tex_appendix.append('\n\\subsection{Clustering Backend}')
    tex_appendix.append('\\begin{table}[h]')
    tex_appendix.append('\\centering')
    tex_appendix.append('\\caption{Average ARI across canonical datasets for different clustering backends applied to the boosting trajectory.}')
    tex_appendix.append('\\begin{tabular}{lc}')
    tex_appendix.append('\\toprule')
    tex_appendix.append('\\textbf{Clustering Backend} & \\textbf{Mean ARI} \\\\')
    tex_appendix.append('\\midrule')
    for _, row in summary.iterrows():
        tex_appendix.append(f"{row['Method_Clean']} & {row['ARI_mean']:.3f} \\\\")
    tex_appendix.append('\\bottomrule')
    tex_appendix.append('\\end{tabular}')
    tex_appendix.append('\\end{table}')
except Exception as e:
    print("Error processing advanced clustering:", e)

# 2. Depth Ablation
try:
    df_depth = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_auto_depth_summary.csv'))
    tex_appendix.append('\n\\subsection{Tree Depth}')
    tex_appendix.append('\\begin{table}[h]')
    tex_appendix.append('\\centering')
    tex_appendix.append('\\caption{Comparison of fixed Depth-1 stumps versus deeper trees (Depth 2, 3, 5) and an auto-selection heuristic.}')
    tex_appendix.append('\\begin{tabular}{lccccc}')
    tex_appendix.append('\\toprule')
    tex_appendix.append('\\textbf{Dataset} & \\textbf{Depth-1} & \\textbf{Depth-2} & \\textbf{Depth-3} & \\textbf{Depth-5} & \\textbf{Auto-Select} \\\\')
    tex_appendix.append('\\midrule')
    
    datasets = df_depth['Dataset'].unique()
    for ds in datasets:
        ds_data = df_depth[df_depth['Dataset'] == ds]
        d1 = ds_data[ds_data['Method'] == 'Fixed_Depth_1']['ARI_mean'].values[0] if len(ds_data[ds_data['Method'] == 'Fixed_Depth_1']) > 0 else 0
        d2 = ds_data[ds_data['Method'] == 'Fixed_Depth_2']['ARI_mean'].values[0] if len(ds_data[ds_data['Method'] == 'Fixed_Depth_2']) > 0 else 0
        d3 = ds_data[ds_data['Method'] == 'Fixed_Depth_3']['ARI_mean'].values[0] if len(ds_data[ds_data['Method'] == 'Fixed_Depth_3']) > 0 else 0
        d5 = ds_data[ds_data['Method'] == 'Fixed_Depth_5']['ARI_mean'].values[0] if len(ds_data[ds_data['Method'] == 'Fixed_Depth_5']) > 0 else 0
        dauto = ds_data[ds_data['Method'] == 'Auto_Select']['ARI_mean'].values[0] if len(ds_data[ds_data['Method'] == 'Auto_Select']) > 0 else 0
        
        row_vals = [d1, d2, d3, d5, dauto]
        max_val = max(row_vals)
        strs = [f"\\mathbf{{{v:.3f}}}" if v == max_val else f"{v:.3f}" for v in row_vals]
        
        tex_appendix.append(f"{ds.capitalize()} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ & ${strs[4]}$ \\\\")
        
    tex_appendix.append('\\bottomrule')
    tex_appendix.append('\\end{tabular}')
    tex_appendix.append('\\end{table}')
except Exception as e:
    print("Error processing depth:", e)

# 3. Design Space (Stochasticity & Representation)
try:
    df_canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    methods = {
        'RTC_Base_Delta_none': 'No Subsampling (Delta)',
        'RTC_Stochastic_Delta_0.1_none': '10\\% Features (Delta)',
        'RTC_Stochastic_Delta_sqrt_none': '$\\sqrt{P}$ Features (Delta)',
        'RTC_Stochastic_Logodds_sqrt': '$\\sqrt{P}$ Features (Log-Odds)'
    }
    subset = df_canon[df_canon['Method'].isin(methods.keys())].copy()
    summary = subset.groupby(['Dataset', 'Method'])['ARI'].mean().reset_index()
    
    tex_appendix.append('\n\\subsection{Stochastic Subsampling and Representation}')
    tex_appendix.append('\\begin{table}[h]')
    tex_appendix.append('\\centering')
    tex_appendix.append('\\caption{Impact of stochastic feature subsampling and trajectory representation type.}')
    tex_appendix.append('\\begin{tabular}{lcccc}')
    tex_appendix.append('\\toprule')
    tex_appendix.append('\\textbf{Dataset} & \\textbf{No Subsampling} & \\textbf{10\\% Features} & \\textbf{$\\sqrt{P}$ (Delta)} & \\textbf{$\\sqrt{P}$ (Log-Odds)} \\\\')
    tex_appendix.append('\\midrule')
    
    for ds in summary['Dataset'].unique():
        ds_data = summary[summary['Dataset'] == ds]
        def get_val(m):
            v = ds_data[ds_data['Method'] == m]['ARI']
            return v.values[0] if len(v) > 0 else 0
            
        v1 = get_val('RTC_Base_Delta_none')
        v2 = get_val('RTC_Stochastic_Delta_0.1_none')
        v3 = get_val('RTC_Stochastic_Delta_sqrt_none')
        v4 = get_val('RTC_Stochastic_Logodds_sqrt')
        
        row_vals = [v1, v2, v3, v4]
        max_val = max(row_vals)
        strs = [f"\\mathbf{{{v:.3f}}}" if v == max_val else f"{v:.3f}" for v in row_vals]
        
        tex_appendix.append(f"{ds.capitalize()} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ \\\\")
        
    tex_appendix.append('\\bottomrule')
    tex_appendix.append('\\end{tabular}')
    tex_appendix.append('\\end{table}')
except Exception as e:
    print("Error processing design space:", e)

with open('paper/appendix.tex', 'w') as f:
    f.write('\n'.join(tex_appendix))
print("Generated appendix.tex with full tables.")
