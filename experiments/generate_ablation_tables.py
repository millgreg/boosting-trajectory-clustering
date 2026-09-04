import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_DIR = os.path.join(PAPER_DIR, 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Clustering Backend Ablation
try:
    df_adv = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_advanced_summary.csv'))
    df_adv['Method_Clean'] = df_adv['Method'].replace({
        'BTC_Base_per_tree_delta': 'K-Means (Default)',
        'BTC_GMM_per_tree_delta': 'GMM (Full Covariance)',
        'BTC_UMAP_2': 'UMAP (2D) + K-Means',
        'BTC_UMAP_5': 'UMAP (5D) + K-Means',
        'BTC_Base_cumul_logodds': 'K-Means (Log-Odds)',
        'BTC_GMM_cumul_logodds': 'GMM (Log-Odds)'
    })
    
    summary = df_adv.groupby('Method_Clean')['ARI_mean'].mean().reset_index().sort_values('ARI_mean', ascending=False)
    
    tex_backend = [
        '\\begin{table}[h]',
        '\\centering',
        '\\caption{Average ARI across canonical datasets for different clustering backends applied to the boosting trajectory.}',
        '\\label{tab:app_backend}',
        '\\begin{tabular}{lc}',
        '\\toprule',
        '\\textbf{Clustering Backend} & \\textbf{Mean ARI} \\\\',
        '\\midrule'
    ]
    for _, row in summary.iterrows():
        tex_backend.append(f"{row['Method_Clean']} & {row['ARI_mean']:.3f} \\\\")
    tex_backend.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    
    backend_file = os.path.join(OUT_DIR, 'ablation_backend.tex')
    with open(backend_file, 'w') as f:
        f.write('\n'.join(tex_backend) + '\n')
    print(f"Generated {backend_file}")
except Exception as e:
    print("Error processing advanced clustering:", e)

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

DISPLAY_NAMES = {
    'arrhythmia': 'Arrhythmia',
    'brca': 'BRCA',
    'ctg': 'CTG',
    'dermatology': 'Dermatology',
    'diabetes': 'Diabetes',
    'gbmlgg': 'GBMLGG',
    'har': 'HAR',
    'leukemia': 'Leukemia',
    'mice': 'Mice',
    'paul15': 'Paul15',
    'segment': 'Segment',
    'thyroid': 'Thyroid',
    'vehicle': 'Vehicle'
}

# 2. Depth Ablation
try:
    df_depth = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_auto_depth_summary.csv'))
    tex_depth = [
        '\\begin{table}[h]',
        '\\centering',
        '\\caption{Comparison of fixed Depth-1 stumps versus deeper trees (Depth 2, 3, 5) and an auto-selection heuristic.}',
        '\\label{tab:app_depth}',
        '\\begin{tabular}{lccccc}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{Depth-1} & \\textbf{Depth-2} & \\textbf{Depth-3} & \\textbf{Depth-5} & \\textbf{Auto-Select} \\\\',
        '\\midrule'
    ]
    
    datasets = [d for d in DATASET_ORDER if d in df_depth['Dataset'].unique()]
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
        
        disp = DISPLAY_NAMES.get(ds, ds.capitalize())
        tex_depth.append(f"{disp} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ & ${strs[4]}$ \\\\")
        
    tex_depth.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    
    depth_file = os.path.join(OUT_DIR, 'ablation_depth.tex')
    with open(depth_file, 'w') as f:
        f.write('\n'.join(tex_depth) + '\n')
    print(f"Generated {depth_file}")
except Exception as e:
    print("Error processing depth:", e)

# 3. Design Space (Stochasticity & Representation)
try:
    df_canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    methods = {
        'BTC_Base_Delta_none': 'No Subsampling (Delta)',
        'BTC_Stochastic_Delta_0.1_none': '10\\% Features (Delta)',
        'BTC_Stochastic_Delta_sqrt_none': '$\\sqrt{P}$ Features (Delta)',
        'BTC_Stochastic_Logodds_sqrt': '$\\sqrt{P}$ Features (Log-Odds)'
    }
    subset = df_canon[df_canon['Method'].isin(methods.keys())].copy()
    summary = subset.groupby(['Dataset', 'Method'])['ARI'].mean().reset_index()
    
    tex_design = [
        '\\begin{table}[h]',
        '\\centering',
        '\\caption{Impact of stochastic feature subsampling and trajectory representation type.}',
        '\\label{tab:app_design}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{No Subsampling} & \\textbf{10\\% Features} & \\textbf{$\\sqrt{P}$ (Delta)} & \\textbf{$\\sqrt{P}$ (Log-Odds)} \\\\',
        '\\midrule'
    ]
    
    datasets = [d for d in DATASET_ORDER if d in summary['Dataset'].unique()]
    for ds in datasets:
        ds_data = summary[summary['Dataset'] == ds]
        def get_val(m):
            v = ds_data[ds_data['Method'] == m]['ARI']
            return v.values[0] if len(v) > 0 else 0
            
        v1 = get_val('BTC_Base_Delta_none')
        v2 = get_val('BTC_Stochastic_Delta_0.1_none')
        v3 = get_val('BTC_Stochastic_Delta_sqrt_none')
        v4 = get_val('BTC_Stochastic_Logodds_sqrt')
        
        row_vals = [v1, v2, v3, v4]
        max_val = max(row_vals)
        strs = [f"\\mathbf{{{v:.3f}}}" if v == max_val else f"{v:.3f}" for v in row_vals]
        
        disp = DISPLAY_NAMES.get(ds, ds.capitalize())
        tex_design.append(f"{disp} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ \\\\")
        
    tex_design.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    
    design_file = os.path.join(OUT_DIR, 'ablation_design.tex')
    with open(design_file, 'w') as f:
        f.write('\n'.join(tex_design) + '\n')
    print(f"Generated {design_file}")
except Exception as e:
    print("Error processing design space:", e)
