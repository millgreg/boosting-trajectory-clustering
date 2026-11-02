import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_DIR = os.path.join(PAPER_DIR, 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

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

# 1. Depth Ablation
try:
    df_depth = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_auto_depth_summary.csv'))
    tex_depth = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Comparison of fixed Depth-1 decision stumps versus deeper trees (Depths 2, 3, 5). Values report the mean and standard deviation of the Adjusted Rand Index (ARI) computed over 30 independent random seeds; the best mean on each dataset is bolded.}',
        '\\label{tab:app_depth}',
        '\\small',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{Depth-1 (Stump)} & \\textbf{Depth-2} & \\textbf{Depth-3} & \\textbf{Depth-5} \\\\',
        '\\midrule'
    ]
    
    datasets = [d for d in DATASET_ORDER if d in df_depth['Dataset'].unique()]
    means_all = {'Fixed_Depth_1': [], 'Fixed_Depth_2': [], 'Fixed_Depth_3': [], 'Fixed_Depth_5': []}
    
    for ds in datasets:
        ds_data = df_depth[df_depth['Dataset'] == ds]
        
        row_vals = []
        for m in ['Fixed_Depth_1', 'Fixed_Depth_2', 'Fixed_Depth_3', 'Fixed_Depth_5']:
            sub = ds_data[ds_data['Method'] == m]
            mean_val = sub['ARI_mean'].values[0] if len(sub) > 0 else 0.0
            std_val = sub['ARI_std'].values[0] if len(sub) > 0 else 0.0
            row_vals.append((mean_val, std_val))
            means_all[m].append(mean_val)
            
        max_mean = max(v[0] for v in row_vals)
        strs = []
        for mean_val, std_val in row_vals:
            cell_str = f"{mean_val:.3f} \\pm {std_val:.3f}"
            if abs(mean_val - max_mean) < 1e-4:
                strs.append(f"\\mathbf{{{cell_str}}}")
            else:
                strs.append(cell_str)
                
        disp = DISPLAY_NAMES.get(ds, ds.capitalize())
        tex_depth.append(f"{disp} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ \\\\")
        
    avg_row = [np.mean(means_all[m]) for m in ['Fixed_Depth_1', 'Fixed_Depth_2', 'Fixed_Depth_3', 'Fixed_Depth_5']]
    max_avg = max(avg_row)
    avg_strs = [f"\\mathbf{{{v:.3f}}}" if abs(v - max_avg) < 1e-4 else f"{v:.3f}" for v in avg_row]
    
    tex_depth.extend([
        '\\midrule',
        f"Average & ${avg_strs[0]}$ & ${avg_strs[1]}$ & ${avg_strs[2]}$ & ${avg_strs[3]}$ \\\\",
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

# 2. Design Space (Stochasticity & Representation)
try:
    df_canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    methods_des = [
        ('BTC_Base_Delta_none', 'No Subsampling'),
        ('BTC_Stochastic_Delta_0.1_none', '10\\% Features'),
        ('BTC_Stochastic_Delta_sqrt_none', '$\\sqrt{P}$ (Delta)'),
        ('BTC_Stochastic_Logodds_sqrt', '$\\sqrt{P}$ (Log-Odds)')
    ]
    sub_des = df_canon[df_canon['Method'].isin([m[0] for m in methods_des])]
    des_agg = sub_des.groupby(['Dataset', 'Method'])['ARI'].agg(['mean', 'std']).reset_index()
    
    tex_design = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Impact of stochastic feature subsampling and trajectory representation type. Values report the mean and standard deviation of the Adjusted Rand Index (ARI) computed over 30 independent random seeds; the best mean on each dataset is bolded.}',
        '\\label{tab:app_design}',
        '\\small',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{No Subsampling} & \\textbf{10\\% Features} & \\textbf{$\\sqrt{P}$ (Delta)} & \\textbf{$\\sqrt{P}$ (Log-Odds)} \\\\',
        '\\midrule'
    ]
    
    datasets = [d for d in DATASET_ORDER if d in des_agg['Dataset'].unique()]
    means_all_des = {m[0]: [] for m in methods_des}
    
    for ds in datasets:
        ds_data = des_agg[des_agg['Dataset'] == ds]
        row_vals = []
        for m_key, _ in methods_des:
            sub = ds_data[ds_data['Method'] == m_key]
            mean_val = sub['mean'].values[0] if len(sub) > 0 else 0.0
            std_val = sub['std'].values[0] if len(sub) > 0 else 0.0
            row_vals.append((mean_val, std_val))
            means_all_des[m_key].append(mean_val)
            
        max_mean = max(v[0] for v in row_vals)
        strs = []
        for mean_val, std_val in row_vals:
            cell_str = f"{mean_val:.3f} \\pm {std_val:.3f}"
            if abs(mean_val - max_mean) < 1e-4:
                strs.append(f"\\mathbf{{{cell_str}}}")
            else:
                strs.append(cell_str)
                
        disp = DISPLAY_NAMES.get(ds, ds.capitalize())
        tex_design.append(f"{disp} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ \\\\")
        
    avg_row_des = [np.mean(means_all_des[m[0]]) for m in methods_des]
    max_avg_des = max(avg_row_des)
    avg_strs_des = [f"\\mathbf{{{v:.3f}}}" if abs(v - max_avg_des) < 1e-4 else f"{v:.3f}" for v in avg_row_des]
    
    tex_design.extend([
        '\\midrule',
        f"Average & ${avg_strs_des[0]}$ & ${avg_strs_des[1]}$ & ${avg_strs_des[2]}$ & ${avg_strs_des[3]}$ \\\\",
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

# 3. Clustering Backend Ablation (Option B: Full Dataset Breakdown)
try:
    df_back = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_backend_results.csv'))
    methods_back = [
        ('KMeans', '$K$-Means (Default)'),
        ('Agglomerative', 'Agglomerative (Ward)'),
        ('Spectral', 'Spectral (RBF)'),
        ('GMM', 'GMM')
    ]
    back_agg = df_back.groupby(['Dataset', 'Method'])['ARI'].agg(['mean', 'std']).reset_index()
    
    tex_backend = [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Clustering performance (Adjusted Rand Index) across all 13 benchmark datasets for different clustering backends applied to the boosting trajectory matrix $R$. Values report the mean and standard deviation computed over 30 independent random seeds; the best mean on each dataset is bolded.}',
        '\\label{tab:app_backend}',
        '\\small',
        '\\setlength{\\tabcolsep}{5pt}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Dataset} & \\textbf{$K$-Means (Default)} & \\textbf{Agglomerative (Ward)} & \\textbf{Spectral (RBF)} & \\textbf{GMM} \\\\',
        '\\midrule'
    ]
    
    datasets = [d for d in DATASET_ORDER if d in back_agg['Dataset'].unique()]
    means_all_back = {m[0]: [] for m in methods_back}
    
    for ds in datasets:
        ds_data = back_agg[back_agg['Dataset'] == ds]
        row_vals = []
        for m_key, _ in methods_back:
            sub = ds_data[ds_data['Method'] == m_key]
            mean_val = sub['mean'].values[0] if len(sub) > 0 else 0.0
            std_val = sub['std'].values[0] if len(sub) > 0 else 0.0
            row_vals.append((mean_val, std_val))
            means_all_back[m_key].append(mean_val)
            
        max_mean = max(v[0] for v in row_vals)
        strs = []
        for mean_val, std_val in row_vals:
            cell_str = f"{mean_val:.3f} \\pm {std_val:.3f}"
            if abs(mean_val - max_mean) < 1e-4:
                strs.append(f"\\mathbf{{{cell_str}}}")
            else:
                strs.append(cell_str)
                
        disp = DISPLAY_NAMES.get(ds, ds.capitalize())
        tex_backend.append(f"{disp} & ${strs[0]}$ & ${strs[1]}$ & ${strs[2]}$ & ${strs[3]}$ \\\\")
        
    avg_row_back = [np.mean(means_all_back[m[0]]) for m in methods_back]
    max_avg_back = max(avg_row_back)
    avg_strs_back = [f"\\mathbf{{{v:.3f}}}" if abs(v - max_avg_back) < 1e-4 else f"{v:.3f}" for v in avg_row_back]
    
    tex_backend.extend([
        '\\midrule',
        f"Average & ${avg_strs_back[0]}$ & ${avg_strs_back[1]}$ & ${avg_strs_back[2]}$ & ${avg_strs_back[3]}$ \\\\",
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    
    backend_file = os.path.join(OUT_DIR, 'ablation_backend.tex')
    with open(backend_file, 'w') as f:
        f.write('\n'.join(tex_backend) + '\n')
    print(f"Generated {backend_file}")
except Exception as e:
    print("Error processing backend:", e)

