import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PAPER_DIR = os.path.join(BASE_DIR, '..', 'paper')
OUT_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

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

def get_data():
    canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    km = pd.read_csv(os.path.join(RESULTS_DIR, 'kmeans_baselines.csv'))
    return pd.concat([canon, km], ignore_index=True)

def fig_main_results(df):
    print('Generating Fig 2: Main Results...')
    # Extract configs we want to compare
    methods_to_plot = {
        'BTC_Stochastic_Delta_sqrt_none': 'BTC',
        'Baseline_RFC': 'RFC',
        'Baseline_KMeans': 'K-Means',
        'Baseline_PCA_KMeans': 'PCA + K-Means',
        'Baseline_UMAP_KMeans': 'UMAP + K-Means'
    }
    
    subset = df[df['Method'].isin(methods_to_plot.keys())].copy()
    subset['Method Name'] = subset['Method'].map(methods_to_plot)
    subset['Dataset'] = subset['Dataset'].map(DISPLAY_NAMES).fillna(subset['Dataset'])
    
    # Calculate mean and std
    summary = subset.groupby(['Dataset', 'Method Name'])['ARI'].agg(['mean', 'std']).reset_index()
    
    # Sort datasets by BTC mean ARI
    btc_means = summary[summary['Method Name'] == 'BTC'].set_index('Dataset')['mean'].sort_values(ascending=False)
    order = btc_means.index.tolist()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Okabe-Ito / ColorBrewer colorblind-safe palette (5 methods matching Table 2):
    method_palette = {
        'BTC': '#009E73',           # Sea Green
        'RFC': '#0072B2',           # Royal Blue
        'K-Means': '#7570b3',       # Muted Purple
        'PCA + K-Means': '#E69F00', # Warm Amber
        'UMAP + K-Means': '#56B4E9' # Sky Blue
    }
    hue_order = ['BTC', 'RFC', 'K-Means', 'PCA + K-Means', 'UMAP + K-Means']
    
    sns.barplot(
        data=subset, 
        x='Dataset', y='ARI', hue='Method Name',
        order=order,
        hue_order=hue_order,
        errorbar='sd', ax=ax, err_kws={'linewidth': 1.2}, capsize=0.03,
        palette=method_palette
    )
    
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_xlabel('')
    ax.set_title('Clustering Performance Across Datasets')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_main_results.pdf'))
    plt.close()

def fig_timing(df):
    print('Generating Fig 3: Timing...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Scalability (log-log)
    try:
        scale_df = pd.read_csv(os.path.join(RESULTS_DIR, 'scalability_results.csv'))
        ax1.plot(scale_df['N'], scale_df['BTC_Time_s'], 'o-', label='BTC', color='#009E73', linewidth=2)
        ax1.plot(scale_df['N'], scale_df['Spectral_Time_s'], 's-', label='RFC (Spectral)', color='#0072B2', linewidth=2)
        ax1.plot(scale_df['N'], scale_df['Agglomerative_Time_s'], '^-', label='RFC (Agglomerative)', color='#E69F00', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Samples (N)')
        ax1.set_ylabel('Clustering Time (seconds)')
        ax1.set_title('Computational Scalability')
        ax1.legend()
    except Exception as e:
        print("Error with scalability data:", e)
        
    # Panel B: Speedup on real datasets
    btc_time = df[df['Method'] == 'BTC_Stochastic_Delta_sqrt_none'].groupby('Dataset')['Fit_Time_s'].mean()
    rfc_time = df[df['Method'] == 'Baseline_RFC'].groupby('Dataset')['Fit_Time_s'].mean()
    
    # Inner join to only get datasets that have both
    common_ds = btc_time.index.intersection(rfc_time.index)
    speedup = (rfc_time[common_ds] / btc_time[common_ds]).sort_values(ascending=False)
    speedup.index = [DISPLAY_NAMES.get(k, k) for k in speedup.index]
    
    sns.barplot(x=speedup.index, y=speedup.values, ax=ax2, color='#009E73')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('Speedup Factor (RFC time / BTC time)')
    ax2.set_title('Speedup over RFC on Benchmark Datasets')
    ax2.axhline(1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_timing.pdf'))
    plt.close()

def fig_depth_ablation():
    print('Generating Fig 4: Depth Ablation...')
    try:
        depth_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_auto_depth_summary.csv'))
        depth_map = {
            'Fixed_Depth_1': 'Depth-1 (Stump)',
            'Fixed_Depth_2': 'Depth-2',
            'Fixed_Depth_3': 'Depth-3',
            'Fixed_Depth_5': 'Depth-5'
        }
        depth_df = depth_df[depth_df['Method'].isin(depth_map.keys())].copy()
        depth_df['Tree Depth'] = depth_df['Method'].map(depth_map)
        depth_df['Dataset'] = depth_df['Dataset'].map(DISPLAY_NAMES).fillna(depth_df['Dataset'])
        
        # Sort by canonical dataset order
        order = [DISPLAY_NAMES[d] for d in DATASET_ORDER if DISPLAY_NAMES[d] in depth_df['Dataset'].unique()]
        
        # Okabe-Ito colorblind palette for 4 depth levels
        depth_palette = ['#0072B2', '#56B4E9', '#E69F00', '#D55E00']
        
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.barplot(
            data=depth_df, x='Dataset', y='ARI_mean', hue='Tree Depth',
            order=order, palette=depth_palette
        )
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Mean Adjusted Rand Index (ARI)')
        ax.set_xlabel('')
        ax.set_title('Impact of Tree Depth: Stumps vs Deeper Trees')
        plt.legend(title='')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_depth_ablation.pdf'))
        plt.close()
    except Exception as e:
        print("Error with depth ablation data:", e)

def fig_design_space(df):
    print('Generating Fig 5: Design Space...')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Stochasticity impact
    methods_stoch = {
        'BTC_Base_Delta_none': 'No Subsampling',
        'BTC_Stochastic_Delta_0.1_none': '10% Features',
        'BTC_Stochastic_Delta_sqrt_none': 'Sqrt Features'
    }
    subset_stoch = df[df['Method'].isin(methods_stoch.keys())].copy()
    subset_stoch['Config'] = subset_stoch['Method'].map(methods_stoch)
    
    # Average across all datasets
    sns.boxplot(data=subset_stoch, x='Config', y='ARI', hue='Config', legend=False, ax=ax1, order=['No Subsampling', '10% Features', 'Sqrt Features'], palette='Blues')
    ax1.set_title('Effect of Stochastic Feature Subsampling')
    ax1.set_ylabel('ARI (Distribution Across Datasets)')
    ax1.set_xlabel('')
    
    # 2. Representation Type
    methods_rep = {
        'BTC_Stochastic_Delta_sqrt_none': 'Per-Tree Delta',
        'BTC_Stochastic_Logodds_sqrt': 'Cumulative Log-Odds'
    }
    subset_rep = df[df['Method'].isin(methods_rep.keys())].copy()
    subset_rep['Config'] = subset_rep['Method'].map(methods_rep)
    
    sns.boxplot(data=subset_rep, x='Config', y='ARI', hue='Config', legend=False, ax=ax2, order=['Per-Tree Delta', 'Cumulative Log-Odds'], palette='Greens')
    ax2.set_title('Effect of Trajectory Representation')
    ax2.set_ylabel('ARI (Distribution Across Datasets)')
    ax2.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_design_space.pdf'))
    plt.close()

if __name__ == '__main__':
    df = get_data()
    fig_main_results(df)
    fig_timing(df)
    fig_depth_ablation()
    fig_design_space(df)
    print("Done generating all figures.")
