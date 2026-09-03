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

RESULTS_DIR = r'c:\Users\gm10327\Projects\Miller_BoostingCluster\experiments\results'

def get_data():
    canon = pd.read_csv(os.path.join(RESULTS_DIR, 'canonical_results_30seeds.csv'))
    km = pd.read_csv(os.path.join(RESULTS_DIR, 'kmeans_baselines.csv'))
    return pd.concat([canon, km], ignore_index=True)

def fig_main_results(df):
    print('Generating Fig 2: Main Results...')
    # Extract configs we want to compare
    methods_to_plot = {
        'RTC_Stochastic_Delta_sqrt_none': 'RTC (Ours)',
        'Baseline_RFC': 'RFC',
        'Baseline_KMeans': 'K-Means',
        'Baseline_PCA_KMeans': 'PCA + K-Means',
        'Baseline_UMAP_KMeans': 'UMAP + K-Means'
    }
    
    subset = df[df['Method'].isin(methods_to_plot.keys())].copy()
    subset['Method Name'] = subset['Method'].map(methods_to_plot)
    
    # Calculate mean and std
    summary = subset.groupby(['Dataset', 'Method Name'])['ARI'].agg(['mean', 'std']).reset_index()
    
    # Sort datasets by RTC mean ARI
    rtc_means = summary[summary['Method Name'] == 'RTC (Ours)'].set_index('Dataset')['mean'].sort_values(ascending=False)
    order = rtc_means.index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=subset, 
        x='Dataset', y='ARI', hue='Method Name',
        order=order,
        hue_order=['RTC (Ours)', 'RFC', 'PCA + K-Means', 'K-Means'],
        errorbar='sd', ax=ax, errwidth=1.5, capsize=0.05,
        palette=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
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
        ax1.plot(scale_df['N'], scale_df['RTC_Time_s'], 'o-', label='RTC (Ours)', color='#2ca02c', linewidth=2)
        ax1.plot(scale_df['N'], scale_df['Spectral_Time_s'], 's-', label='RFC Spectral', color='#1f77b4', linewidth=2)
        ax1.plot(scale_df['N'], scale_df['Agglomerative_Time_s'], '^-', label='Agglomerative', color='#ff7f0e', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Samples (N)')
        ax1.set_ylabel('Clustering Time (seconds)')
        ax1.set_title('Computational Scalability')
        ax1.legend()
    except Exception as e:
        print("Error with scalability data:", e)
        
    # Panel B: Speedup on real datasets
    default_rtc = 'RTC_Stochastic_Delta_sqrt_none'
    rtc_time = df[df['Method'] == default_rtc].groupby('Dataset')['Fit_Time_s'].mean()
    rfc_time = df[df['Method'] == 'Baseline_RFC'].groupby('Dataset')['Fit_Time_s'].mean()
    
    # Inner join to only get datasets that have both
    common_ds = rtc_time.index.intersection(rfc_time.index)
    speedup = (rfc_time[common_ds] / rtc_time[common_ds]).sort_values(ascending=False)
    
    sns.barplot(x=speedup.index, y=speedup.values, ax=ax2, color='#2ca02c')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylabel('Speedup Factor (RFC time / RTC time)')
    ax2.set_title('Speedup over RFC on Benchmark Datasets')
    ax2.axhline(1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_timing.pdf'))
    plt.close()

def fig_depth_ablation():
    print('Generating Fig 4: Depth Ablation...')
    try:
        depth_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_auto_depth_summary.csv'))
        # Prepare data for plotting
        depth_df['Method Label'] = depth_df['Method'].apply(lambda x: 'Depth-1 Stumps' if 'Fixed_Depth_1' in x else 'Auto-Select Depth')
        
        # Sort by dataset
        order = depth_df['Dataset'].unique()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=depth_df, x='Dataset', y='ARI_mean', hue='Method Label',
            order=order, palette=['#2ca02c', '#d62728']
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
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
        'RTC_Base_Delta_none': 'No Subsampling',
        'RTC_Stochastic_Delta_0.1_none': '10% Features',
        'RTC_Stochastic_Delta_sqrt_none': 'Sqrt Features'
    }
    subset_stoch = df[df['Method'].isin(methods_stoch.keys())].copy()
    subset_stoch['Config'] = subset_stoch['Method'].map(methods_stoch)
    
    # Average across all datasets
    sns.boxplot(data=subset_stoch, x='Config', y='ARI', ax=ax1, order=['No Subsampling', '10% Features', 'Sqrt Features'], palette='Blues')
    ax1.set_title('Effect of Stochastic Feature Subsampling')
    ax1.set_ylabel('ARI (Distribution Across Datasets)')
    ax1.set_xlabel('')
    
    # 2. Representation Type
    methods_rep = {
        'RTC_Stochastic_Delta_sqrt_none': 'Per-Tree Delta',
        'RTC_Stochastic_Logodds_sqrt': 'Cumulative Log-Odds'
    }
    subset_rep = df[df['Method'].isin(methods_rep.keys())].copy()
    subset_rep['Config'] = subset_rep['Method'].map(methods_rep)
    
    sns.boxplot(data=subset_rep, x='Config', y='ARI', ax=ax2, order=['Per-Tree Delta', 'Cumulative Log-Odds'], palette='Greens')
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
