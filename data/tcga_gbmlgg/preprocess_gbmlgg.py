import os
import gzip
import json
import pickle
import urllib.request
import pandas as pd
import numpy as np

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"File {filepath} already exists. Skipping download.")

def main():
    data_dir = os.path.abspath(os.path.dirname(__file__))
    expr_url = 'https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/HiSeqV2.gz'
    clin_url = 'https://tcga.xenahubs.net/download/TCGA.GBMLGG.sampleMap/GBMLGG_clinicalMatrix'
    
    expr_path = os.path.join(data_dir, 'HiSeqV2.gz')
    clin_path = os.path.join(data_dir, 'GBMLGG_clinicalMatrix')
    
    download_file(expr_url, expr_path)
    download_file(clin_url, clin_path)
    
    print("Loading clinical data...")
    clin_df = pd.read_csv(clin_path, sep='\t')
    
    # Extract relevant columns
    # vital_status: "Alive" or "Dead"
    # histological_type: "Glioblastoma Multiforme (GBM)" -> Grade IV
    # neoplasm_histologic_grade: "G2" or "G3"
    
    clin_df = clin_df[['sampleID', 'vital_status', 'histological_type', 'neoplasm_histologic_grade']].dropna(subset=['vital_status'])
    
    # Map Vital Status to Binary
    clin_df['y'] = clin_df['vital_status'].map({'Alive': 0, 'Dead': 1, 'LIVING': 0, 'DECEASED': 1})
    clin_df = clin_df.dropna(subset=['y'])
    clin_df['y'] = clin_df['y'].astype(int)
    
    # Map to Grade II, Grade III, Grade IV
    def get_grade(row):
        ht = str(row.get('histological_type', ''))
        g = str(row.get('neoplasm_histologic_grade', ''))
        if 'GBM' in ht or 'Glioblastoma' in ht:
            return 3  # Grade IV 
        if g == 'G3':
            return 2  # Grade III
        if g == 'G2':
            return 1  # Grade II
        return np.nan
        
    clin_df['Grade'] = clin_df.apply(get_grade, axis=1)
    clin_df = clin_df.dropna(subset=['Grade'])
    clin_df['Grade'] = clin_df['Grade'].astype(int) - 1 # 0: Grade II, 1: Grade III, 2: Grade IV
    
    print("Loading expression data... (This may take a minute or two)")
    expr_df = pd.read_csv(expr_path, sep='\t', index_col=0)
    
    # Xena expression data has genes as rows and samples as columns.
    # Transpose to get samples as rows
    expr_df = expr_df.T
    expr_df.index.name = 'sampleID'
    
    print("Merging data...")
    merged_df = expr_df.merge(clin_df, on='sampleID', how='inner')
    print(f"Matched {len(merged_df)} samples.")
    
    # Extract X, y (vital status), and subtypes (grade)
    y = merged_df['y'].values
    subtypes = merged_df['Grade'].values
    sample_ids = merged_df['sampleID'].values
    
    # Expression features are all columns except the clinical ones
    feature_cols = [c for c in merged_df.columns if c not in ['sampleID', 'vital_status', 'histological_type', 'neoplasm_histologic_grade', 'y', 'Grade']]
    X = merged_df[feature_cols].values
    
    # Fill any NaNs in expression with 0 (standard practice for log2(x+1) RSEM data)
    X = np.nan_to_num(X)
    
    # Select the top 2000 most highly variable genes to speed up baselines
    print("Selecting highly variable genes...")
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-2000:]
    X_top = X[:, top_indices]
    gene_names = [feature_cols[i] for i in top_indices]
    
    output_dict = {
        'X': X_top,
        'y': y,
        'subtypes': subtypes,
        'feature_names': gene_names,
        'sample_ids': sample_ids,
        'subtype_names': ['Grade II', 'Grade III', 'Grade IV (GBM)']
    }
    
    out_path = os.path.join(data_dir, 'gbmlgg_preprocessed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print(f"Successfully saved {len(sample_ids)} samples and {X_top.shape[1]} features to {out_path}.")
    print(f"Subtype distribution: {dict(zip(output_dict['subtype_names'], np.bincount(subtypes)))}")

if __name__ == '__main__':
    main()
