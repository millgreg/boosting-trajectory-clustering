import os
import gzip
import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"File {filepath} already exists. Skipping download.")

def main():
    data_dir = os.path.abspath(os.path.dirname(__file__))
    expr_url = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz'
    clin_url = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix'
    
    expr_path = os.path.join(data_dir, 'HiSeqV2.gz')
    clin_path = os.path.join(data_dir, 'BRCA_clinicalMatrix')
    
    download_file(expr_url, expr_path)
    download_file(clin_url, clin_path)
    
    print("Loading clinical data...")
    clin_df = pd.read_csv(clin_path, sep='\t')
    
    # Binary proxy task: ER Status
    clin_df['y'] = clin_df['breast_carcinoma_estrogen_receptor_status'].map({'Positive': 1, 'Negative': 0})
    
    # Target subtypes: PAM50 molecular subtypes
    clin_df['subtypes'] = clin_df['PAM50Call_RNAseq'].replace({'Her2': 'HER2'})
    
    # Filter to samples with valid ER status and one of the 4 PAM50 subtypes
    valid_mask = clin_df['y'].notna() & clin_df['subtypes'].isin(['Basal', 'HER2', 'LumA', 'LumB'])
    clin_filtered = clin_df[valid_mask].copy().reset_index(drop=True)
    print(f"Filtered to {len(clin_filtered)} clinical samples with ER status and PAM50 labels.")
    
    # Survival metadata
    days_death = pd.to_numeric(clin_filtered.get('days_to_death', pd.Series()), errors='coerce')
    days_follow = pd.to_numeric(clin_filtered.get('days_to_last_followup', pd.Series()), errors='coerce')
    survival_time = days_death.fillna(days_follow).values
    vital_status = clin_filtered.get('vital_status', pd.Series()).map({'DECEASED': 1, 'LIVING': 0}).fillna(0).values
    
    print("Loading HiSeqV2 expression data (this may take a minute)...")
    expr_df = pd.read_csv(expr_path, sep='\t', index_col=0)
    
    # Align samples
    common_samples = [s for s in clin_filtered['sampleID'] if s in expr_df.columns]
    print(f"Matched {len(common_samples)} samples in expression matrix.")
    
    clin_matched = clin_filtered.set_index('sampleID').loc[common_samples].reset_index()
    expr_matched = expr_df[common_samples]
    
    # Select top 2000 most variable genes
    print("Selecting top 2000 most variable genes...")
    gene_variances = expr_matched.var(axis=1)
    top_genes = gene_variances.sort_values(ascending=False).head(2000).index.tolist()
    
    # Extract expressions: (n_samples, 2000)
    X_raw = expr_matched.loc[top_genes].T.values.astype(float)
    X_raw = np.nan_to_num(X_raw)
    
    # Standardize features per gene across cohort
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    y = clin_matched['y'].values.astype(int)
    subtypes = clin_matched['subtypes'].values
    sample_ids = clin_matched['sampleID'].values
    
    output_dict = {
        'X': X_scaled,
        'X_raw': X_raw,
        'y': y,
        'subtypes': subtypes,
        'sample_ids': sample_ids,
        'gene_names': top_genes,
        'survival_time': survival_time,
        'survival_status': vital_status,
        'full_X_raw': X_raw,
        'full_y': y,
        'full_sample_ids': sample_ids
    }
    
    out_path = os.path.join(data_dir, 'brca_preprocessed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print(f"Successfully saved {len(sample_ids)} BRCA samples and {X_raw.shape[1]} genes to {out_path}.")
    print(f"Subtype distribution: {dict(zip(*np.unique(subtypes, return_counts=True)))}")
    print(f"Proxy task distribution (ER- / ER+): {dict(zip(*np.unique(y, return_counts=True)))}")

if __name__ == '__main__':
    main()
