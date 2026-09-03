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
    expr_url = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz'
    clin_url = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix'
    
    expr_path = os.path.join(data_dir, 'HiSeqV2.gz')
    clin_path = os.path.join(data_dir, 'BRCA_clinicalMatrix')
    
    download_file(expr_url, expr_path)
    download_file(clin_url, clin_path)
    
    print("Loading clinical data...")
    clin_df = pd.read_csv(clin_path, sep='\t')
    
    if 'ER_Status_nature2012' in clin_df.columns:
        clin_df['y'] = clin_df['ER_Status_nature2012'].map({'Positive': 1, 'Negative': 0})
    else:
        clin_df['y'] = clin_df['breast_carcinoma_estrogen_receptor_status'].map({'Positive': 1, 'Negative': 0})
        
    clin_df = clin_df.dropna(subset=['y'])
    clin_df['y'] = clin_df['y'].astype(int)
    
    clin_df['subtypes'] = clin_df['PAM50Call_RNAseq']
    target_df = clin_df[clin_df['y'] == 1].dropna(subset=['subtypes'])
    
    print("Loading expression data... (This may take a minute or two)")
    expr_df = pd.read_csv(expr_path, sep='\t', index_col=0).T
    expr_df.index.name = 'sampleID'
    
    print("Merging data...")
    merged_full = expr_df.merge(clin_df[['sampleID', 'y']], on='sampleID', how='inner')
    full_y = merged_full['y'].values
    full_sample_ids = merged_full['sampleID'].values
    full_X = merged_full.drop(columns=['sampleID', 'y']).values
    full_X = np.nan_to_num(full_X)
    
    merged_target = expr_df.merge(target_df[['sampleID', 'subtypes']], on='sampleID', how='inner')
    target_subtypes = merged_target['subtypes'].values
    target_sample_ids = merged_target['sampleID'].values
    target_X = merged_target.drop(columns=['sampleID', 'subtypes']).values
    target_X = np.nan_to_num(target_X)
    
    print("Selecting highly variable genes...")
    variances = np.var(target_X, axis=0)
    top_indices = np.argsort(variances)[-2000:]
    
    output_dict = {
        'full_X_raw': full_X[:, top_indices],
        'full_y': full_y,
        'full_sample_ids': full_sample_ids,
        'X': target_X[:, top_indices],
        'sample_ids': target_sample_ids,
        'subtypes': target_subtypes
    }
    
    out_path = os.path.join(data_dir, 'brca_preprocessed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print(f"Successfully saved {len(target_sample_ids)} target samples to {out_path}.")
    
if __name__ == '__main__':
    main()
