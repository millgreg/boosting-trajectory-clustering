import os
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc

def load_and_preprocess():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(base_dir, 'data', 'scrna')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading Paul15 scRNA-seq dataset...")
    # sc.datasets.paul15() downloads automatically
    adata = sc.datasets.paul15()
    
    # Extract data and labels
    X_full = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    y_str = adata.obs['paul15_clusters'].astype(str).values
    
    # Binary task: Erythroid vs Non-Erythroid
    # 'Ery' is in the string for Erythroid cells
    y_binary = np.array([1 if 'Ery' in s else 0 for s in y_str])
    
    print(f"Total cells: {len(y_binary)}")
    print(f"Erythroid cells: {np.sum(y_binary)}")
    
    # Target Cohort for subtyping: The true Erythroid cells
    target_mask = (y_binary == 1)
    X_target = X_full[target_mask]
    subtypes = y_str[target_mask]
    target_sample_ids = np.arange(len(X_full))[target_mask]
    
    ds = {
        'X': X_target,
        'X_raw': X_target,
        'subtypes': subtypes,
        'sample_ids': target_sample_ids,
        'full_X_raw': X_full,
        'full_y': y_binary,
        'full_sample_ids': np.arange(len(X_full))
    }
    
    out_file = os.path.join(data_dir, 'paul15_preprocessed.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(ds, f)
        
    print(f"Saved to {out_file}")
    print(f"  Total samples: {len(ds['full_y'])}")
    print(f"  Target cohort (Erythroid cells): {len(ds['subtypes'])}")
    print(f"  Subtypes (True Erythroid Clusters): {np.unique(ds['subtypes'])}")

if __name__ == '__main__':
    load_and_preprocess()
