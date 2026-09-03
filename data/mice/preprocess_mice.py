import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(base_dir, 'data', 'mice')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Fetching Mice Protein Expression dataset...")
    data = fetch_openml('miceprotein', version=4, parser='auto')
    
    X_raw = data.data.values.astype(float)
    y_str = data.target.astype(str).values
    
    # Impute missing values with mean
    imp = SimpleImputer(strategy='mean')
    X_imputed = imp.fit_transform(X_raw)
    
    # The classes are composed of: Genotype - Behavior - Treatment
    # E.g., 'c-CS-m', 't-CS-m', etc. 'c' = Control, 't' = Trisomic.
    # Supervised Binary Task: Trisomic (1) vs Control (0)
    y_binary = np.array([1 if val.startswith('t') else 0 for val in y_str])
    
    # Subtyping Task: Subtypes within Trisomic (combinations of Behavior and Treatment)
    target_mask = (y_binary == 1)
    
    # Generate mock sample IDs
    full_sample_ids = [f"MICE_{i}" for i in range(len(y_binary))]
    sample_ids = [full_sample_ids[i] for i, mask in enumerate(target_mask) if mask]
    
    # Keep original strings for the subtypes
    subtypes = y_str[target_mask]
    
    out_dict = {
        'full_X_raw': X_imputed,
        'full_y': y_binary,
        'full_sample_ids': full_sample_ids,
        'sample_ids': sample_ids,
        'subtypes': subtypes
    }
    
    out_path = os.path.join(out_dir, 'mice_preprocessed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out_dict, f)
        
    print(f"Saved {len(full_sample_ids)} total samples.")
    print(f"Target 'Trisomic' samples: {len(sample_ids)}")
    print(f"Subtypes in target: {np.unique(subtypes)}")

if __name__ == '__main__':
    main()
