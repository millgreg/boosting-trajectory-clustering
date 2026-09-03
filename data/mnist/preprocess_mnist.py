import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_preprocess_mnist():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(base_dir, 'data', 'mnist')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MNIST from OpenML (this may take a minute)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X_full = mnist.data / 255.0  # Scale pixels to [0, 1]
    y_labels = mnist.target.astype(int)
    
    # Define Supervised Task: Even (1) vs Odd (0)
    y_binary = (y_labels % 2 == 0).astype(int)
    
    # We want to create two datasets:
    # 1. 10k subset (for full clustering benchmark with Spectral baseline)
    # 2. 70k full (for RTC scalability flex)
    
    print("Creating 2,500 sample subset...")
    # Stratify by true digits to ensure balanced subtypes
    X_2k, _, y_bin_2k, _, y_labels_2k, _ = train_test_split(
        X_full, y_binary, y_labels, train_size=2500, stratify=y_labels, random_state=42
    )
    
    def package_dataset(X, y_bin, labels):
        sample_ids = np.arange(len(X))
        
        # Target cohort = Even digits (y_bin == 1)
        target_mask = (y_bin == 1)
        X_target = X[target_mask]
        subtypes = labels[target_mask]
        target_sample_ids = sample_ids[target_mask]
        
        return {
            'X': X_target,
            'X_raw': X_target,
            'subtypes': subtypes,
            'sample_ids': target_sample_ids,
            'full_X_raw': X,
            'full_y': y_bin,
            'full_sample_ids': sample_ids
        }
    
    ds_2k = package_dataset(X_2k, y_bin_2k, y_labels_2k)
    ds_full = package_dataset(X_full, y_binary, y_labels)
    
    out_2k = os.path.join(data_dir, 'mnist_preprocessed.pkl')
    out_full = os.path.join(data_dir, 'mnist_full_preprocessed.pkl')
    
    with open(out_2k, 'wb') as f:
        pickle.dump(ds_2k, f)
        
    with open(out_full, 'wb') as f:
        pickle.dump(ds_full, f)
        
    print(f"Saved 2.5k subset to {out_2k}")
    print(f"  Total samples: {len(ds_2k['full_y'])}")
    print(f"  Target Even cohort (to cluster): {len(ds_2k['subtypes'])}")
    print(f"  Subtypes (True Even Digits): {np.unique(ds_2k['subtypes'])}")
    
    print(f"\nSaved Full 70k to {out_full}")
    print(f"  Total samples: {len(ds_full['full_y'])}")
    print(f"  Target Even cohort (to cluster): {len(ds_full['subtypes'])}")
    
if __name__ == '__main__':
    load_and_preprocess_mnist()
