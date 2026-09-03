import os
import pickle
import numpy as np
import openml

def main():
    print("Fetching CTG dataset from OpenML (id 1560)...")
    dataset = openml.datasets.get_dataset(1560)
    X_df, y_series, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    
    # y: 1=Normal, 2=Suspect, 3=Pathologic
    y = y_series.astype(int).values
    
    # Binary task: 0 = Normal (1), 1 = Abnormal (2 or 3)
    binary_y = np.where(y == 1, 0, 1)
    
    X_raw = X_df.values.astype(float)
    
    # Target cohort: Abnormal
    target_mask = binary_y == 1
    X_target = X_raw[target_mask]
    # Subtypes: Suspect (2) vs Pathologic (3)
    subtypes = y[target_mask]
    
    # Create sample IDs
    sample_ids = [f"ctg_{i}" for i in range(len(X_raw))]
    target_sample_ids = [sample_ids[i] for i in range(len(X_raw)) if target_mask[i]]
    
    data = {
        'full_X_raw': X_raw,
        'full_y': binary_y,
        'full_sample_ids': sample_ids,
        'X': X_target,
        'X_raw': X_target,
        'subtypes': subtypes,
        'sample_ids': target_sample_ids,
        'feature_names': list(X_df.columns)
    }
    
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, 'ctg_preprocessed.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved preprocessed CTG data to {out_path}")
    print(f"Full dataset: {X_raw.shape[0]} samples")
    print(f"Binary classes: Normal={np.sum(binary_y == 0)}, Abnormal={np.sum(binary_y == 1)}")
    print(f"Target cohort: {X_target.shape[0]} samples")
    print(f"Subtypes: Suspect={np.sum(subtypes == 2)}, Pathologic={np.sum(subtypes == 3)}")

if __name__ == '__main__':
    main()
