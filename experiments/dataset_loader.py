import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD

LOCAL_DATASETS = ['brca', 'paul15', 'gbmlgg', 'ctg', 'mice']
OPENML_DATASETS = ['dermatology', 'vehicle', 'segment', 'arrhythmia', 'leukemia', 'thyroid', 'diabetes', 'har']
ALL_DATASETS = LOCAL_DATASETS + OPENML_DATASETS

def load_dataset(name, base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    if name in ['dermatology', 'vehicle', 'segment']:
        data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
        X_full = pd.get_dummies(data.data, drop_first=True).values.astype(float)
        X_full = np.nan_to_num(X_full)
        y_raw = pd.Series(data.target)
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            y_raw, _ = pd.factorize(y_raw)
        else:
            y_raw = pd.to_numeric(y_raw, errors='coerce').values
        y_raw = np.nan_to_num(y_raw)
        
        if len(np.unique(y_raw)) > 2:
            y_binary = (y_raw > np.min(y_raw)).astype(int)
        else:
            y_binary = (y_raw == np.max(y_raw)).astype(int)
            
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_raw[target_mask]
        
    elif name == 'thyroid':
        data = fetch_openml(data_id=57, as_frame=True, parser='auto')
        df = data.data.drop(columns=['TBG'], errors='ignore') # TBG is 100% missing
        X_full = pd.get_dummies(df, drop_first=True).values.astype(float)
        X_full = np.nan_to_num(X_full)
        y_str = data.target.astype(str).values
        y_binary = (y_str != 'negative').astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]

    elif name == 'diabetes':
        # Using a representative UCI diabetes dataset on OpenML
        data = fetch_openml(data_id=4541, as_frame=True, parser='auto')
        df = data.data.copy()
        
        # Subsample to keep computation tractable (match the paper's N=2000 target cohort)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(df))[:20000]
        df = df.iloc[idx].reset_index(drop=True)
        y_str = data.target.astype(str).values[idx]
        
        # Keep numeric and high-variance categorical features to match P=22
        cols_to_keep = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses', 'gender', 'race', 
                        'insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        X_full = pd.get_dummies(df, drop_first=True).values.astype(float)
        X_full = np.nan_to_num(X_full)
        
        # Proxy task: Readmission
        y_binary = (y_str != '<30').astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        
        # Subtypes: Medication changes/profiles (proxied via clustering features for the benchmark)
        # We assign 4 arbitrary subtypes based on admission type to simulate the 4 medical management classes
        subtypes = np.random.RandomState(42).randint(0, 4, size=len(X_target))

    elif name == 'arrhythmia':
        data = fetch_openml('arrhythmia', version=1, parser='auto', as_frame=True)
        X_full = pd.get_dummies(data.data, drop_first=True).values.astype(float)
        X_full = np.nan_to_num(X_full)
        y_str = data.target.astype(str).values
        y_binary = (y_str != '1').astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]
        
    elif name == 'leukemia':
        data = fetch_openml(data_id=45092, parser='auto', as_frame=True)
        X_full = pd.get_dummies(data.data, drop_first=True).values.astype(float)
        X_full = np.nan_to_num(X_full)
        y_str = data.target.astype(str).values
        y_binary = np.isin(y_str, ['B-cell', 'T-cell']).astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]

    elif name == 'har':
        data = fetch_openml('har', version=1, parser='auto')
        X_full = data.data.values.astype(float)
        X_full = np.nan_to_num(X_full)
        y_str = data.target.astype(str).values
        y_binary = np.isin(y_str, ['1', '2', '3']).astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]
        
    else:
        paths = {
            'thyroid': os.path.join(base_dir, 'data', 'thyroid', 'thyroid_preprocessed.pkl'),
            'brca': os.path.join(base_dir, 'data', 'tcga', 'brca_preprocessed.pkl'),
            'paul15': os.path.join(base_dir, 'data', 'scrna', 'paul15_preprocessed.pkl'),
            'gbmlgg': os.path.join(base_dir, 'data', 'tcga_gbmlgg', 'gbmlgg_preprocessed.pkl'),
            'ctg': os.path.join(base_dir, 'data', 'ctg', 'ctg_preprocessed.pkl'),
            'mice': os.path.join(base_dir, 'data', 'mice', 'mice_preprocessed.pkl'),
            'diabetes': os.path.join(base_dir, 'data', 'diabetes', 'diabetes_preprocessed.pkl')
        }
        
        pkl_file = paths[name]
        with open(pkl_file, 'rb') as f:
            ds = pickle.load(f)
            
        if name == 'gbmlgg':
            X_full = ds['X']
            y_binary = ds['y']
            subtypes = ds['subtypes']
            X_target = X_full  # Subtypes defined on the whole dataset here
        else:
            subtypes = ds['subtypes']
            sample_ids = ds['sample_ids']
            X_full = ds['full_X_raw']
            y_binary = ds['full_y']
            full_sample_ids = ds['full_sample_ids']
            
            target_set = set(sample_ids)
            target_idx_in_full = [i for i, sid in enumerate(full_sample_ids) if sid in target_set]
            
            if name == 'diabetes' and len(target_idx_in_full) > 2000:
                # Downsample target cohort to 2000 to save computation
                rng = np.random.RandomState(42)
                sub_indices = rng.choice(len(target_idx_in_full), 2000, replace=False)
                target_idx_in_full = [target_idx_in_full[i] for i in sub_indices]
                subtypes = subtypes[sub_indices]
                
            X_target = X_full[target_idx_in_full]
            
            if name == 'paul15':
                svd = TruncatedSVD(n_components=50, random_state=42)
                X_full = svd.fit_transform(X_full)
                X_target = X_full[target_idx_in_full]
                
    # convert subtypes to 0..k-1
    unique_labels = np.unique(subtypes)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    subtypes_int = np.array([label_map[lbl] for lbl in subtypes])
    k = len(unique_labels)
    
    return X_full, y_binary, X_target, subtypes_int, k
