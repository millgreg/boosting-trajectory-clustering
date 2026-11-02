import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD

LOCAL_DATASETS = ['brca', 'paul15', 'gbmlgg', 'ctg', 'mice', 'diabetes']
OPENML_DATASETS = ['dermatology', 'vehicle', 'segment', 'arrhythmia', 'leukemia', 'thyroid', 'har']
ALL_DATASETS = LOCAL_DATASETS + OPENML_DATASETS

MAX_MISSING_FRACTION = 0.5


def clean_features(df):
    """Encode OpenML features without inventing values.

    Columns missing in more than half of the rows are dropped, remaining numeric
    gaps take the column median, and categorical gaps become all-zero dummies.
    No labels are used.
    """
    df = df.loc[:, df.isna().mean() <= MAX_MISSING_FRACTION].copy()
    numeric = df.select_dtypes('number').columns
    df[numeric] = df[numeric].fillna(df[numeric].median())
    return pd.get_dummies(df, drop_first=True).values.astype(float)


def load_dataset(name, base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    if name in ['dermatology', 'vehicle', 'segment']:
        data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
        X_full = clean_features(data.data)
        y_raw = pd.Series(data.target)
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            y_raw, _ = pd.factorize(y_raw, sort=True)
        else:
            y_raw = pd.to_numeric(y_raw, errors='coerce').values
        y_raw = np.nan_to_num(y_raw)
        
        if len(np.unique(y_raw)) > 2:
            # The first class in sorted order becomes y=0, independent of row order; the rest form the target cohort.
            y_binary = (y_raw > np.min(y_raw)).astype(int)
        else:
            y_binary = (y_raw == np.max(y_raw)).astype(int)
            
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_raw[target_mask]
        
    elif name == 'thyroid':
        data = fetch_openml(data_id=57, as_frame=True, parser='auto')
        X_full = clean_features(data.data)
        y_str = data.target.astype(str).values
        y_binary = (y_str != 'negative').astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]

    elif name == 'arrhythmia':
        data = fetch_openml('arrhythmia', version=1, parser='auto', as_frame=True)
        X_full = clean_features(data.data)
        y_str = data.target.astype(str).values
        y_binary = (y_str != '1').astype(int)
        target_mask = (y_binary == 1)
        X_target = X_full[target_mask]
        subtypes = y_str[target_mask]
        
    elif name == 'leukemia':
        data = fetch_openml(data_id=45092, parser='auto', as_frame=True)
        X_full = clean_features(data.data)
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
