import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import BTC
from experiments.dataset_loader import load_dataset

N_SEEDS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATASETS = ['thyroid', 'brca', 'paul15', 'gbmlgg', 'ctg', 'mice', 'har']

def run_seed(seed, name, X_full, y_binary, X_target, subtypes, k):
    res = []
    configs = [
        ('KMeans', 'kmeans', None),
        ('GMM', 'gmm', None),
        ('Spectral', 'spectral', 'rbf'),
        ('Agglomerative', 'agglomerative', None)
    ]
    
    try:
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=seed)
        gb.fit(X_full, y_binary)
    except Exception as e:
        print(f"Error fitting proxy for {name} seed {seed}: {e}")
        return res
        
    for label, backend, similarity in configs:
        t_start = time.time()
        try:
            btc = BTC(booster=gb, n_clusters=k, representation='per_tree_delta', normalization='none', 
                      clustering_algo=backend, similarity=similarity, random_state=seed)
            labels = btc.fit_predict(X_target)
            ari = adjusted_rand_score(subtypes, labels)
        except Exception as e:
            ari = 0.0
            
        t_end = time.time()
        res.append({'Dataset': name, 'seed': seed, 'Method': label, 'ARI': ari, 'Fit_Time_s': t_end - t_start})
        
    return res

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    
    for name in DATASETS:
        print(f"Running backend ablation for {name}...")
        try:
            X_full, y_binary, X_target, subtypes, k = load_dataset(name)
        except Exception as e:
            continue
            
        results = Parallel(n_jobs=-1)(delayed(run_seed)(seed, name, X_full, y_binary, X_target, subtypes, k) for seed in range(N_SEEDS))
        for r in results:
            all_results.extend(r)
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, 'ablation_backend_results.csv'), index=False)
    summary = df.groupby(['Dataset', 'Method']).agg({'ARI': ['mean', 'std']}).reset_index()
    summary.columns = ['Dataset', 'Method', 'ARI_mean', 'ARI_std']
    # NOTE: saving to ablation_advanced_summary.csv to match what was there historically (or we can update the table generation script)
    summary.to_csv(os.path.join(OUT_DIR, 'ablation_advanced_summary.csv'), index=False)
    print("Backend ablation complete.")
