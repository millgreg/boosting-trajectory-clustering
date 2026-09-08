import os
import time
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sys

# Add parent and local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baselines.rfc import RFC
from btc.core import BTC

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, 'experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'scalability_results.csv')
    
    print("Running genuine empirical scalability benchmark...")
    print("Evaluating clustering time across increasing sample size N.")
    
    n_samples_list = [500, 1000, 2500, 5000, 10000]
    n_features = 20
    n_informative = 10
    n_clusters = 3
    n_trees = 100
    
    # Warmup to eliminate first-call library overhead
    X_w, y_w = make_classification(n_samples=100, n_features=n_features, random_state=0)
    gb_w = GradientBoostingClassifier(n_estimators=10, max_depth=1).fit(X_w, y_w)
    BTC(booster=gb_w, n_clusters=2).fit_predict(X_w)
    
    records = []
    
    for n in n_samples_list:
        print(f"\nEvaluating N = {n}...")
        X, y = make_classification(
            n_samples=n, 
            n_features=n_features, 
            n_informative=n_informative, 
            random_state=42
        )
        
        # Fit models on proxy task
        gb = GradientBoostingClassifier(
            n_estimators=n_trees, 
            max_depth=1, 
            max_features='sqrt', 
            random_state=42
        )
        gb.fit(X, y)
        
        rf = RandomForestClassifier(
            n_estimators=n_trees, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # 1. Measure BTC clustering time (trajectory extraction + K-Means)
        t0 = time.perf_counter()
        btc = BTC(
            booster=gb, 
            n_clusters=n_clusters, 
            representation='per_tree_delta', 
            normalization='none', 
            clustering_algo='kmeans', 
            random_state=42
        )
        btc.fit_predict(X)
        btc_time = time.perf_counter() - t0
        print(f"  BTC: {btc_time:.4f}s")
        
        # 2. Measure RFC Agglomerative clustering time (proximity matrix + average linkage)
        try:
            t0 = time.perf_counter()
            rfc_agg = RFC(
                rf_model=rf, 
                n_clusters=n_clusters, 
                clustering_algo='hierarchical', 
                random_state=42
            )
            rfc_agg.fit_predict(X)
            agg_time = time.perf_counter() - t0
            print(f"  RFC Agglomerative: {agg_time:.4f}s")
        except Exception as e:
            print(f"  RFC Agglomerative failed: {e}")
            agg_time = np.nan
            
        # 3. Measure RFC Spectral clustering time (proximity matrix + spectral graph partition)
        try:
            t0 = time.perf_counter()
            rfc_spec = RFC(
                rf_model=rf, 
                n_clusters=n_clusters, 
                clustering_algo='spectral', 
                random_state=42
            )
            rfc_spec.fit_predict(X)
            spectral_time = time.perf_counter() - t0
            print(f"  RFC Spectral: {spectral_time:.4f}s")
        except Exception as e:
            print(f"  RFC Spectral failed: {e}")
            spectral_time = np.nan
            
        records.append({
            'N': n,
            'BTC_Time_s': btc_time,
            'Agglomerative_Time_s': agg_time,
            'Spectral_Time_s': spectral_time
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(out_path, index=False)
    print(f"\nEmpirical scalability benchmark complete! Saved to {out_path}")
    print(res_df)

if __name__ == '__main__':
    main()
