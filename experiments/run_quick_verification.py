import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import ALL_DATASETS, load_dataset
from baselines.rfc import RFC
from btc.core import BTC

def main():
    print("="*70)
    print("Running Fast 1-Seed Smoke Test Across All 13 Benchmark Datasets")
    print("="*70)
    
    results = []
    
    for name in ALL_DATASETS:
        t0 = time.time()
        print(f"\nEvaluating dataset: {name.upper()}...")
        try:
            X_full, y_binary, X_target, subtypes, k = load_dataset(name)
            print(f"  Loaded: N_full={len(X_full)}, N_target={len(X_target)}, P={X_full.shape[1]}, K={k}")
            
            # 1. K-Means Baseline
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            km_labels = km.fit_predict(X_target)
            ari_km = adjusted_rand_score(subtypes, km_labels)
            
            # 2. RFC Baseline
            rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
            rf.fit(X_full, y_binary)
            rfc = RFC(rf_model=rf, n_clusters=k, clustering_algo='spectral', random_state=0)
            rfc_labels = rfc.fit_predict(X_target)
            ari_rfc = adjusted_rand_score(subtypes, rfc_labels)
            
            # 3. BTC (Default: Stumps, sqrt features, per_tree_delta)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt', random_state=0)
            gb.fit(X_full, y_binary)
            btc = BTC(
                booster=gb, 
                n_clusters=k, 
                representation='per_tree_delta', 
                normalization='none', 
                clustering_algo='kmeans', 
                random_state=0
            )
            btc_labels = btc.fit_predict(X_target)
            ari_btc = adjusted_rand_score(subtypes, btc_labels)
            
            elapsed = time.time() - t0
            print(f"  Results (seed 0): BTC={ari_btc:.3f} | RFC={ari_rfc:.3f} | K-Means={ari_km:.3f} (Time: {elapsed:.1f}s)")
            
            results.append({
                'Dataset': name.capitalize(),
                'BTC_Seed0': ari_btc,
                'RFC_Seed0': ari_rfc,
                'KMeans_Seed0': ari_km,
                'Status': 'PASS'
            })
        except Exception as e:
            print(f"  FAILED on {name}: {e}")
            results.append({
                'Dataset': name.capitalize(),
                'BTC_Seed0': np.nan,
                'RFC_Seed0': np.nan,
                'KMeans_Seed0': np.nan,
                'Status': f'FAIL: {e}'
            })
            
    res_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("SMOKE TEST SUMMARY TABLE (Seed 0)")
    print("="*70)
    print(res_df.to_string(index=False))
    
    # Check if all passed
    if (res_df['Status'] == 'PASS').all():
        print("\nSUCCESS: All 13 datasets loaded and clustered cleanly without errors!")
    else:
        print("\nWARNING: Some datasets failed during evaluation.")

if __name__ == '__main__':
    main()
