import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from btc.core import BTC
from baselines.rfc import RFC
from experiments.dataset_loader import load_dataset, ALL_DATASETS

N_SEEDS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

def run_seed(seed, name, X_full, y_binary, X_target, y_true_subtypes, k):
    seed_results = []
    
    # Baseline: RFC
    t0 = time.time()
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_full, y_binary)
        
        # Supervised AUC
        if len(np.unique(y_binary)) > 1:
            preds_rf = rf.predict_proba(X_full)[:, 1]
            auc = roc_auc_score(y_binary, preds_rf)
        else:
            auc = np.nan

        rfc = RFC(rf_model=rf, n_clusters=k, clustering_algo='spectral', random_state=seed)
        labels = rfc.fit_predict(X_target)
        ari = adjusted_rand_score(y_true_subtypes, labels)
        nmi = normalized_mutual_info_score(y_true_subtypes, labels)
        t_sec = time.time() - t0
    except Exception as e:
        print(f"Error RFC {name} seed {seed}: {e}")
        ari, nmi, auc, t_sec = 0.0, 0.0, np.nan, np.nan
    seed_results.append({
        'Dataset': name, 
        'seed': seed, 
        'Method': 'Baseline_RFC', 
        'ARI': ari, 
        'NMI': nmi, 
        'AUC': auc, 
        'Fit_Time_s': t_sec,
        'N_Samples': X_full.shape[0],
        'N_Target': X_target.shape[0],
        'N_Features': X_full.shape[1],
        'K_Clusters': k
    })

    # Core Configurations
    configs = [
        # (name, representation, normalization, consensus_runs, max_features)
        ('BTC_Base_Delta_persample', 'per_tree_delta', 'per-sample', None, None),
        ('BTC_Base_Delta_none', 'per_tree_delta', 'none', None, None),
        ('BTC_Base_Logodds', 'cumul_logodds', 'none', None, None),
        ('BTC_Consensus_Delta_persample', 'per_tree_delta', 'per-sample', 100, None),
        ('BTC_Consensus_Delta_none', 'per_tree_delta', 'none', 100, None),
        ('BTC_Consensus_Logodds', 'cumul_logodds', 'none', 100, None),
        ('BTC_Stochastic_Delta_sqrt_persample', 'per_tree_delta', 'per-sample', None, 'sqrt'),
        ('BTC_Stochastic_Delta_sqrt_none', 'per_tree_delta', 'none', None, 'sqrt'),
        ('BTC_Stochastic_Logodds_sqrt', 'cumul_logodds', 'none', None, 'sqrt'),
        ('BTC_Stochastic_Delta_0.1_persample', 'per_tree_delta', 'per-sample', None, 0.1),
        ('BTC_Stochastic_Delta_0.1_none', 'per_tree_delta', 'none', None, 0.1),
        ('BTC_Stochastic_Logodds_0.1', 'cumul_logodds', 'none', None, 0.1),
    ]

    n_features = X_full.shape[1]

    for config_name, rep, norm, consensus, mf in configs:
        # adjust max_features if 0.1 is less than 1 feature
        if mf == 0.1 and (0.1 * n_features) < 1:
            mf = 'sqrt'
            
        t0 = time.time()
        try:
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features=mf, random_state=seed)
            gb.fit(X_full, y_binary)
            
            # Supervised AUC
            if len(np.unique(y_binary)) > 1:
                preds = gb.predict_proba(X_full)[:, 1]
                auc = roc_auc_score(y_binary, preds)
            else:
                auc = np.nan
            
            btc = BTC(
                booster=gb, 
                n_clusters=k, 
                representation=rep, 
                normalization=norm, 
                clustering_algo='kmeans', 
                consensus_runs=consensus, 
                random_state=seed
            )
            labels = btc.fit_predict(X_target)
            ari = adjusted_rand_score(y_true_subtypes, labels)
            nmi = normalized_mutual_info_score(y_true_subtypes, labels)
            t_sec = time.time() - t0
        except Exception as e:
            print(f"Error {config_name} {name} seed {seed}: {e}")
            ari, nmi, auc, t_sec = 0.0, 0.0, np.nan, np.nan
            
        seed_results.append({
            'Dataset': name, 
            'seed': seed, 
            'Method': config_name, 
            'ARI': ari, 
            'NMI': nmi, 
            'AUC': auc, 
            'Fit_Time_s': t_sec,
            'N_Samples': X_full.shape[0],
            'N_Target': X_target.shape[0],
            'N_Features': X_full.shape[1],
            'K_Clusters': k
        })

    return seed_results

def main():
    all_results = []
    
    for name in ALL_DATASETS:
        print(f"Running benchmark for dataset: {name}")
        try:
            X_full, y_binary, X_target, subtypes, k = load_dataset(name)
        except Exception as e:
            print(f"Skipping {name} due to load error: {e}")
            continue
            
        results = Parallel(n_jobs=-1)(
            delayed(run_seed)(seed, name, X_full, y_binary, X_target, subtypes, k) 
            for seed in range(N_SEEDS)
        )
        
        # Flatten results
        for seed_res in results:
            all_results.extend(seed_res)
            
    df = pd.DataFrame(all_results)
    out_path = os.path.join(OUT_DIR, 'canonical_results_30seeds.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved master results to {out_path}")

if __name__ == '__main__':
    main()
