import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.dataset_loader import load_dataset, ALL_DATASETS

N_SEEDS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')

def run_seed(seed, name, X_target, subtypes, k):
    seed_results = []
    
    # 1. K-Means
    t0 = time.time()
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X_target)
        t1 = time.time()
        ari = adjusted_rand_score(subtypes, labels)
        nmi = normalized_mutual_info_score(subtypes, labels)
    except Exception as e:
        print(f"Error KMeans {name} seed {seed}: {e}")
        ari, nmi, t1 = 0.0, 0.0, time.time()
    seed_results.append({'Dataset': name, 'seed': seed, 'Method': 'Baseline_KMeans', 'ARI': ari, 'NMI': nmi, 'AUC': np.nan, 'Fit_Time_s': t1 - t0})

    # 2. PCA + K-Means
    t0 = time.time()
    try:
        n_features = X_target.shape[1]
        n_comp = min(n_features, X_target.shape[0])
        pca = PCA(n_components=n_comp, random_state=seed)
        pca.fit(X_target)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.90) + 1
        X_pca = pca.transform(X_target)[:, :d]
        km_pca = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_pca = km_pca.fit_predict(X_pca)
        t1 = time.time()
        ari_pca = adjusted_rand_score(subtypes, labels_pca)
        nmi_pca = normalized_mutual_info_score(subtypes, labels_pca)
    except Exception as e:
        print(f"Error PCA+KMeans {name} seed {seed}: {e}")
        ari_pca, nmi_pca, t1 = 0.0, 0.0, time.time()
    seed_results.append({'Dataset': name, 'seed': seed, 'Method': 'Baseline_PCA_KMeans', 'ARI': ari_pca, 'NMI': nmi_pca, 'AUC': np.nan, 'Fit_Time_s': t1 - t0})

    # 3. UMAP + K-Means
    t0 = time.time()
    try:
        import umap
        n_comp_umap = min(10, X_target.shape[1] - 1, X_target.shape[0] - 2)
        if n_comp_umap < 1: n_comp_umap = 1
        reducer = umap.UMAP(n_components=n_comp_umap, random_state=seed, n_jobs=1)
        X_umap = reducer.fit_transform(X_target)
        km_umap = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_umap = km_umap.fit_predict(X_umap)
        t1 = time.time()
        ari_umap = adjusted_rand_score(subtypes, labels_umap)
        nmi_umap = normalized_mutual_info_score(subtypes, labels_umap)
    except Exception as e:
        print(f"Error UMAP+KMeans {name} seed {seed}: {e}")
        ari_umap, nmi_umap, t1 = 0.0, 0.0, time.time()
    seed_results.append({'Dataset': name, 'seed': seed, 'Method': 'Baseline_UMAP_KMeans', 'ARI': ari_umap, 'NMI': nmi_umap, 'AUC': np.nan, 'Fit_Time_s': t1 - t0})

    return seed_results

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    for name in ALL_DATASETS:
        print(f'Running unsupervised baselines for dataset: {name}')
        try:
            _, _, X_target, subtypes, k = load_dataset(name)
        except Exception as e:
            print(f'Skipping {name} due to load error: {e}')
            continue
            
        results = Parallel(n_jobs=-1)(delayed(run_seed)(seed, name, X_target, subtypes, k) for seed in range(N_SEEDS))
        for seed_res in results:
            all_results.extend(seed_res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, 'kmeans_baselines.csv'), index=False)
    print("Done computing unsupervised baselines.")
