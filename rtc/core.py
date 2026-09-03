import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from typing import Any, Optional, Union

def extract_staged_decision_function(booster: Any, X: np.ndarray) -> np.ndarray:
    """
    Extracts staged raw log-odds scores (pre-sigmoid/softmax predictions) from various boosters.
    Returns array of shape (n_samples, n_iterations).
    """
    if hasattr(booster, 'staged_decision_function'):
        # sklearn GradientBoostingClassifier
        stages = []
        for f in booster.staged_decision_function(X):
            if f.ndim > 1 and f.shape[1] > 1:
                stages.append(f)
            else:
                stages.append(f.ravel())
        if stages[0].ndim > 1:
            return np.hstack(stages)
        else:
            return np.column_stack(stages)
        
    elif type(booster).__name__ == 'XGBClassifier':
        # xgboost staged raw margin predictions
        staged_preds = []
        n_estimators = getattr(booster, 'n_estimators', 100)
        # Check if the booster supports iteration_range
        for i in range(1, n_estimators + 1):
            margin = booster.predict(X, output_margin=True, iteration_range=(0, i))
            if margin.ndim > 1:
                # If binary, it might return shape (n_samples, 2) or (n_samples,)
                if margin.shape[1] == 2:
                    staged_preds.append(margin[:, 1] - margin[:, 0])
                else:
                    staged_preds.append(margin)
            else:
                staged_preds.append(margin.ravel())
        if staged_preds[0].ndim > 1:
            return np.hstack(staged_preds)
        else:
            return np.column_stack(staged_preds)
        
    elif type(booster).__name__ == 'LGBMClassifier':
        # lightgbm staged raw margin predictions
        staged_preds = []
        n_estimators = getattr(booster, 'n_estimators', 100)
        for i in range(1, n_estimators + 1):
            margin = booster.predict(X, raw_score=True, num_iteration=i)
            if margin.ndim > 1:
                if margin.shape[1] == 2:
                    staged_preds.append(margin[:, 1] - margin[:, 0])
                else:
                    staged_preds.append(margin)
            else:
                staged_preds.append(margin.ravel())
        if staged_preds[0].ndim > 1:
            return np.hstack(staged_preds)
        else:
            return np.column_stack(staged_preds)
        
    else:
        raise ValueError(f"Unsupported booster type: {type(booster)}")

def extract_staged_predictions(booster: Any, X: np.ndarray) -> np.ndarray:
    """
    Extracts staged probabilities from various boosters.
    Returns array of shape (n_samples, n_iterations) for binary or (n_samples, n_iterations * n_classes) for multi-class.
    """
    if hasattr(booster, 'staged_predict_proba'):
        # sklearn GradientBoostingClassifier
        staged_preds = []
        for p in booster.staged_predict_proba(X):
            if p.shape[1] > 2:
                staged_preds.append(p)
            else:
                staged_preds.append(p[:, 1].ravel())
        if staged_preds[0].ndim > 1:
            return np.hstack(staged_preds)
        else:
            return np.column_stack(staged_preds)
        
    elif type(booster).__name__ == 'XGBClassifier':
        # xgboost
        staged_preds = []
        n_estimators = getattr(booster, 'n_estimators', 100)
        for i in range(1, n_estimators + 1):
            p = booster.predict_proba(X, iteration_range=(0, i))
            if p.shape[1] > 2:
                staged_preds.append(p)
            else:
                staged_preds.append(p[:, 1].ravel())
        if staged_preds[0].ndim > 1:
            return np.hstack(staged_preds)
        else:
            return np.column_stack(staged_preds)
            
    elif type(booster).__name__ == 'LGBMClassifier':
        # lightgbm
        staged_preds = []
        n_estimators = getattr(booster, 'n_estimators', 100)
        for i in range(1, n_estimators + 1):
            p = booster.predict(X, raw_score=False, num_iteration=i)
            if p.ndim > 1 and p.shape[1] > 2:
                staged_preds.append(p)
            elif p.ndim > 1 and p.shape[1] == 2:
                staged_preds.append(p[:, 1].ravel())
            else:
                staged_preds.append(p.ravel())
        if staged_preds[0].ndim > 1:
            return np.hstack(staged_preds)
        else:
            return np.column_stack(staged_preds)
    else:
        raise ValueError(f"Unsupported booster type: {type(booster)}")

class RTC(BaseEstimator, ClusterMixin):
    def __init__(self, booster: Optional[Any] = None, n_clusters: int = 3, representation: str = 'per_tree_delta',
                 clustering_algo: str = 'kmeans', similarity: str = 'rbf', normalization: str = 'none', 
                 trajectory_pca: Optional[int] = None, trajectory_umap: Optional[int] = None, windowed: bool = False, window_size: int = 10, window_stride: int = 5,
                 consensus_runs: Optional[int] = None, random_state: Optional[int] = None, target_class: int = 1):
        """
        Residual Trajectory Clustering (RTC)
        
        Parameters:
        - booster: A fitted gradient boosting model (Sklearn, XGBoost, LightGBM)
        - n_clusters: Number of subclusters to find
        - representation: Trajectory type ('per_tree_delta', 'cumul_logodds', 'cumul_proba', 'proba_deriv', 'ensemble')
        - clustering_algo: Clustering algorithm ('kmeans', 'agglomerative', 'spectral', 'gmm')
        - similarity: Affinity metric if clustering_algo is 'spectral' ('rbf', 'cosine', 'corr')
        - normalization: Trajectory scaling ('none', 'per-round', 'per-sample')
        - trajectory_pca: Optional number of PCA components to reduce trajectory dimensionality (default: None)
        - trajectory_umap: Optional number of UMAP components to reduce trajectory dimensionality (default: None)
        - windowed: Whether to use windowed multi-scale features (default: False)
        - window_size: Size of sliding window (default: 10)
        - window_stride: Stride of sliding window (default: 5)
        - consensus_runs: Optional number of KMeans runs for consensus clustering (default: None)
        - target_class: The class label of the samples being clustered (default: 1)
        """
        self.booster = booster
        self.n_clusters = n_clusters
        self.representation = representation
        self.clustering_algo = clustering_algo
        self.similarity = similarity
        self.normalization = normalization
        self.trajectory_pca = trajectory_pca
        self.trajectory_umap = trajectory_umap
        self.windowed = windowed
        self.window_size = window_size
        self.window_stride = window_stride
        self.consensus_runs = consensus_runs
        self.random_state = random_state
        self.target_class = target_class
        
    @staticmethod
    def _windowed_features(R: np.ndarray, window_size: int = 10, stride: int = 5) -> np.ndarray:
        """Extract multi-scale features from trajectory windows."""
        features = []
        n_rounds = R.shape[1]
        for start in range(0, n_rounds - window_size + 1, stride):
            window = R[:, start:start + window_size]
            features.append(np.mean(window, axis=1))    # level
            features.append(np.std(window, axis=1))      # volatility
            # slope via simple linear regression
            t = np.arange(window_size, dtype=float)
            t_centered = t - t.mean()
            denominator = (t_centered ** 2).mean()
            if denominator == 0:
                slopes = np.zeros(R.shape[0])
            else:
                slopes = (window * t_centered).mean(axis=1) / denominator
            features.append(slopes)                      # trend
        return np.column_stack(features)
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster the given samples X. 
        Note: X should ONLY contain samples from self.target_class.
        """
        # 1. Extract Trajectory Representation
        if self.representation == 'cumul_proba':
            R = extract_staged_predictions(self.booster, X)
        elif self.representation == 'cumul_logodds':
            R = extract_staged_decision_function(self.booster, X)
        elif self.representation == 'per_tree_delta':
            lo = extract_staged_decision_function(self.booster, X)
            R = np.diff(lo, axis=1, prepend=lo[:, 0:1])
        elif self.representation == 'proba_deriv':
            P = extract_staged_predictions(self.booster, X)
            R = np.diff(P, axis=1)
        elif self.representation == 'ensemble':
            lo = extract_staged_decision_function(self.booster, X)
            delta = np.diff(lo, axis=1, prepend=lo[:, 0:1])
            R = np.hstack([lo, delta])
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
            
        # 2. Normalization / Scaling
        if self.normalization == 'per-round':
            R = (R - np.mean(R, axis=0)) / (np.std(R, axis=0) + 1e-8)
        elif self.normalization == 'per-sample':
            R = (R - np.mean(R, axis=1, keepdims=True)) / (np.std(R, axis=1, keepdims=True) + 1e-8)
            
        # 2.3 Optional windowed features
        if self.windowed:
            R = self._windowed_features(R, self.window_size, self.window_stride)
            
        # 2.5 Optional PCA on trajectory space
        if self.trajectory_pca is not None and self.trajectory_pca < R.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.trajectory_pca, random_state=self.random_state)
            R = pca.fit_transform(R)
            
        # 2.6 Optional UMAP on trajectory space
        if self.trajectory_umap is not None and self.trajectory_umap < R.shape[1]:
            import umap
            reducer = umap.UMAP(n_components=self.trajectory_umap, random_state=self.random_state)
            R = reducer.fit_transform(R)
            
        # Save trajectory representations used for clustering
        self.trajectories_ = R
        
        # 3. Clustering Step
        if self.clustering_algo == 'kmeans':
            if self.consensus_runs is not None and self.consensus_runs > 1:
                # Consensus clustering
                n = R.shape[0]
                C = np.zeros((n, n))
                for m in range(self.consensus_runs):
                    km = KMeans(n_clusters=self.n_clusters, n_init=1,
                                random_state=(self.random_state or 0) + m)
                    labels_m = km.fit_predict(R)
                    for k in range(self.n_clusters):
                        mask = (labels_m == k)
                        C[np.ix_(mask, mask)] += 1
                C /= self.consensus_runs
                self.consensus_matrix_ = C
                clusterer = AgglomerativeClustering(
                    n_clusters=self.n_clusters, metric='precomputed', linkage='average')
                self.labels_ = clusterer.fit_predict(1.0 - C)
            else:
                clusterer = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
                self.labels_ = clusterer.fit_predict(R)
            
        elif self.clustering_algo == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
            self.labels_ = clusterer.fit_predict(R)
            
        elif self.clustering_algo == 'gmm':
            from sklearn.mixture import GaussianMixture
            clusterer = GaussianMixture(n_components=self.n_clusters, covariance_type='full', random_state=self.random_state)
            self.labels_ = clusterer.fit_predict(R)
            
        elif self.clustering_algo == 'spectral':
            # Similarity Matrix Computation
            if self.similarity == 'corr':
                R_centered = R - np.mean(R, axis=1, keepdims=True)
                norms = np.linalg.norm(R_centered, axis=1, keepdims=True)
                norms[norms == 0] = 1e-8
                R_norm = R_centered / norms
                S = np.dot(R_norm, R_norm.T)
                S = np.clip(S, 0.0, 1.0)
            elif self.similarity == 'cosine':
                S = cosine_similarity(R)
                S = np.clip(S, 0.0, 1.0)
            elif self.similarity == 'rbf':
                S = rbf_kernel(R, gamma=1.0 / R.shape[1])
            else:
                raise ValueError(f"Unknown similarity: {self.similarity}")
                
            S = (S + S.T) / 2
            np.fill_diagonal(S, 1.0)
            self.similarity_matrix_ = S
            
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters, 
                affinity='precomputed', 
                random_state=self.random_state
            )
            self.labels_ = clusterer.fit_predict(S)
        else:
            raise ValueError(f"Unknown clustering_algo: {self.clustering_algo}")
            
        return self.labels_


class MultiTaskRTC(BaseEstimator, ClusterMixin):
    def __init__(self, boosters: Optional[list] = None, n_clusters: int = 3, representation: str = 'per_tree_delta',
                 normalization: str = 'none', trajectory_pca: Optional[int] = None, consensus_runs: Optional[int] = None, random_state: Optional[int] = None):
        """
        Multi-Task RTC: stacks trajectories from multiple GBDT models.
        
        Parameters:
        - boosters: list of fitted GradientBoostingClassifier models
        - Other params same as RTC
        """
        self.boosters = boosters or []
        self.n_clusters = n_clusters
        self.representation = representation
        self.normalization = normalization
        self.trajectory_pca = trajectory_pca
        self.consensus_runs = consensus_runs
        self.random_state = random_state
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        trajectories = []
        for booster in self.boosters:
            if self.representation == 'cumul_logodds':
                R = extract_staged_decision_function(booster, X)
            elif self.representation == 'per_tree_delta':
                lo = extract_staged_decision_function(booster, X)
                R = np.diff(lo, axis=1, prepend=lo[:, 0:1])
            elif self.representation == 'cumul_proba':
                R = extract_staged_predictions(booster, X)
            elif self.representation == 'proba_deriv':
                P = extract_staged_predictions(booster, X)
                R = np.diff(P, axis=1)
            else:
                raise ValueError(f"Unknown representation: {self.representation}")
            
            if self.normalization == 'per-round':
                R = (R - np.mean(R, axis=0)) / (np.std(R, axis=0) + 1e-8)
            elif self.normalization == 'per-sample':
                R = (R - np.mean(R, axis=1, keepdims=True)) / (np.std(R, axis=1, keepdims=True) + 1e-8)
            
            trajectories.append(R)
        
        # Stack all trajectories horizontally
        R_multi = np.hstack(trajectories)
        
        # Optional PCA
        if self.trajectory_pca is not None and self.trajectory_pca < R_multi.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.trajectory_pca, random_state=self.random_state)
            R_multi = pca.fit_transform(R_multi)
        
        self.trajectories_ = R_multi
        
        if self.consensus_runs is not None and self.consensus_runs > 1:
            n = R_multi.shape[0]
            C = np.zeros((n, n))
            for m in range(self.consensus_runs):
                km = KMeans(n_clusters=self.n_clusters, n_init=1, random_state=(self.random_state or 0) + m)
                labels_m = km.fit_predict(R_multi)
                for k in range(self.n_clusters):
                    mask = (labels_m == k)
                    C[np.ix_(mask, mask)] += 1
            C /= self.consensus_runs
            self.consensus_matrix_ = C
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='average')
            self.labels_ = clusterer.fit_predict(1.0 - C)
        else:
            clusterer = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
            self.labels_ = clusterer.fit_predict(R_multi)
            
        return self.labels_


def auto_select_depth(X_train: np.ndarray, y_train: np.ndarray, X_target: np.ndarray, n_clusters: int, 
                      depths: list = [1, 2, 3, 5], n_estimators: int = 100,
                      representation: str = 'cumul_logodds', random_state: int = 42) -> tuple:
    """
    Automatically select GBDT depth based on trajectory quality metrics.
    
    Returns the best (depth, rtc_instance, labels, score, var_ratio_top5, sil) tuple.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_result = None
    
    for depth in depths:
        gbm = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=depth, random_state=random_state)
        gbm.fit(X_train, y_train)
        
        rtc = RTC(booster=gbm, n_clusters=n_clusters, representation=representation,
                  normalization='none', random_state=random_state)
        labels = rtc.fit_predict(X_target)
        
        R = rtc.trajectories_
        
        # Compute trajectory quality score:
        # 1. PCA explained variance ratio (top-k should be high)
        pca = PCA(n_components=min(20, R.shape[1]))
        pca.fit(R)
        var_ratio_top5 = sum(pca.explained_variance_ratio_[:5])
        
        # 2. Silhouette score on the clustering
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(R, labels, sample_size=min(2000, len(labels)))
        else:
            sil = -1
        
        # Combined score (weighted)
        score = 0.5 * var_ratio_top5 + 0.5 * (sil + 1) / 2  # normalize sil to [0,1]
        
        if score > best_score:
            best_score = score
            best_result = (depth, rtc, labels, score, var_ratio_top5, sil)
            
    return best_result
