from typing import Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

class RFC(BaseEstimator, ClusterMixin):
    def __init__(self, rf_model: Optional[Any] = None, n_clusters: int = 3, 
                 clustering_algo: str = 'spectral', random_state: Optional[int] = None):
        """
        Random Forest Clustering (RFC)
        Uses leaf co-occurrence proximity from a fitted Random Forest.
        
        Parameters:
        - rf_model: A fitted RandomForestClassifier
        - n_clusters: Number of subclusters to find
        - clustering_algo: Clustering algorithm ('spectral', 'hierarchical')
        - random_state: Random state for clustering algorithms
        """
        self.rf_model = rf_model
        self.n_clusters = n_clusters
        self.clustering_algo = clustering_algo
        self.random_state = random_state
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RFC":
        """
        Compute leaf proximity matrix and cluster the given samples X.
        """
        if self.rf_model is None:
            raise ValueError("A fitted RandomForestClassifier must be provided.")

        # leaf_indices shape: (n_samples, n_estimators)
        leaf_indices = self.rf_model.apply(X) 
        n_samples, n_estimators = leaf_indices.shape
        
        # Compute similarity matrix (proportion of trees where samples share a leaf)
        S = np.zeros((n_samples, n_samples))
        for t in range(n_estimators):
            leaves = leaf_indices[:, t]
            unique_leaves = np.unique(leaves)
            for leaf in unique_leaves:
                mask = (leaves == leaf)
                # Add 1.0 to all pairs in the same leaf
                S[np.ix_(mask, mask)] += 1.0
                
        S /= n_estimators
        self.similarity_matrix_ = S
        
        # Clustering Step
        if self.clustering_algo == 'spectral':
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters, 
                affinity='precomputed', 
                random_state=self.random_state
            )
            self.labels_ = clusterer.fit_predict(S)
        elif self.clustering_algo == 'hierarchical':
            # Convert affinity to distance
            distance = np.clip(1.0 - S, 0.0, 1.0)
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters, 
                metric='precomputed', 
                linkage='average'
            )
            self.labels_ = clusterer.fit_predict(distance)
        else:
            raise ValueError(f"Unknown clustering_algo: {self.clustering_algo}")
            
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster the given samples X using the leaf proximity matrix.
        """
        return self.fit(X, y).labels_
