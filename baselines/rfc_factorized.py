"""Spectral Random Forest Clustering without forming the N x N proximity matrix.

Breiman's proximity is P = Z Z^T / T, where Z is the sparse one-hot leaf incidence
matrix of N samples over all leaves of T trees (Aumon et al., 2026). Every sample has
proximity 1 with itself. Like scikit-learn's spectral clustering, which ignores the
diagonal when it builds the normalized Laplacian, this uses the affinity W = P - I with
degrees d = W 1. The leading eigenvectors of D^{-1/2} W D^{-1/2} are found with a
matrix-free operator, v -> A (A^T v) - v / d with A = D^{-1/2} Z / sqrt(T), so time and
memory grow with the number of nonzeros in Z rather than with N squared. As in
scikit-learn, the embedding is scaled by D^{-1/2} before K-Means.
"""
from typing import Any, Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


class FactorizedRFC(BaseEstimator, ClusterMixin):
    def __init__(self, rf_model: Optional[Any] = None, n_clusters: int = 3, random_state: Optional[int] = None):
        self.rf_model = rf_model
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FactorizedRFC":
        if self.rf_model is None:
            raise ValueError("A fitted RandomForestClassifier must be provided.")
        leaves = self.rf_model.apply(X)
        n, n_trees = leaves.shape
        Z = OneHotEncoder(handle_unknown='ignore').fit_transform(leaves).tocsr().astype(float)
        degree = np.asarray(Z @ (Z.T @ np.ones(n))).ravel() / n_trees - 1.0
        degree = np.maximum(degree, 1e-12)
        inv_sqrt = 1.0 / np.sqrt(degree)
        A = diags(inv_sqrt) @ Z / np.sqrt(n_trees)
        operator = LinearOperator((n, n), matvec=lambda v: A @ (A.T @ v) - v / degree, dtype=float)
        start = np.random.RandomState(self.random_state).uniform(-1, 1, n)
        values, vectors = eigsh(operator, k=self.n_clusters, which='LA', v0=start)
        embedding = vectors[:, np.argsort(values)[::-1]] * inv_sqrt[:, None]
        self.embedding_ = embedding
        self.labels_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit_predict(embedding)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).labels_
