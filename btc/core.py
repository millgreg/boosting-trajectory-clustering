"""Boosting Trajectory Clustering (BTC).

A fitted binary gradient boosting classifier maps each sample to the sequence of per-tree
margin updates it receives. BTC clusters samples in that space. For depth-1 trees this is
K-Means on an explicit feature map of a weighted leaf-collision proximity (Tan et al., 2020;
Aumon et al., 2026).
"""
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans


def extract_raw_init(booster: Any, X: np.ndarray) -> np.ndarray:
    """Baseline margin F_0 before any tree, one value per sample."""
    if hasattr(booster, '_raw_predict_init'):
        return booster._raw_predict_init(X).ravel()
    name = type(booster).__name__
    if name == 'XGBClassifier':
        import json
        import scipy.special
        try:
            config = json.loads(booster.get_booster().save_config())
            base_score = float(config['learner']['learner_model_param']['base_score'].strip('[]'))
            value = float(scipy.special.logit(base_score)) if 0 < base_score < 1 else 0.0
        except (KeyError, ValueError):
            value = 0.0
        return np.full(X.shape[0], value)
    if name == 'LGBMClassifier':
        try:
            value = float(booster.booster_.dump_model()['tree_info'][0]['tree_structure'].get('internal_value', 0.0))
        except (KeyError, IndexError):
            value = 0.0
        return np.full(X.shape[0], value)
    return np.zeros(X.shape[0])


def extract_staged_decision_function(booster: Any, X: np.ndarray) -> np.ndarray:
    """Cumulative margins F_1, ..., F_T, with shape (n_samples, n_trees)."""
    if hasattr(booster, 'staged_decision_function'):
        return np.column_stack([f.ravel() for f in booster.staged_decision_function(X)])
    name = type(booster).__name__
    n_trees = getattr(booster, 'n_estimators', None) or 100
    if name == 'XGBClassifier':
        return np.column_stack([booster.predict(X, output_margin=True, iteration_range=(0, i)).ravel()
                                for i in range(1, n_trees + 1)])
    if name == 'LGBMClassifier':
        return np.column_stack([booster.predict(X, raw_score=True, num_iteration=i).ravel()
                                for i in range(1, n_trees + 1)])
    raise ValueError(f"Unsupported booster type: {type(booster)}")


class BTC(BaseEstimator, ClusterMixin):
    """Cluster samples by the margin updates a fitted binary booster assigns them.

    Parameters
    ----------
    booster : fitted binary GradientBoostingClassifier, XGBClassifier or LGBMClassifier
    n_clusters : number of clusters
    representation : 'per_tree_delta' (default, the per-tree updates) or 'cumul_logodds'
        (the cumulative margins)
    clustering_algo : 'kmeans' (default) or 'agglomerative' (Ward linkage)
    random_state : seed for K-Means
    """

    def __init__(self, booster: Optional[Any] = None, n_clusters: int = 3, representation: str = 'per_tree_delta',
                 clustering_algo: str = 'kmeans', random_state: Optional[int] = None):
        self.booster = booster
        self.n_clusters = n_clusters
        self.representation = representation
        self.clustering_algo = clustering_algo
        self.random_state = random_state

    def _trajectories(self, X: np.ndarray) -> np.ndarray:
        margins = extract_staged_decision_function(self.booster, X)
        if self.representation == 'cumul_logodds':
            return margins
        if self.representation == 'per_tree_delta':
            return np.diff(np.column_stack([extract_raw_init(self.booster, X), margins]), axis=1)
        raise ValueError(f"Unknown representation: {self.representation}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BTC":
        if self.booster is None:
            raise ValueError("A fitted booster model must be provided.")
        n_classes = len(getattr(self.booster, 'classes_', (0, 1)))
        if n_classes > 2:
            raise ValueError(f"BTC supports binary proxy tasks only, but the booster has {n_classes} classes.")
        self.trajectories_ = self._trajectories(X)
        if self.clustering_algo == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        elif self.clustering_algo == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        else:
            raise ValueError(f"Unknown clustering_algo: {self.clustering_algo}")
        self.labels_ = model.fit_predict(self.trajectories_)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).labels_
