"""Spectral clustering on Random Forest proximity variants without forming an N x N matrix.

Each proximity is written as P = Q W^T with sparse sample-by-leaf factors (Aumon et al., 2026):

* Breiman: Q = W = Z / sqrt(T), where Z is the one-hot leaf incidence over all trees.
* Out-of-bag (separable approximation of Aumon et al.): Q = W, with entry o_t(i) / sqrt(S_i)
  at sample i's leaf in tree t, where o_t(i) = 1 if i is out of bag for tree t and S_i is the
  number of such trees.
* RF-GAP (Rhodes et al., 2023): Q has o_t(i) / S_i at sample i's leaf, and W has
  c_t(j) / M_t(l) at sample j's leaf l, where c_t(j) is j's in-bag multiplicity and M_t(l)
  the total in-bag multiplicity in that leaf. P is asymmetric and is symmetrized as (P + P^T) / 2.

Spectral clustering uses the affinity without self-similarities, as scikit-learn does, the
normalized affinity D^{-1/2} A D^{-1/2}, its leading eigenvectors, rows scaled by D^{-1/2},
and K-Means.
"""
from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans

VARIANTS = ('breiman', 'oob', 'rfgap')


def _factors(forest: Any, X_train: np.ndarray, target_rows: np.ndarray, variant: str):
    leaves_train = forest.apply(X_train)
    n_train, n_trees = leaves_train.shape
    offsets = np.concatenate([[0], np.cumsum(leaves_train.max(axis=0) + 1)[:-1]])
    columns_train = leaves_train + offsets
    n_columns = int(columns_train.max()) + 1
    target_columns = columns_train[target_rows]
    n = len(target_rows)
    rows = np.repeat(np.arange(n), n_trees)

    if variant == 'breiman':
        values = np.full(n * n_trees, 1.0 / np.sqrt(n_trees))
        Q = csr_matrix((values, (rows, target_columns.ravel())), shape=(n, n_columns))
        return Q, Q, Q.multiply(Q).sum(axis=1).A1

    counts = np.zeros((n_train, n_trees))
    for t, drawn in enumerate(forest.estimators_samples_):
        counts[:, t] = np.bincount(drawn, minlength=n_train)
    out_of_bag = (counts[target_rows] == 0).astype(float)
    n_oob = np.maximum(out_of_bag.sum(axis=1), 1.0)

    if variant == 'oob':
        values = (out_of_bag / np.sqrt(n_oob)[:, None]).ravel()
        Q = csr_matrix((values, (rows, target_columns.ravel())), shape=(n, n_columns))
        return Q, Q, Q.multiply(Q).sum(axis=1).A1

    if variant == 'rfgap':
        leaf_mass = np.bincount(columns_train.ravel(), weights=counts.ravel(), minlength=n_columns)
        q_values = (out_of_bag / n_oob[:, None]).ravel()
        w_values = (counts[target_rows] / np.maximum(leaf_mass[target_columns], 1e-12)).ravel()
        Q = csr_matrix((q_values, (rows, target_columns.ravel())), shape=(n, n_columns))
        W = csr_matrix((w_values, (rows, target_columns.ravel())), shape=(n, n_columns))
        return Q, W, np.asarray(Q.multiply(W).sum(axis=1)).ravel()

    raise ValueError(f'Unknown variant: {variant}')


def dense_proximity(forest: Any, X_train: np.ndarray, target_rows: np.ndarray, variant: str) -> np.ndarray:
    """Symmetrized proximity among target rows, for testing on small data."""
    Q, W, _ = _factors(forest, X_train, target_rows, variant)
    P = (Q @ W.T).toarray()
    return (P + P.T) / 2


def spectral_embedding(forest: Any, X_train: np.ndarray, target_rows: np.ndarray, variant: str,
                       n_components: int, random_state: Optional[int] = None) -> np.ndarray:
    Q, W, self_similarity = _factors(forest, X_train, target_rows, variant)
    n = Q.shape[0]

    def affinity(v):
        return 0.5 * (Q @ (W.T @ v) + W @ (Q.T @ v)) - self_similarity * v

    degree = np.maximum(affinity(np.ones(n)), 1e-12)
    inv_sqrt = 1.0 / np.sqrt(degree)
    operator = LinearOperator((n, n), matvec=lambda v: inv_sqrt * affinity(inv_sqrt * v), dtype=float)
    start = np.random.RandomState(random_state).uniform(-1, 1, n)
    values, vectors = eigsh(operator, k=n_components, which='LA', v0=start)
    return vectors[:, np.argsort(values)[::-1]] * inv_sqrt[:, None]


def cluster(forest: Any, X_train: np.ndarray, target_rows: np.ndarray, variant: str, n_clusters: int,
            random_state: Optional[int] = None) -> np.ndarray:
    embedding = spectral_embedding(forest, X_train, target_rows, variant, n_clusters, random_state)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(embedding)
