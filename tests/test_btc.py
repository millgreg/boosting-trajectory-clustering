"""Regression tests for the BTC estimator API and trajectory construction."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

from btc import BTC


@pytest.fixture(scope="module")
def fitted():
    X, y = make_classification(300, 12, n_informative=6, random_state=0)
    gb = GradientBoostingClassifier(n_estimators=15, max_depth=1, random_state=0).fit(X, y)
    return X, y, gb


def test_sklearn_estimator_api():
    est = BTC(n_clusters=2)
    assert est.get_params()["n_clusters"] == 2
    assert clone(est).n_clusters == 2
    assert "BTC" in repr(est)


def test_trajectory_shape_and_labels(fitted):
    X, y, gb = fitted
    target = X[y == 1]
    labels = BTC(booster=gb, n_clusters=3, random_state=0).fit_predict(target)
    assert labels.shape == (len(target),)
    assert set(labels) <= {0, 1, 2}


def test_deltas_reconstruct_final_margin(fitted):
    X, _, gb = fitted
    model = BTC(booster=gb, n_clusters=2, random_state=0).fit(X)
    baseline = gb._raw_predict_init(X).ravel()
    np.testing.assert_allclose(baseline + model.trajectories_.sum(axis=1), gb.decision_function(X))
