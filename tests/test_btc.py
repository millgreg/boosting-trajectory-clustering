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


def test_clean_features_does_not_invent_zero_values():
    import pandas as pd
    from experiments.dataset_loader import clean_features
    frame = pd.DataFrame({'lab': [1.0, 3.0, np.nan, 5.0], 'mostly_missing': [np.nan, np.nan, np.nan, 2.0]})
    cleaned = clean_features(frame)
    assert cleaned.shape == (4, 1)
    assert cleaned[2, 0] == 3.0


def test_multiclass_booster_is_rejected():
    X, y = make_classification(200, 12, n_informative=6, n_classes=3, random_state=0)
    gb = GradientBoostingClassifier(n_estimators=5, max_depth=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match='binary'):
        BTC(booster=gb, n_clusters=3).fit(X)


def test_clean_features_drops_id_like_columns():
    import pandas as pd
    from experiments.dataset_loader import clean_features
    frame = pd.DataFrame({'value': np.arange(60.0), 'identifier': [f'id{i}' for i in range(60)], 'group': ['a', 'b'] * 30})
    assert clean_features(frame).shape == (60, 2)


def test_primary_tumor_mask():
    from experiments.dataset_loader import primary_tumor_mask
    assert primary_tumor_mask(['TCGA-AA-0001-01', 'TCGA-AA-0001-11', 'TCGA-AA-0002-06']).tolist() == [True, False, False]


def test_factorized_rfc_matches_dense_spectral_embedding():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.manifold import spectral_embedding
    from baselines.rfc import RFC
    from baselines.rfc_factorized import FactorizedRFC
    X, y = make_classification(300, 10, n_informative=5, random_state=1)
    forest = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    dense = RFC(rf_model=forest, n_clusters=3, random_state=0).fit(X)
    fast = FactorizedRFC(rf_model=forest, n_clusters=3, random_state=0).fit(X)
    reference = spectral_embedding(dense.similarity_matrix_, n_components=3, random_state=0, drop_first=False)
    q1, _ = np.linalg.qr(reference)
    q2, _ = np.linalg.qr(fast.embedding_)
    assert np.linalg.svd(q1.T @ q2, compute_uv=False).min() > 0.999


@pytest.mark.parametrize('library', ['xgboost', 'lightgbm'])
def test_other_boosters_reconstruct_final_margin(library):
    X, y = make_classification(300, 12, n_informative=6, random_state=0)
    if library == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=10, max_depth=1, verbosity=0).fit(X, y)
        final = model.predict(X, output_margin=True)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=10, max_depth=1, verbose=-1).fit(X, y)
        final = model.predict(X, raw_score=True)
    btc = BTC(booster=model, n_clusters=2, random_state=0).fit(X)
    from btc import extract_raw_init
    np.testing.assert_allclose(extract_raw_init(model, X) + btc.trajectories_.sum(axis=1), final, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('variant', ['breiman', 'oob', 'rfgap'])
def test_proximity_factorization_matches_brute_force(variant):
    from sklearn.ensemble import RandomForestClassifier
    from baselines import rf_proximities as rp
    X, y = make_classification(120, 6, n_informative=4, random_state=2)
    forest = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, y)
    target = np.where(y == 1)[0]
    leaves = forest.apply(X)
    counts = np.stack([np.bincount(s, minlength=len(X)) for s in forest.estimators_samples_], axis=1)
    oob = counts == 0
    n = len(target)
    expected = np.zeros((n, n))
    for a, i in enumerate(target):
        for b, j in enumerate(target):
            same = leaves[i] == leaves[j]
            if variant == 'breiman':
                expected[a, b] = same.mean()
            elif variant == 'oob':
                expected[a, b] = (oob[i] & oob[j] & same).sum() / np.sqrt(oob[i].sum() * oob[j].sum())
            else:
                total = sum(counts[j, t] / counts[leaves[:, t] == leaves[i, t], t].sum()
                            for t in np.where(oob[i] & same)[0])
                expected[a, b] = total / oob[i].sum()
    if variant == 'rfgap':
        expected = (expected + expected.T) / 2
    np.testing.assert_allclose(rp.dense_proximity(forest, X, target, variant), expected, atol=1e-12)
