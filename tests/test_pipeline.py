from typing import get_args

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from src import api
from src.calibration import calibration_table, summarize
from src.data import TARGET, features_and_target, load_data, split_data
from src.ablation import add_age_band, add_call_date, log_skewed
from src.evaluation import best_threshold, bootstrap_intervals, time_split
from src.fairness import equal_opportunity_thresholds, wilson_interval
from src.features import FeatureEngineer
from src.models import FAMILIES, build_pipeline, predict
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def data(config):
    return load_data(config)


@pytest.fixture(scope="module")
def splits(data, config):
    return split_data(data, config)


def test_data_has_no_leak_and_a_binary_target(data):
    assert "duration" not in data.columns
    assert set(data[TARGET].unique()) == {0, 1}
    assert data.isna().sum().sum() == 0


def test_split_is_disjoint_and_stratified(data, splits):
    indexes = [set(part.index) for part in splits.values()]
    assert set.union(*indexes) == set(data.index)
    assert sum(len(index) for index in indexes) == len(data)
    assert [len(part) / len(data) for part in splits.values()] == pytest.approx([0.6, 0.2, 0.2], abs=0.001)
    rates = [part[TARGET].mean() for part in splits.values()]
    assert max(rates) - min(rates) < 0.005


def test_feature_engineering():
    X = pd.DataFrame({
        "housing": ["no", "yes", "no"], "loan": ["no", "no", "yes"],
        "campaign": [1, 2, 3], "previous": [0, 3, 1],
    })
    features = FeatureEngineer().transform(X)
    assert features["has_loan"].tolist() == [0, 1, 1]
    assert features["contacted_before"].tolist() == [0, 1, 1]


@pytest.mark.parametrize("name", ["baseline", *FAMILIES])
def test_pipeline_returns_probabilities(data, name):
    X, y = features_and_target(data.sample(2000, random_state=0))
    proba = predict(build_pipeline(name, {}, seed=0).fit(X, y), X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_time_split_keeps_the_sizes_and_the_order(splits):
    oldest, newest = time_split(splits)
    assert len(oldest) == len(splits["train"])
    assert len(newest) == len(splits["validation"])
    assert oldest.index.max() < newest.index.min()


def test_best_threshold_separates_perfect_scores():
    threshold = best_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < threshold <= 0.8


def test_extra_features(config):
    X = pd.DataFrame({
        "age": [25, 45, 70], "month": ["may", "may", "jun"], "day": [5.0, 5.0, 30.0],
        "balance": [-100.0, 0.0, 5000.0], "campaign": [1, 2, 30], "pdays": [-1, 10, 300],
        "previous": [0, 1, 20],
    })
    assert add_age_band(X, config)["age_band"].tolist() == ["18-29", "30-59", "60+"]
    assert add_call_date(X)["call_date"].tolist() == ["may-5", "may-5", "jun-30"]
    logged = log_skewed(X)
    assert logged["balance"].is_monotonic_increasing and logged["balance"].iloc[0] < 0
    assert logged["age"].equals(X["age"])


def test_bootstrap_intervals_contain_the_score_and_pair_the_gaps():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    probas = {"good": np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1), "random": rng.random(500)}
    table = bootstrap_intervals(y, probas, "good", n_resamples=200, seed=0)
    score = roc_auc_score(y, probas["good"])
    assert table.loc["good", "roc_auc_low"] < score < table.loc["good", "roc_auc_high"]
    assert table.loc["good", ["roc_auc_gap_low", "roc_auc_gap_high"]].tolist() == [0, 0]
    assert table.loc["random", "roc_auc_gap_high"] < 0


def test_wilson_interval_known_values():
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_perfect_predictions_have_no_calibration_error():
    y, proba = np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])
    assert summarize(y, proba, calibration_table(y, proba))["ece"] == 0


def test_group_thresholds_give_the_same_recall():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b"], 1000)
    y = rng.integers(0, 2, 2000)
    proba = rng.random(2000) * np.where(groups == "a", 1.0, 0.5)
    thresholds = equal_opportunity_thresholds(y, proba, groups, threshold=0.4)
    predicted = proba >= np.vectorize(thresholds.get)(groups)
    recalls = [predicted[(groups == group) & (y == 1)].mean() for group in ["a", "b"]]
    assert recalls[0] == pytest.approx(recalls[1], abs=0.01)


def test_api_categories_match_the_training_data(data):
    for column in data.select_dtypes(exclude="number").columns:
        assert set(get_args(api.Client.model_fields[column].annotation)) == set(data[column])


def test_api_predicts_and_rejects_bad_input(data, monkeypatch):
    X, y = features_and_target(data.sample(2000, random_state=0))
    pipeline = build_pipeline("logistic_regression", {}, seed=0).fit(X, y)
    monkeypatch.setattr(api, "final_model", lambda: {"name": "test", "pipeline": pipeline, "threshold": 0.5})
    client = TestClient(api.app)
    clients = X.head(2).to_dict(orient="records")

    response = client.post("/predict", json=clients)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert client.post("/predict", json=[]).status_code == 422
    assert client.post("/predict", json=[{**clients[0], "housing": "Yes"}]).status_code == 422
