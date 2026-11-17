import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .data import age_group, features_and_target
from .evaluation import time_split
from .models import MODEL_NAMES, build_pipeline
from .tuning import cross_validated_scores

SKEWED = ["balance", "campaign", "pdays", "previous"]
TIME_PROXIES = ["contact", "month", "day"]


def add_age_band(X, config):
    """Age as three categories, so that a linear model can follow its U-shaped effect."""
    return X.assign(age_band=age_group(X, config))


def log_skewed(X):
    """Signed log of the skewed numeric features. It keeps their order, so trees barely change."""
    return X.assign(**{column: np.sign(X[column]) * np.log1p(X[column].abs()) for column in SKEWED})


def add_call_date(X):
    """Month and day as one category: the date of the call, without the year."""
    return X.assign(call_date=X["month"] + "-" + X["day"].astype(int).astype(str))


def drop_columns(X, columns):
    return X.drop(columns=columns)


def _with_step(name, function, steps, **kw_args):
    """The pipeline with one more step in front, which changes the columns."""
    return Pipeline([(name, FunctionTransformer(function, kw_args=kw_args or None)), *steps])


def _scores(pipeline, splits, config):
    """CV average precision on train (same period) and ROC-AUC on the time split (newer calls)."""
    X, y = features_and_target(splits["train"])
    oldest, newest = time_split(splits)
    X_new, y_new = features_and_target(newest)
    scores = cross_validated_scores(pipeline, X, y, config)["average_precision"]
    proba = pipeline.fit(*features_and_target(oldest)).predict_proba(X_new)[:, 1]
    return {
        "cv_average_precision": scores.mean(),
        "cv_average_precision_std": scores.std(),
        "time_split_roc_auc": roc_auc_score(y_new, proba),
    }


def feature_ablation(splits, best_params, config):
    """Each tuned model without the engineered features, and with each extra feature."""
    rows = []
    for name in MODEL_NAMES:
        steps = build_pipeline(name, best_params[name], config["seed"]).steps
        versions = {
            "original columns": Pipeline(steps).set_params(features="passthrough"),
            "pipeline": Pipeline(steps),
            "pipeline + age_band": _with_step("age_band", add_age_band, steps, config=config),
            "pipeline + log_skewed": _with_step("log_skewed", log_skewed, steps),
            "pipeline + call_date": _with_step("call_date", add_call_date, steps),
        }
        rows += [
            {"model": name, "features": version, **_scores(pipeline, splits, config)}
            for version, pipeline in versions.items()
        ]
    return pd.DataFrame(rows)


def time_proxy_ablation(splits, best_params, config):
    """Each tuned model without contact, month or day."""
    rows = []
    for name in MODEL_NAMES:
        steps = build_pipeline(name, best_params[name], config["seed"]).steps
        versions = {
            "nothing": Pipeline(steps),
            **{column: _with_step("drop", drop_columns, steps, columns=[column]) for column in TIME_PROXIES},
        }
        rows += [
            {"model": name, "removed": removed, **_scores(pipeline, splits, config)}
            for removed, pipeline in versions.items()
        ]
    return pd.DataFrame(rows)
