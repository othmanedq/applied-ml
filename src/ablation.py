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


def add_age_band(X, config):
    """Age as three categories, so that a linear model can follow its U-shaped effect."""
    return X.assign(age_band=age_group(X, config))


def log_skewed(X):
    """Signed log of the skewed numeric features. It keeps their order, so trees barely change."""
    return X.assign(**{column: np.sign(X[column]) * np.log1p(X[column].abs()) for column in SKEWED})


def add_call_date(X):
    """Month and day as one category: the date of the call, without the year."""
    return X.assign(call_date=X["month"] + "-" + X["day"].astype(int).astype(str))


def _versions(name, params, config):
    """The tuned pipeline, the same without the engineered features, and with each extra feature."""
    band = ("age_band", FunctionTransformer(add_age_band, kw_args={"config": config}))
    logs = ("log_skewed", FunctionTransformer(log_skewed))
    date = ("call_date", FunctionTransformer(add_call_date))
    steps = build_pipeline(name, params, config["seed"]).steps
    return {
        "original columns": Pipeline(steps).set_params(features="passthrough"),
        "pipeline": Pipeline(steps),
        "pipeline + age_band": Pipeline([band, *steps]),
        "pipeline + log_skewed": Pipeline([logs, *steps]),
        "pipeline + call_date": Pipeline([date, *steps]),
    }


def _cv_average_precision(pipeline, X, y, config):
    scores = cross_validated_scores(pipeline, X, y, config)["average_precision"]
    return {"cv_average_precision": scores.mean(), "cv_average_precision_std": scores.std()}


def feature_ablation(X, y, best_params, config):
    """CV score of each tuned model without the engineered features, and with each extra feature."""
    rows = [
        {"model": name, "features": version, **_cv_average_precision(pipeline, X, y, config)}
        for name in MODEL_NAMES
        for version, pipeline in _versions(name, best_params[name], config).items()
    ]
    return pd.DataFrame(rows)


def time_proxy_ablation(splits, best_params, config):
    """Each tuned model without contact, month or day: CV on train, and ROC-AUC on the time split."""
    X, y = features_and_target(splits["train"])
    oldest, newest = time_split(splits)
    X_old, y_old = features_and_target(oldest)
    X_new, y_new = features_and_target(newest)
    rows = []
    for name in MODEL_NAMES:
        for removed in [[], ["contact"], ["month"], ["day"]]:
            pipeline = build_pipeline(name, best_params[name], config["seed"])
            scores = _cv_average_precision(pipeline, X.drop(columns=removed), y, config)
            pipeline.fit(X_old.drop(columns=removed), y_old)
            proba = pipeline.predict_proba(X_new.drop(columns=removed))[:, 1]
            rows.append({
                "model": name,
                "removed": ", ".join(removed) or "nothing",
                **scores,
                "time_split_roc_auc": roc_auc_score(y_new, proba),
            })
    return pd.DataFrame(rows)
