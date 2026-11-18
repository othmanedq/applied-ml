import pandas as pd

from .models import MODEL_NAMES, build_pipeline
from .tuning import SCORING, cross_validated_scores, summarize_cv

DECISION_SCORING = {**SCORING, "precision_at_0.5": "precision", "recall_at_0.5": "recall"}


def compare_class_weights(X, y, config):
    """CV scores of each model with default settings, with and without class weights."""
    rows = []
    for name in MODEL_NAMES:
        for class_weight in [None, "balanced"]:
            pipeline = build_pipeline(name, {"class_weight": class_weight}, config["seed"])
            scores = cross_validated_scores(pipeline, X, y, config, DECISION_SCORING)
            rows.append({"model": name, "class_weight": class_weight or "none", **summarize_cv(scores)})
    return pd.DataFrame(rows)
