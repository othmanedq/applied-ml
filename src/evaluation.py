import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data import TARGET, features_and_target
from .models import build_pipeline, predict

BOOTSTRAP_METRICS = {"average_precision": average_precision_score, "roc_auc": roc_auc_score}


def best_threshold(y_true, proba):
    """Threshold with the best balanced accuracy, which is also the largest TPR - FPR.

    This counts one missed subscriber as costly as (non-subscribers / subscribers) useless calls.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    # roc_curve starts with an infinite threshold, so we cap it at 1.
    return float(min(thresholds[np.argmax(tpr - fpr)], 1.0))


def decision_metrics(y_true, predicted):
    return {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
    }


def compute_metrics(y_true, proba, threshold):
    return {
        "average_precision": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "threshold": threshold,
        **decision_metrics(y_true, (proba >= threshold).astype(int)),
    }


def metrics_table(y_true, probas, thresholds):
    rows = {name: compute_metrics(y_true, proba, thresholds[name]) for name, proba in probas.items()}
    return pd.DataFrame(rows).T.rename_axis("model")


def plot_curves(y_true, probas):
    """ROC and precision-recall curves. The dashed lines show a model with no skill."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in probas.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=roc_ax)
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=pr_ax)
    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="No skill")
    pr_ax.axhline(np.mean(y_true), color="grey", linestyle="--", label="No skill")
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    roc_ax.legend()
    pr_ax.legend()
    fig.tight_layout()
    return fig


def plot_threshold(y_true, proba, threshold):
    precision, recall, pr_thresholds = precision_recall_curve(y_true, proba)
    fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pr_thresholds, precision[:-1], label="Precision")
    ax.plot(pr_thresholds, recall[:-1], label="Recall")
    ax.plot(roc_thresholds[1:], (tpr[1:] + 1 - fpr[1:]) / 2, label="Balanced accuracy")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Chosen threshold = {threshold:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    return fig


def time_split(splits):
    """Oldest and newest rows of train + validation, with the sizes of train and validation.

    The rows are in time order (May 2008 to November 2010).
    """
    development = pd.concat([splits["train"], splits["validation"]]).sort_index()
    n_train = len(splits["train"])
    return development.iloc[:n_train], development.iloc[n_train:]


def predict_newest(pipeline, splits):
    """Fit the pipeline on the oldest rows and predict the newest ones."""
    oldest, newest = time_split(splits)
    X_new, _ = features_and_target(newest)
    return predict(pipeline.fit(*features_and_target(oldest)), X_new)


def time_split_predictions(splits, best_params, seed):
    """Fit every model on the oldest rows and predict the newest ones."""
    oldest, newest = time_split(splits)
    probas = {
        name: predict_newest(build_pipeline(name, params, seed), splits)
        for name, params in best_params.items()
    }
    return oldest, newest[TARGET], probas


def split_scores(fit_part, y_eval, probas):
    """Scores of each model on one evaluation set, with the subscription rate of the fitting set."""
    rows = {
        name: {
            "fit_subscription_rate": fit_part[TARGET].mean(),
            "eval_subscription_rate": y_eval.mean(),
            "mean_predicted": proba.mean(),
            "roc_auc": roc_auc_score(y_eval, proba),
            "average_precision": average_precision_score(y_eval, proba),
        }
        for name, proba in probas.items()
    }
    return pd.DataFrame(rows).T.rename_axis("model")


def bootstrap_intervals(y_true, probas, reference, n_resamples, seed):
    """95% bootstrap intervals of the scores of each model, and of its gap to the reference model.

    Every resample draws the same clients for all models, so the gaps are paired.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, len(y_true), len(y_true))
        draws.append({
            (name, metric): score(y_true[rows], proba[rows])
            for name, proba in probas.items()
            for metric, score in BOOTSTRAP_METRICS.items()
        })
    draws = pd.DataFrame(draws)
    table = {}
    for name in probas:
        table[name] = {}
        for metric in BOOTSTRAP_METRICS:
            gap = draws[name, metric] - draws[reference, metric]
            low, high = draws[name, metric].quantile([0.025, 0.975])
            gap_low, gap_high = gap.quantile([0.025, 0.975])
            table[name].update({
                f"{metric}_low": low, f"{metric}_high": high,
                f"{metric}_gap_low": gap_low, f"{metric}_gap_high": gap_high,
            })
    return pd.DataFrame(table).T.rename_axis("model")
