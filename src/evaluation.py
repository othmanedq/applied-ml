import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
from .models import build_pipeline


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


def plot_confusion_matrix(y_true, predicted):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, predicted, display_labels=["No", "Yes"], colorbar=False, ax=ax
    )
    fig.tight_layout()
    return fig


def time_split(splits):
    """Oldest and newest rows of train + validation, with the sizes of train and validation.

    The rows are in time order (May 2008 to November 2010).
    """
    development = pd.concat([splits["train"], splits["validation"]]).sort_index()
    n_train = len(splits["train"])
    return development.iloc[:n_train], development.iloc[n_train:]


def time_split_check(splits, final_pipeline, name, params, seed):
    """Compare the random split with a time split of the same size.

    The time split fits the same model on the oldest rows and scores it on the newest ones.
    """
    oldest, newest = time_split(splits)
    time_pipeline = build_pipeline(name, params, seed).fit(*features_and_target(oldest))
    return pd.DataFrame({
        "random": _split_scores(splits["train"], splits["validation"], final_pipeline),
        "time": _split_scores(oldest, newest, time_pipeline),
    }).T.rename_axis("split")


def _split_scores(fit_part, eval_part, pipeline):
    X_eval, y_eval = features_and_target(eval_part)
    proba = pipeline.predict_proba(X_eval)[:, 1]
    average_precision = average_precision_score(y_eval, proba)
    return {
        "fit_subscription_rate": fit_part[TARGET].mean(),
        "eval_subscription_rate": y_eval.mean(),
        "mean_predicted": proba.mean(),
        "roc_auc": roc_auc_score(y_eval, proba),
        "average_precision": average_precision,
        "ap_lift": average_precision / y_eval.mean(),
    }
