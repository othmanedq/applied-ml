import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_table(y_true, proba, n_bins=10):
    """Mean predicted probability and observed rate in equal-width probability bins."""
    bins = pd.cut(proba, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = (
        pd.DataFrame({"observed": np.asarray(y_true), "predicted": proba, "bin": bins})
        .groupby("bin", observed=True)
        .agg(
            count=("observed", "size"),
            mean_predicted=("predicted", "mean"),
            observed_rate=("observed", "mean"),
        )
    )
    table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


def summarize(y_true, proba, table):
    weights = table["count"] / table["count"].sum()
    return {
        "brier": brier_score_loss(y_true, proba),
        "ece": (weights * table["gap"].abs()).sum(),
        "mean_predicted": proba.mean(),
        "observed_rate": np.mean(y_true),
    }


def plot_reliability(tables, proba, split):
    """Reliability curve of each split, and histogram of the predicted probabilities of one split."""
    fig, (curve_ax, hist_ax) = plt.subplots(
        2, 1, figsize=(6, 7), height_ratios=[3, 1], sharex=True
    )
    curve_ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name, table in tables.items():
        curve_ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name.capitalize())
    curve_ax.set(ylabel="Observed subscription rate", title="Reliability curve of the final model")
    curve_ax.legend()
    hist_ax.hist(proba, bins=50)
    hist_ax.set(xlabel="Predicted probability", ylabel=f"Clients ({split})")
    fig.tight_layout()
    return fig
