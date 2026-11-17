import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import decision_metrics

RATES = ["selection_rate", "true_positive_rate", "false_positive_rate"]


def wilson_interval(successes, n, z=1.96):
    """95% Wilson interval of a proportion. It stays reliable for small groups."""
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half_width, center + half_width


def group_table(y_true, y_pred, groups):
    """Size, subscription rate and decision rates of each group, with 95% Wilson intervals."""
    data = pd.DataFrame({
        "y": np.asarray(y_true), "pred": np.asarray(y_pred), "group": np.asarray(groups)
    })
    rows = []
    for group, part in data.groupby("group"):
        subscribers, non_subscribers = part[part["y"] == 1], part[part["y"] == 0]
        row = {
            "group": group,
            "clients": len(part),
            "subscribers": len(subscribers),
            "subscription_rate": part["y"].mean(),
            "precision": part.loc[part["pred"] == 1, "y"].mean(),
        }
        subsets = {
            "selection_rate": part,
            "true_positive_rate": subscribers,
            "false_positive_rate": non_subscribers,
        }
        for rate, subset in subsets.items():
            row[rate] = subset["pred"].mean()
            row[f"{rate}_low"], row[f"{rate}_high"] = wilson_interval(subset["pred"].sum(), len(subset))
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(y_true, y_pred, table):
    """Largest gap between groups for each fairness definition, and overall performance."""
    gaps = table[RATES].max() - table[RATES].min()
    return {
        "demographic_parity_gap": gaps["selection_rate"],
        "equal_opportunity_gap": gaps["true_positive_rate"],
        "equalized_odds_gap": max(gaps["true_positive_rate"], gaps["false_positive_rate"]),
        **decision_metrics(y_true, y_pred),
    }


def equal_opportunity_thresholds(y_true, proba, groups, threshold):
    """One threshold per group, so that every group gets the same recall (true positive rate).

    The target recall is the overall recall of the single threshold. The threshold of a group is
    the score above which this share of its subscribers lies.
    """
    subscribers = np.asarray(y_true) == 1
    groups = np.asarray(groups)
    target_recall = np.mean(proba[subscribers] >= threshold)
    return {
        group: float(np.quantile(proba[subscribers & (groups == group)], 1 - target_recall))
        for group in np.unique(groups)
    }


def plot_groups(tables):
    fig, axes = plt.subplots(1, len(RATES), figsize=(15, 4.5), sharey=True)
    groups = next(iter(tables.values())).index
    width = 0.8 / len(tables)
    for ax, rate in zip(axes, RATES):
        for i, (version, table) in enumerate(tables.items()):
            positions = np.arange(len(groups)) + i * width
            errors = [table[rate] - table[f"{rate}_low"], table[f"{rate}_high"] - table[rate]]
            ax.bar(positions, table[rate], width, yerr=errors, capsize=3, label=version)
        ax.set_xticks(np.arange(len(groups)) + width * (len(tables) - 1) / 2, groups)
        ax.set_title(rate.replace("_", " ").capitalize())
    axes[0].set_ylabel("Rate with 95% Wilson interval")
    axes[0].legend()
    fig.tight_layout()
    return fig
