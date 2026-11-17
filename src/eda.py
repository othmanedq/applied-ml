import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TARGET


def column_types(data):
    numeric = data.drop(columns=TARGET).select_dtypes("number").columns.tolist()
    categorical = data.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def overview(data):
    return pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing": data.isna().sum(),
        "unknown": (data == "unknown").sum(),
        "unique_values": data.nunique(),
        "example": data.iloc[0],
    }).rename_axis("column")


def numeric_summary(data, columns):
    summary = data[columns].describe().T
    q1, q3 = summary["25%"], summary["75%"]
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    summary["outliers_iqr"] = ((data[columns] < low) | (data[columns] > high)).sum()
    summary["skew"] = data[columns].skew()
    return summary.rename_axis("column")


def subscription_rate_table(data, columns):
    """Count, share and subscription rate of every category."""
    table = pd.concat(
        {
            column: data.groupby(column)[TARGET].agg(count="size", subscription_rate="mean")
            for column in columns
        },
        names=["column", "category"],
    )
    table.insert(1, "share", table["count"] / len(data))
    return table


def time_drift_table(data, n_blocks=10):
    """Subscription rate and share of unknown contact type in blocks of consecutive rows.

    The rows are in time order.
    """
    ordered = data.sort_index()
    blocks = np.array_split(np.arange(len(ordered)), n_blocks)
    return pd.DataFrame({
        "block": range(1, n_blocks + 1),
        "rows": [len(block) for block in blocks],
        "subscription_rate": [ordered[TARGET].iloc[block].mean() for block in blocks],
        "contact_unknown_share": [(ordered["contact"].iloc[block] == "unknown").mean() for block in blocks],
    })


def _grid(n_plots, n_columns=3, height=3.4):
    n_rows = int(np.ceil(n_plots / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, height * n_rows))
    axes = np.ravel(axes)
    for ax in axes[n_plots:]:
        ax.axis("off")
    return fig, axes


def plot_target(data):
    counts = data[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["No", "Yes"], counts.to_numpy())
    for x, count in enumerate(counts):
        ax.text(x, count, f"{count:,} ({count / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title="Did the client subscribe?", ylabel="Clients")
    return fig


def plot_numeric(data, columns):
    """Distribution of each numeric feature for both classes.

    Very skewed features are cut at the 1st and 99th percentiles. Integer features with a small
    range get one bin per value.
    """
    fig, axes = _grid(len(columns))
    for ax, column in zip(axes, columns):
        values = data[column]
        if abs(values.skew()) > 1:
            low, high = values.quantile([0.01, 0.99])
        else:
            low, high = values.min(), values.max()
        if (values % 1 == 0).all() and high - low <= 100:
            bins = np.arange(low, high + 2) - 0.5
        else:
            bins = np.linspace(low, high, 41)
        for value, label in [(0, "No"), (1, "Yes")]:
            ax.hist(values[data[TARGET] == value], bins=bins, density=True, alpha=0.6, label=label)
        ax.set_title(column)
    axes[0].legend(title="Subscribed")
    fig.tight_layout()
    return fig


def plot_subscription_rate(table, overall_rate):
    """Subscription rate of every category, with its number of clients. Dashed line: overall rate."""
    columns = table.index.get_level_values("column").unique()
    fig, axes = _grid(len(columns), height=3.8)
    for ax, column in zip(axes, columns):
        rates = table.loc[column].sort_values("subscription_rate")
        labels = [f"{category} (n={count:,})" for category, count in zip(rates.index, rates["count"])]
        ax.barh(labels, rates["subscription_rate"].to_numpy())
        ax.axvline(overall_rate, color="black", linestyle="--")
        ax.set_title(column)
    fig.tight_layout()
    return fig


def plot_correlation(data, columns):
    """Spearman correlation, because several features are very skewed."""
    corr = data[columns + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image)
    fig.tight_layout()
    return fig


def plot_time_drift(table):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["block"], table["subscription_rate"], marker="o", label="Subscription rate")
    ax.plot(table["block"], table["contact_unknown_share"], marker="o", label="Share of contact = unknown")
    ax.set(xlabel="Block of rows (oldest to newest)", ylabel="Share of clients", title="Changes over time")
    ax.legend()
    return fig
