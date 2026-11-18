import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# A random forest averages leaf probabilities. LightGBM and logistic regression are explained
# on their raw output, the log-odds: sigmoid(base value + sum of SHAP values) = probability.
SHAP_UNITS = {"logistic_regression": "log-odds", "random_forest": "probability", "lightgbm": "log-odds"}
TOP_FEATURES = 9


def compute_shap_values(pipeline, name, background, X):
    """SHAP values on the features after preprocessing.

    TreeExplainer is exact and fast for tree models. The linear model needs background data.
    """
    preprocess, model = pipeline[:-1], pipeline[-1]
    names = preprocess[-1].get_feature_names_out()
    if name == "logistic_regression":
        explainer = shap.LinearExplainer(model, preprocess.transform(background), feature_names=names)
    else:
        explainer = shap.TreeExplainer(model, feature_names=names)
    shap_values = explainer(preprocess.transform(X))
    # The random forest gives one column per class: we keep the positive class.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def global_importance(shap_values):
    """Mean absolute SHAP value of each original feature. One-hot columns are added together."""
    contributions = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    original = [name.split("=")[0] for name in contributions.columns]
    by_feature = contributions.T.groupby(original).sum().rename_axis("feature")
    return by_feature.abs().mean(axis=1).sort_values(ascending=False).rename("mean_abs_shap")


def pick_cases(y_true, proba, threshold):
    """Most confident true positive, false positive, false negative and true negative."""
    y_true = np.asarray(y_true)
    predicted = proba >= threshold
    masks = {
        "true_positive": (y_true == 1) & predicted,
        "false_positive": (y_true == 0) & predicted,
        "false_negative": (y_true == 1) & ~predicted,
        "true_negative": (y_true == 0) & ~predicted,
    }
    confidence = np.abs(proba - threshold)
    return {
        case: int(np.argmax(np.where(mask, confidence, -1)))
        for case, mask in masks.items()
        if mask.any()
    }


def case_table(shap_values, cases, proba, unit):
    """Largest SHAP contributions for each explained client."""
    tables = []
    for case, position in cases.items():
        contributions = pd.DataFrame({
            "feature": shap_values.feature_names,
            "value": shap_values.data[position],
            "shap": shap_values.values[position],
        })
        order = contributions["shap"].abs().sort_values(ascending=False).index
        tables.append(contributions.loc[order].head(TOP_FEATURES).assign(
            case=case, probability=proba[position], unit=unit
        ))
    return pd.concat(tables)[["case", "probability", "feature", "value", "shap", "unit"]]


def plot_beeswarm(shap_values, unit, top=15):
    shap.plots.beeswarm(shap_values, max_display=top, show=False)
    plt.gca().set_xlabel(f"SHAP value ({unit})")
    return plt.gcf()


def plot_waterfall(shap_values, title):
    # The last row of the waterfall adds up all the other features.
    shap.plots.waterfall(shap_values, max_display=TOP_FEATURES + 1, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    return fig
