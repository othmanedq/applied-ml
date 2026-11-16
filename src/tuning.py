import json

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import FIGURES, MODELS, ROOT
from .models import FAMILIES, MODEL_NAMES, build_pipeline, suggest_params

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}
HISTORY_FIGURE = "tuning_history"


def cross_validated_scores(pipeline, X, y, config, scoring=SCORING):
    """Scores of each fold. The whole pipeline is fitted again inside every fold."""
    cv = StratifiedKFold(n_splits=config["tuning"]["cv_folds"], shuffle=True, random_state=config["seed"])
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    scores = {metric: results[f"test_{metric}"] for metric in scoring}
    scores["brier"] = -scores["brier"]
    return scores


def screen_models(X, y, config):
    """CV scores of every model of FAMILIES with default settings, best first."""
    rows = []
    for name, family in FAMILIES.items():
        scores = cross_validated_scores(build_pipeline(name, {}, config["seed"]), X, y, config)
        rows.append({
            "family": family,
            "model": name,
            **{metric: values.mean() for metric, values in scores.items()},
            "average_precision_std": scores["average_precision"].std(),
        })
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False, ignore_index=True)


def tune(name, X, y, config):
    """Optuna search. Each trial is scored by the mean average precision over the CV folds."""

    def objective(trial):
        pipeline = build_pipeline(name, suggest_params(trial, name), config["seed"])
        scores = cross_validated_scores(pipeline, X, y, config)
        trial.set_user_attr("cv_average_precision_std", float(scores["average_precision"].std()))
        trial.set_user_attr("cv_roc_auc", float(scores["roc_auc"].mean()))
        trial.set_user_attr("cv_brier", float(scores["brier"].mean()))
        return float(scores["average_precision"].mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"])
    )
    study.optimize(objective, n_trials=config["tuning"]["n_trials"])
    return study


def trials_table(study, name):
    """One row per trial: parameters, CV scores, duration and the saved files linked to it."""
    figure = (FIGURES / f"{HISTORY_FIGURE}.png").relative_to(ROOT)
    model_file = (MODELS / f"{name}.joblib").relative_to(ROOT)
    best = study.best_trial.number
    rows = []
    for trial in study.trials:
        rows.append({
            "model": name,
            "trial": trial.number,
            "params": json.dumps(trial.params),
            "cv_average_precision": trial.value,
            **trial.user_attrs,
            "seconds": round(trial.duration.total_seconds(), 1),
            "is_best": trial.number == best,
            "artifacts": f"{figure}; {model_file}" if trial.number == best else str(figure),
        })
    return pd.DataFrame(rows)


def plot_history(run_log):
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(15, 4), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        trials = run_log[run_log["model"] == name]
        ax.scatter(trials["trial"], trials["cv_average_precision"], s=15, label="Trial")
        ax.plot(trials["trial"], trials["cv_average_precision"].cummax(), color="black", label="Best so far")
        ax.set(title=name, xlabel="Trial")
    axes[0].set_ylabel("Mean CV average precision")
    axes[0].legend()
    fig.tight_layout()
    return fig
