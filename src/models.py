from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import FeatureEngineer, build_preprocessor

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]

# Family of every model compared with default settings before the tuning.
FAMILIES = {
    "logistic_regression": "linear",
    "knn": "nearest neighbours",
    "naive_bayes": "naive Bayes",
    "decision_tree": "single tree",
    "random_forest": "bagging",
    "extra_trees": "bagging",
    "lightgbm": "boosting",
    "hist_gradient_boosting": "boosting",
}

# Only these models are sensitive to the scale of numeric features.
SCALED = {"logistic_regression", "knn"}


def make_model(name, params, seed):
    if name == "baseline":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        # More trees only lower the variance, so the number of trees is fixed, not tuned.
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "naive_bayes":
        return GaussianNB(**params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")


def predict(model, X):
    """Probability of subscription."""
    return model.predict_proba(X)[:, 1]


def build_pipeline(name, params, seed):
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(scale_numeric=name in SCALED)),
        ("model", make_model(name, params, seed)),
    ])


def suggest_params(trial, name):
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.7),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        }
    raise ValueError(f"Unknown model: {name}")
