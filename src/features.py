from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add two business features. Nothing is learned from the data, so nothing can leak."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["has_loan"] = ((X["housing"] == "yes") | (X["loan"] == "yes")).astype(int)
        X["contacted_before"] = (X["previous"] > 0).astype(int)
        return X


def one_hot_name(feature, category):
    return f"{feature}={category}"


def build_preprocessor(scale_numeric):
    """Scale the numeric features if asked, and one-hot encode the categories.

    The data has no missing values: "unknown" is kept as its own category.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    categorical = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, feature_name_combiner=one_hot_name
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )
