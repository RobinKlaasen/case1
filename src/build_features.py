"""
Feature engineering for Case 1.

Produces the feature matrix X and target vector y from the cleaned DataFrame.
Feature lists and exclusions are driven by config.yaml so that participants
can add or remove features by editing config, not code.

Assumptions:
- Input is the output of clean.clean().
- Categorical features are one-hot encoded with drop_first=True to avoid
  the dummy variable trap for linear models.
- "Premium cardholder" is defined by card_type values listed in config
  under premium.card_types. This definition is documented here and in the
  write-up so the committee can challenge it.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target y from the cleaned DataFrame.

    Returns:
        X: DataFrame of model-ready features (numeric + encoded categoricals)
        y: boolean Series of fraud labels
    """
    feat = config["features"]

    # Assemble raw feature columns (before encoding)
    numeric_cols = [c for c in feat["numeric"] if c in df.columns]
    categorical_cols = [c for c in feat["categorical"] if c in df.columns]
    binary_cols = [c for c in feat["binary"] if _binary_col(c) in df.columns]

    # Map config feature names to actual (possibly prefixed) column names
    col_map = _build_column_map(df)
    numeric_actual = [col_map.get(c, c) for c in numeric_cols if col_map.get(c, c) in df.columns]
    categorical_actual = [col_map.get(c, c) for c in categorical_cols if col_map.get(c, c) in df.columns]
    binary_actual = [col_map.get(c, c) for c in binary_cols if col_map.get(c, c) in df.columns]

    # Bucket infrequent MCC categories before one-hot encoding to avoid
    # ~100 dummies. top_mcc_categories from config controls how many to keep.
    df = df.copy()
    top_n = feat.get("top_mcc_categories", 15)
    if "mcc_category" in df.columns:
        df["mcc_category"] = _bucket_mcc(df["mcc_category"], top_n)

    X_num = df[numeric_actual].copy()
    X_bin = df[binary_actual].astype(float)
    X_cat = pd.get_dummies(df[categorical_actual], drop_first=True, dtype=float)

    X = pd.concat([X_num, X_bin, X_cat], axis=1)
    y = df["is_fraud"].astype(int)

    return X, y


def mark_premium(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Return a boolean Series indicating premium cardholders, using the
    definition in config.yaml (premium.card_types).

    This is kept separate from build_features() so evaluation code can
    slice metrics by premium status without it becoming a model feature.
    """
    premium_types = config["premium"]["card_types"]
    card_type_col = _build_column_map(df).get("card_type", "card_card_type")
    if card_type_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[card_type_col].isin(premium_types)


def _binary_col(name: str) -> str:
    """Map config binary feature name to actual DataFrame column name."""
    return _build_column_map_static().get(name, name)


def _build_column_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from config feature names to actual DataFrame column names.
    After the merge in load_raw.py, card and user columns are prefixed.
    """
    return _build_column_map_static()


def _bucket_mcc(mcc_series: pd.Series, top_n: int) -> pd.Series:
    """Replace infrequent MCC categories with 'Other' to limit dummy explosion."""
    top_cats = mcc_series.value_counts().head(top_n).index
    return mcc_series.where(mcc_series.isin(top_cats), other="Other")


def _build_column_map_static() -> dict:
    """Static mapping of config names → merged DataFrame column names."""
    return {
        # numeric
        "yearly_income": "user_yearly_income",
        "current_age": "user_current_age",
        "credit_score": "user_credit_score",
        "credit_limit": "card_credit_limit",
        # categorical
        "card_type": "card_card_type",
        "card_brand": "card_card_brand",
        # binary
        "has_chip": "card_has_chip",
        "card_on_dark_web": "card_card_on_dark_web",
    }
