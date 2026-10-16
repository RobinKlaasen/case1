"""
Clean the merged DataFrame produced by shared/load_raw.py.

Cleaning steps performed here:
1. Parse dollar-formatted strings ("$1,234.56") to float for amount,
   yearly_income, per_capita_income, total_debt, credit_limit.
2. Parse the date column to datetime and extract temporal features
   (transaction_hour, transaction_dow).
3. Derive is_foreign_state: whether the merchant state differs from the
   user's inferred home state.
4. Normalise binary YES/NO columns (has_chip, card_on_dark_web) to bool.
5. Drop rows where is_fraud is null (should not happen after load_and_merge,
   but defensive).

Assumptions:
- Dollar strings may be negative (e.g. "$-77.00" = a reversal/refund).
  We keep negatives as-is; whether to exclude reversals is a modelling
  decision, not a cleaning decision.
- "Home state" is inferred from user_address (last two characters of the
  state abbreviation embedded in the address string). This is imperfect;
  document the limitation in the write-up.
- merchant_state = "ONLINE" or similar non-state values exist; these are
  treated as foreign-state transactions by default.
"""

import re

import pandas as pd


def parse_dollar(series: pd.Series) -> pd.Series:
    """Convert "$1,234.56" or "$-77.00" strings to float. NaN-safe."""
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )


def extract_home_state(address_series: pd.Series) -> pd.Series:
    """
    Extract US state abbreviation from address strings like
    "462 Rose Lane, Somecity CA 91234".
    Returns a Series of two-letter state codes, or NaN if unparseable.
    """
    return address_series.str.extract(r"\b([A-Z]{2})\s+\d{5}", expand=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a new DataFrame.
    Does not modify the input in place.
    """
    df = df.copy()

    # 1. Parse dollar strings
    for col in ["amount", "user_yearly_income", "user_per_capita_income",
                "user_total_debt", "card_credit_limit"]:
        if col in df.columns:
            df[col] = parse_dollar(df[col])

    # 2. Parse date and extract temporal features
    df["date"] = pd.to_datetime(df["date"])
    df["transaction_hour"] = df["date"].dt.hour
    df["transaction_dow"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # 3. Derive is_foreign_state
    df["user_home_state"] = extract_home_state(df["user_address"])
    df["is_foreign_state"] = df["merchant_state"] != df["user_home_state"]
    # Transactions with unknown home state are conservatively flagged as foreign
    df["is_foreign_state"] = df["is_foreign_state"].fillna(True)

    # 4. Normalise binary YES/NO columns
    for col in ["card_has_chip", "card_card_on_dark_web"]:
        if col in df.columns:
            df[col] = df[col].map({"YES": True, "NO": False, "Yes": True, "No": False})

    # 5. Drop unlabelled rows (defensive)
    df = df.dropna(subset=["is_fraud"])

    return df
