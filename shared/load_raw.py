"""
Load and merge the five raw Kaggle files into a single analysis-ready DataFrame.

Assumptions:
- Raw files live in the directory pointed to by raw_dir (from config.yaml).
- train_fraud_labels.json is structured as {"target": {str(tx_id): "Yes"/"No"}}.
  Labels cover only a subset of all transactions (the training set).
  Transactions without a label are dropped here — they cannot be used for
  supervised learning without a ground-truth label.
- The join chain is:
    transactions  →  cards   (transactions.card_id == cards.id)
    cards         →  users   (cards.client_id == users.id)
  One card belongs to one user; one user may have many cards.
- mcc_codes.json maps integer MCC codes to human-readable category strings.
  We join this to transactions so downstream code works with category names,
  not opaque integers.
"""

import json
from pathlib import Path

import pandas as pd


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    """Load transactions_data.csv. Returns raw DataFrame with original dtypes."""
    return pd.read_csv(raw_dir / "transactions_data.csv")


def load_users(raw_dir: Path) -> pd.DataFrame:
    """Load users_data.csv. Returns raw DataFrame with original dtypes."""
    return pd.read_csv(raw_dir / "users_data.csv")


def load_cards(raw_dir: Path) -> pd.DataFrame:
    """Load cards_data.csv. Returns raw DataFrame with original dtypes."""
    return pd.read_csv(raw_dir / "cards_data.csv")


def load_fraud_labels(raw_dir: Path) -> pd.Series:
    """
    Load train_fraud_labels.json and return a boolean Series indexed by
    transaction id (integer).

    The JSON has an outer key "target" whose value is {str(tx_id): "Yes"/"No"}.
    We convert to int index and bool values so it can be merged with transactions.
    """
    with open(raw_dir / "train_fraud_labels.json") as f:
        raw = json.load(f)
    labels = pd.Series(raw["target"], name="is_fraud")
    labels.index = labels.index.astype(int)
    labels = labels.map({"Yes": True, "No": False})
    return labels


def load_mcc_codes(raw_dir: Path) -> dict:
    """Load mcc_codes.json and return a dict mapping int MCC code → category string."""
    with open(raw_dir / "mcc_codes.json") as f:
        raw = json.load(f)
    # Keys may be strings; normalise to int for joining with transactions.mcc
    return {int(k): v for k, v in raw.items()}


def load_and_merge(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load all five raw files and return a single merged DataFrame of labelled
    transactions, ready for cleaning and feature engineering.

    Rows: one per labelled transaction (~subset of all transactions).
    Columns: all transaction fields + all card fields (prefixed card_)
             + all user fields (prefixed user_) + mcc_category + is_fraud.

    Columns dropped during merge to avoid ambiguity:
    - cards.id        (same as transactions.card_id)
    - users.id        (same as cards.client_id)
    - cards.client_id (redundant after merge)
    """
    raw_dir = Path(raw_dir)

    transactions = load_transactions(raw_dir)
    users = load_users(raw_dir)
    cards = load_cards(raw_dir)
    labels = load_fraud_labels(raw_dir)
    mcc_codes = load_mcc_codes(raw_dir)

    # Keep only labelled transactions
    transactions = transactions[transactions["id"].isin(labels.index)].copy()
    transactions["is_fraud"] = transactions["id"].map(labels)

    # Add human-readable MCC category
    transactions["mcc_category"] = transactions["mcc"].map(mcc_codes).fillna("Unknown")

    # Merge cards onto transactions
    cards_renamed = cards.rename(
        columns={col: f"card_{col}" for col in cards.columns if col != "id"}
    )
    cards_renamed = cards_renamed.rename(columns={"id": "card_id"})
    df = transactions.merge(cards_renamed, on="card_id", how="left")

    # Merge users onto the result
    users_renamed = users.rename(
        columns={col: f"user_{col}" for col in users.columns if col != "id"}
    )
    users_renamed = users_renamed.rename(columns={"id": "card_client_id"})
    df = df.merge(users_renamed, left_on="card_client_id", right_on="card_client_id", how="left")

    return df
