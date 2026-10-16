"""
Case 1 data loader.

Wraps shared/load_raw.py and handles the one Case 1-specific concern:
the merged dataset is large (~1.2 GB transactions), so we work only with
the labelled subset and optionally sample it for fast iteration.

Import pattern from a notebook in notebooks/:
    import sys; sys.path.insert(0, "..")
    from shared.load_raw import load_and_merge
    from src.load import load_case1_data
"""

from pathlib import Path

import pandas as pd


def load_case1_data(
    raw_dir: str | Path,
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load, merge, and optionally sample the labelled transaction dataset.

    Args:
        raw_dir:      Path to data/raw/
        sample_n:     If set, return a stratified random sample of this size.
                      Useful during development to iterate quickly.
                      Set to None for the full dataset in final runs.
        random_state: For reproducible sampling.

    Returns:
        Merged DataFrame with all five files joined, labelled rows only.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from shared.load_raw import load_and_merge

    df = load_and_merge(raw_dir)

    if sample_n is not None and sample_n < len(df):
        fraud = df[df["is_fraud"]]
        legit = df[~df["is_fraud"]]
        n_fraud = max(1, int(sample_n * len(fraud) / len(df)))
        n_legit = sample_n - n_fraud
        df = pd.concat([
            fraud.sample(min(len(fraud), n_fraud), random_state=random_state),
            legit.sample(min(len(legit), n_legit), random_state=random_state),
        ])
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df
