"""
Preprocessing: Athlete-Level Splitting & Feature Scaling
=========================================================
Implements the leakage-safe data pipeline:

1. Athlete-level train/validation/test split
   - Every sequence from an athlete stays in the SAME split.
   - This prevents the model from memorizing athlete-specific patterns
     and then being tested on data from the same athlete.

2. Feature scaling
   - StandardScaler is fit ONLY on training data.
   - The same scaler is applied to validation and test data.
   - This prevents future/test information from leaking into training.

Usage:
    from src.preprocessing import prepare_splits
    train_df, val_df, test_df, scaler = prepare_splits()
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def athlete_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by athlete — no athlete appears in more than one split.

    Why athlete-level splitting?
        If we randomly split individual rows or sequences, the model could
        see training data from Athlete X in weeks 1-30 and then be tested
        on Athlete X's weeks 31-52. The model would effectively "know"
        this athlete's progression pattern, inflating test performance.
        This is a form of data leakage.

    Args:
        df: Full dataset with 'athlete_id' column.
        train_ratio: Fraction of athletes for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducible splits.

    Returns:
        (train_df, val_df, test_df) — each containing all rows for their athletes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    athlete_ids = df["athlete_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(athlete_ids)

    n = len(athlete_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_athletes = set(athlete_ids[:n_train])
    val_athletes = set(athlete_ids[n_train:n_train + n_val])
    test_athletes = set(athlete_ids[n_train + n_val:])

    train_df = df[df["athlete_id"].isin(train_athletes)].copy()
    val_df = df[df["athlete_id"].isin(val_athletes)].copy()
    test_df = df[df["athlete_id"].isin(test_athletes)].copy()

    print(f"Athlete-level split (seed={seed}):")
    print(f"  Train: {len(train_athletes)} athletes, {len(train_df)} rows")
    print(f"  Val:   {len(val_athletes)} athletes, {len(val_df)} rows")
    print(f"  Test:  {len(test_athletes)} athletes, {len(test_df)} rows")

    # Sanity check: no overlap
    assert train_athletes.isdisjoint(val_athletes), "Train/Val overlap!"
    assert train_athletes.isdisjoint(test_athletes), "Train/Test overlap!"
    assert val_athletes.isdisjoint(test_athletes), "Val/Test overlap!"

    return train_df, val_df, test_df


def fit_scaler(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_type: str = "standard",
) -> StandardScaler | MinMaxScaler:
    """Fit a scaler on training data only.

    Args:
        train_df: Training split DataFrame.
        feature_columns: List of column names to scale.
        scaler_type: "standard" for StandardScaler, "minmax" for MinMaxScaler.

    Returns:
        Fitted scaler object.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    scaler.fit(train_df[feature_columns].values)
    print(f"Fitted {scaler_type} scaler on {len(feature_columns)} features using training data.")
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame.

    Args:
        df: DataFrame to transform.
        scaler: Fitted scaler (from fit_scaler).
        feature_columns: Columns to scale.

    Returns:
        DataFrame with scaled feature columns (other columns unchanged).
    """
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns].values)
    return df


def prepare_splits(
    config_path: str = "config/config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Full preprocessing pipeline: load → clean → split → scale.

    Returns:
        (train_df, val_df, test_df, scaler) — all with scaled features.
    """
    cfg = load_config(config_path)

    processed_path = os.path.join(cfg["preprocessing"]["processed_dir"], "workout_features.csv")
    feature_cols = cfg["features"]["input_features"]
    target_col = cfg["features"]["target"]
    scaler_type = cfg["preprocessing"]["scaler"]
    seed = cfg["random_seed"]

    # Load processed data
    print(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)

    # Drop rows where target or key features are NaN
    # (early weeks without enough rolling data, last week without next-week target)
    required_cols = feature_cols + [target_col]
    initial_len = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} rows with NaN in required columns")
    print(f"Remaining: {len(df)} rows")

    # Ensure target is integer
    df[target_col] = df[target_col].astype(int)

    # Athlete-level split
    train_df, val_df, test_df = athlete_level_split(
        df,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        seed=seed,
    )

    # Fit scaler on training data only
    scaler = fit_scaler(train_df, feature_cols, scaler_type=scaler_type)

    # Apply scaler to all splits
    train_df = apply_scaler(train_df, scaler, feature_cols)
    val_df = apply_scaler(val_df, scaler, feature_cols)
    test_df = apply_scaler(test_df, scaler, feature_cols)
    print("Applied scaler to train, val, and test splits.")

    # Save scaler for reproducibility
    scaler_path = os.path.join(cfg["preprocessing"]["processed_dir"], "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    return train_df, val_df, test_df, scaler


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df, val_df, test_df, scaler = prepare_splits()

    print(f"\n{'='*50}")
    print("SPLIT SUMMARY")
    print(f"{'='*50}")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        target = split_df["target_plateau_next_week"]
        print(f"\n{name}:")
        print(f"  Rows: {len(split_df)}")
        print(f"  Athletes: {split_df['athlete_id'].nunique()}")
        print(f"  Target=0: {(target==0).sum()} ({(target==0).mean()*100:.1f}%)")
        print(f"  Target=1: {(target==1).sum()} ({(target==1).mean()*100:.1f}%)")
