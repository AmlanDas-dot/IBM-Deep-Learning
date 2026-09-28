"""
Feature Engineering & Plateau Labeling
=======================================
Adds rolling metrics, change features, and the plateau labels to the raw dataset.

Key decisions:
    - Plateau is defined by comparing rolling 3-week 1RM averages.
    - `plateau_label`: whether the athlete is currently in a plateau.
    - `target_plateau_next_week`: shifted label — the prediction target.
    - NO future information leaks into features.

Usage:
    python src/feature_engineering.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_rolling_features(
    df: pd.DataFrame,
    rolling_window: int = 3,
) -> pd.DataFrame:
    """Add rolling averages and change features per athlete per exercise.

    Features added:
        - estimated_1rm_rolling: rolling mean of estimated_1rm
        - volume_rolling: rolling mean of volume_kg
        - estimated_1rm_change_3w: pct change of rolling 1RM vs previous block
        - volume_change_3w: pct change of rolling volume vs previous block

    Args:
        df: DataFrame sorted by (athlete_id, exercise, week).
        rolling_window: Window size for rolling calculations.

    Returns:
        DataFrame with new columns appended.
    """
    df = df.copy()
    df = df.sort_values(["athlete_id", "exercise", "week"]).reset_index(drop=True)

    # Group by athlete × exercise for rolling calculations
    grouped = df.groupby(["athlete_id", "exercise"], observed=True)

    # Rolling means (using min_periods to avoid NaN at start — they'll be NaN
    # for weeks < rolling_window, which we'll handle later)
    df["estimated_1rm_rolling"] = grouped["estimated_1rm"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=rolling_window).mean()
    )
    df["volume_rolling"] = grouped["volume_kg"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=rolling_window).mean()
    )

    # Percentage change: compare current rolling block to the previous block
    # shift(rolling_window) gives us the rolling average from `rolling_window` weeks ago
    df["estimated_1rm_change_3w"] = grouped["estimated_1rm_rolling"].transform(
        lambda x: x.pct_change(periods=rolling_window)
    )
    df["volume_change_3w"] = grouped["volume_rolling"].transform(
        lambda x: x.pct_change(periods=rolling_window)
    )

    return df


def add_plateau_labels(
    df: pd.DataFrame,
    rolling_window: int = 3,
    improvement_threshold: float = 0.005,
) -> pd.DataFrame:
    """Generate plateau labels based on 1RM progression.

    Definition:
        A plateau occurs when the rolling 1RM improvement over `rolling_window`
        weeks is below `improvement_threshold` (e.g., < 0.5%).

    Also creates `target_plateau_next_week` — the shifted prediction target.
    This ensures the model predicts the FUTURE state, not the current one.

    Args:
        df: DataFrame with `estimated_1rm_change_3w` already computed.
        rolling_window: Used for context — must match the rolling window in features.
        improvement_threshold: Plateau if improvement < this fraction.

    Returns:
        DataFrame with `plateau_label` and `target_plateau_next_week` columns.
    """
    df = df.copy()

    # Plateau: 1RM improvement is below threshold
    # NaN values (early weeks without enough data) are labeled as non-plateau
    df["plateau_label"] = (
        df["estimated_1rm_change_3w"].fillna(improvement_threshold + 1) < improvement_threshold
    ).astype(int)

    # Target: will the athlete be in a plateau NEXT week?
    # Shift the plateau label backwards by 1 within each athlete-exercise group.
    # shift(-1) means "take the value from the next row" = next week's label.
    df["target_plateau_next_week"] = (
        df.groupby(["athlete_id", "exercise"], observed=True)["plateau_label"]
        .shift(-1)
    )

    # The last week for each athlete-exercise has no "next week" → drop or fill
    # We'll mark as NaN and drop later (can't predict beyond the data)
    # But let's keep it as NaN for now and handle in sequence generation.

    return df


def engineer_features(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Full feature engineering pipeline.

    Steps:
        1. Load raw data
        2. Add rolling features
        3. Add plateau labels
        4. Print class balance
        5. Save processed data

    Returns:
        Processed DataFrame.
    """
    cfg = load_config(config_path)

    raw_path = cfg["dataset"]["output_path"]
    rolling_window = cfg["plateau"]["rolling_window"]
    threshold = cfg["plateau"]["improvement_threshold"]
    processed_dir = cfg["preprocessing"]["processed_dir"]

    print(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  Raw shape: {df.shape}")

    # Step 1: Rolling features
    print(f"\nAdding rolling features (window={rolling_window})")
    df = add_rolling_features(df, rolling_window=rolling_window)

    # Step 2: Plateau labels
    print(f"Adding plateau labels (threshold={threshold*100:.1f}%)")
    df = add_plateau_labels(df, rolling_window=rolling_window, improvement_threshold=threshold)

    # Step 3: Report class balance
    # Only consider rows where both features and target are available
    valid_mask = df["target_plateau_next_week"].notna() & df["estimated_1rm_change_3w"].notna()
    df_valid = df[valid_mask]

    print(f"\n{'='*50}")
    print("CLASS BALANCE (on valid rows only)")
    print(f"{'='*50}")
    print(f"Total valid rows: {len(df_valid)}")
    plateau_counts = df_valid["plateau_label"].value_counts()
    target_counts = df_valid["target_plateau_next_week"].value_counts()
    print(f"\nplateau_label distribution:")
    for label, count in plateau_counts.items():
        print(f"  {int(label)}: {count:>6d} ({count/len(df_valid)*100:.1f}%)")
    print(f"\ntarget_plateau_next_week distribution:")
    for label, count in target_counts.items():
        print(f"  {int(label)}: {count:>6d} ({count/len(df_valid)*100:.1f}%)")

    # Step 4: Save
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "workout_features.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engineer_features()
