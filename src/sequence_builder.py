"""
Sequence Builder: Sliding Window for LSTM Input
=================================================
Converts tabular data into temporal sequences for the LSTM.

Each sequence is a window of `sequence_length` consecutive weeks for ONE
athlete on ONE exercise. The target is the plateau label for the week
immediately AFTER the sequence.

Sequences NEVER cross athlete or exercise boundaries.

Example (sequence_length=6):
    Weeks 7–12 features → predict plateau at Week 13
    Weeks 8–13 features → predict plateau at Week 14

Output shapes:
    X: (num_samples, sequence_length, num_features)  e.g. (N, 6, 8)
    y: (num_samples,)  binary 0/1

Usage:
    from src.sequence_builder import build_sequences
    X_train, y_train = build_sequences(train_df, feature_cols, target_col, seq_len=6)
"""

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from the DataFrame.

    For each athlete-exercise group, slides a window of `sequence_length`
    rows and takes the target from the LAST row in the window.

    Why the target comes from the last row:
        Each row already has `target_plateau_next_week` which is the shifted
        label (next week's plateau status). So the target at the last position
        of the window naturally predicts the FUTURE week after the window ends.

    Args:
        df: DataFrame with scaled features and target column.
            Must be sorted by (athlete_id, exercise, week).
        feature_columns: List of column names to use as input features.
        target_column: Name of the target column.
        sequence_length: Number of time steps in each sequence.

    Returns:
        X: np.ndarray of shape (N, sequence_length, num_features)
        y: np.ndarray of shape (N,)
    """
    df = df.sort_values(["athlete_id", "exercise", "week"]).reset_index(drop=True)

    X_list = []
    y_list = []

    # Group by athlete × exercise — sequences never cross boundaries
    for (athlete_id, exercise), group in df.groupby(["athlete_id", "exercise"], observed=True):
        features = group[feature_columns].values  # shape: (num_weeks, num_features)
        targets = group[target_column].values      # shape: (num_weeks,)

        if len(group) < sequence_length:
            continue  # not enough weeks for even one sequence

        for i in range(len(group) - sequence_length + 1):
            seq_features = features[i : i + sequence_length]
            seq_target = targets[i + sequence_length - 1]  # target at end of window

            # Skip if target is NaN (shouldn't happen after preprocessing, but be safe)
            if np.isnan(seq_target):
                continue

            X_list.append(seq_features)
            y_list.append(seq_target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Built {len(X)} sequences of length {sequence_length}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Class 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  Class 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    return X, y


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from preprocessing import prepare_splits

    cfg = load_config()
    feature_cols = cfg["features"]["input_features"]
    target_col = cfg["features"]["target"]
    seq_len = cfg["sequence"]["length"]

    train_df, val_df, test_df, scaler = prepare_splits()

    print(f"\n{'='*50}")
    print("BUILDING SEQUENCES")
    print(f"{'='*50}")

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n--- {name} ---")
        X, y = build_sequences(split_df, feature_cols, target_col, seq_len)
