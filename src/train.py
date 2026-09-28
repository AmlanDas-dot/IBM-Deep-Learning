"""
Training Script: Baselines + LSTM
===================================
End-to-end training pipeline:
    1. Load & preprocess data
    2. Build sequences
    3. Train majority-class & logistic regression baselines
    4. Train LSTM with callbacks
    5. Save models and training history

Usage:
    python src/train.py
"""

import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import prepare_splits, load_config
from sequence_builder import build_sequences
from models import build_lstm_from_config


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights inversely proportional to class frequency.

    This helps the model pay more attention to the minority class (plateau)
    without artificially duplicating temporal sequences (which could introduce
    artifacts in time-series data).

    Args:
        y: Binary target array.

    Returns:
        Dict mapping class index to weight, e.g. {0: 0.66, 1: 2.15}.
    """
    n_samples = len(y)
    n_classes = 2
    class_counts = np.bincount(y.astype(int))
    weights = {}
    for i in range(n_classes):
        weights[i] = n_samples / (n_classes * class_counts[i])
    print(f"Class weights: {weights}")
    return weights


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = "models",
) -> dict:
    """Train baseline models and return their predictions.

    Baselines:
        1. Majority class: always predicts the most common class.
        2. Logistic Regression: uses the LAST time step's features (flattened).

    Args:
        X_train: Training sequences (N, seq_len, features).
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        save_dir: Directory to save baseline models.

    Returns:
        Dict with baseline results.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # --- 1. Majority class baseline ---
    majority_class = int(np.bincount(y_train.astype(int)).argmax())
    y_pred_majority = np.full_like(y_val, majority_class)
    maj_acc = accuracy_score(y_val, y_pred_majority)
    maj_f1 = f1_score(y_val, y_pred_majority, zero_division=0)

    print(f"\n{'='*50}")
    print("BASELINE 1: Majority Class")
    print(f"{'='*50}")
    print(f"  Always predicts: {majority_class}")
    print(f"  Val Accuracy: {maj_acc:.4f}")
    print(f"  Val F1:       {maj_f1:.4f}")

    results["majority_class"] = {
        "predicted_class": majority_class,
        "val_accuracy": float(maj_acc),
        "val_f1": float(maj_f1),
    }

    # --- 2. Logistic Regression baseline ---
    # Use the last time step's features as input (most recent week)
    X_train_last = X_train[:, -1, :]  # shape: (N, num_features)
    X_val_last = X_val[:, -1, :]

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr_model.fit(X_train_last, y_train)

    y_pred_lr = lr_model.predict(X_val_last)
    y_prob_lr = lr_model.predict_proba(X_val_last)[:, 1]

    lr_acc = accuracy_score(y_val, y_pred_lr)
    lr_f1 = f1_score(y_val, y_pred_lr)
    lr_auc = roc_auc_score(y_val, y_prob_lr)

    print(f"\n{'='*50}")
    print("BASELINE 2: Logistic Regression (last time step)")
    print(f"{'='*50}")
    print(f"  Val Accuracy: {lr_acc:.4f}")
    print(f"  Val F1:       {lr_f1:.4f}")
    print(f"  Val ROC-AUC:  {lr_auc:.4f}")

    results["logistic_regression"] = {
        "val_accuracy": float(lr_acc),
        "val_f1": float(lr_f1),
        "val_roc_auc": float(lr_auc),
    }

    # Save LR model
    lr_path = os.path.join(save_dir, "logistic_regression.pkl")
    with open(lr_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"  Saved to {lr_path}")

    return results


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config_path: str = "config/config.yaml",
    save_dir: str = "models",
) -> tuple:
    """Train the LSTM model with callbacks.

    Callbacks:
        - EarlyStopping: stops if val_loss doesn't improve for N epochs.
        - ReduceLROnPlateau: reduces learning rate when val_loss plateaus.
        - ModelCheckpoint: saves the best model based on val_loss.

    Args:
        X_train: Training sequences (N, seq_len, features).
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        config_path: Path to config file.
        save_dir: Directory to save the model.

    Returns:
        (model, history) — trained model and training history.
    """
    import tensorflow as tf

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    seed = cfg["random_seed"]

    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_from_config(input_shape, config_path)

    print(f"\n{'='*50}")
    print("LSTM MODEL SUMMARY")
    print(f"{'='*50}")
    model.summary()

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Callbacks
    checkpoint_path = os.path.join(save_dir, "best_lstm.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=model_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=model_cfg["reduce_lr_factor"],
            patience=model_cfg["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print(f"\nTraining LSTM for up to {model_cfg['max_epochs']} epochs...")
    print(f"  Batch size: {model_cfg['batch_size']}")
    print(f"  Early stopping patience: {model_cfg['early_stopping_patience']}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=model_cfg["max_epochs"],
        batch_size=model_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    print(f"Saved best model to {checkpoint_path}")

    return model, history


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    feature_cols = cfg["features"]["input_features"]
    target_col = cfg["features"]["target"]
    seq_len = cfg["sequence"]["length"]

    # Step 1: Prepare splits
    print("STEP 1: Preparing data splits")
    print("=" * 50)
    train_df, val_df, test_df, scaler = prepare_splits()

    # Step 2: Build sequences
    print(f"\nSTEP 2: Building sequences (length={seq_len})")
    print("=" * 50)

    print("\n--- Train ---")
    X_train, y_train = build_sequences(train_df, feature_cols, target_col, seq_len)
    print("\n--- Validation ---")
    X_val, y_val = build_sequences(val_df, feature_cols, target_col, seq_len)
    print("\n--- Test ---")
    X_test, y_test = build_sequences(test_df, feature_cols, target_col, seq_len)

    # Save test data for later evaluation
    os.makedirs(cfg["paths"]["processed_data"], exist_ok=True)
    np.save(os.path.join(cfg["paths"]["processed_data"], "X_test.npy"), X_test)
    np.save(os.path.join(cfg["paths"]["processed_data"], "y_test.npy"), y_test)
    np.save(os.path.join(cfg["paths"]["processed_data"], "X_val.npy"), X_val)
    np.save(os.path.join(cfg["paths"]["processed_data"], "y_val.npy"), y_val)
    print("\nSaved test and validation arrays to data/processed/")

    # Step 3: Train baselines
    print(f"\nSTEP 3: Training baselines")
    print("=" * 50)
    baseline_results = train_baselines(X_train, y_train, X_val, y_val)

    # Step 4: Train LSTM
    print(f"\nSTEP 4: Training LSTM")
    print("=" * 50)
    model, history = train_lstm(X_train, y_train, X_val, y_val)

    # Quick validation evaluation
    y_pred_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_pred_prob)

    print(f"\n{'='*50}")
    print("LSTM Validation Results")
    print(f"{'='*50}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  ROC-AUC:  {val_auc:.4f}")

    # Save all results summary
    all_results = {
        "baselines": baseline_results,
        "lstm": {
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "val_roc_auc": float(val_auc),
            "epochs_trained": len(history.history["loss"]),
        },
    }

    results_path = os.path.join(cfg["paths"]["metrics"], "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results summary to {results_path}")


if __name__ == "__main__":
    main()
