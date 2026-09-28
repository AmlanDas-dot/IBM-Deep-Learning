"""
Evaluation: Test-Set Metrics and Model Comparison
===================================================
Evaluates the trained LSTM and baselines on the held-out test set.

Produces:
    - Classification report (precision, recall, F1)
    - Confusion matrix
    - ROC curve + AUC
    - Model comparison table
    - False positive / false negative analysis

Usage:
    python src/evaluate.py
"""

import json
import os
import sys
import pickle

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import prepare_splits, load_config
from sequence_builder import build_sequences


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """Compute all evaluation metrics for a model.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities (for ROC-AUC).
        model_name: Name for display purposes.

    Returns:
        Dict with all metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>5d}  FP={cm[0,1]:>5d}")
    print(f"    FN={cm[1,0]:>5d}  TP={cm[1,1]:>5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Plateau", "Plateau"]))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def main():
    cfg = load_config()
    feature_cols = cfg["features"]["input_features"]
    target_col = cfg["features"]["target"]
    seq_len = cfg["sequence"]["length"]

    # Load test data
    processed_dir = cfg["paths"]["processed_data"]
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    print(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")

    results = {}

    # --- 1. Majority Class Baseline ---
    majority_class = 0  # from training, class 0 is majority
    y_pred_maj = np.full_like(y_test, majority_class)
    y_prob_maj = np.full_like(y_test, 0.0, dtype=float)  # probability = 0 for class 1
    results["majority_class"] = evaluate_model(y_test, y_pred_maj, y_prob_maj, "Majority Class Baseline")

    # --- 2. Logistic Regression Baseline ---
    lr_path = os.path.join(cfg["paths"]["models"], "logistic_regression.pkl")
    with open(lr_path, "rb") as f:
        lr_model = pickle.load(f)

    X_test_last = X_test[:, -1, :]  # last time step features
    y_pred_lr = lr_model.predict(X_test_last)
    y_prob_lr = lr_model.predict_proba(X_test_last)[:, 1]
    results["logistic_regression"] = evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")

    # --- 3. LSTM Model ---
    import tensorflow as tf

    model_path = os.path.join(cfg["paths"]["models"], "best_lstm.keras")
    lstm_model = tf.keras.models.load_model(model_path)
    print(f"\nLoaded LSTM from {model_path}")

    y_prob_lstm = lstm_model.predict(X_test, verbose=0).flatten()
    y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
    results["lstm"] = evaluate_model(y_test, y_pred_lstm, y_prob_lstm, "LSTM")

    # --- Comparison Table ---
    print(f"\n{'='*70}")
    print("MODEL COMPARISON (Test Set)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 75)
    for name, metrics in results.items():
        display_name = name.replace("_", " ").title()
        print(f"{display_name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['roc_auc']:>10.4f}")

    # --- False Positive / False Negative Analysis ---
    cm = np.array(results["lstm"]["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*50}")
    print("LSTM Error Analysis")
    print(f"{'='*50}")
    print(f"  True Positives (correctly predicted plateau):     {tp}")
    print(f"  True Negatives (correctly predicted no plateau):  {tn}")
    print(f"  False Positives (predicted plateau, wasn't):      {fp}")
    print(f"  False Negatives (missed plateau):                 {fn}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")

    # Save results
    results_path = os.path.join(cfg["paths"]["metrics"], "test_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test results to {results_path}")

    # Save ROC data for plotting
    for name, y_prob in [("lr", y_prob_lr), ("lstm", y_prob_lstm)]:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        roc_path = os.path.join(cfg["paths"]["metrics"], f"roc_{name}.json")
        with open(roc_path, "w") as f:
            json.dump(roc_data, f)
    print("Saved ROC curve data for plotting.")


if __name__ == "__main__":
    main()
