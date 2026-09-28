"""
Visualization: Plots for Analysis and Presentation
====================================================
Generates all figures for the project report/presentation.

Plots:
    1. Athlete strength progression over time
    2. Plateau periods highlighted
    3. Training volume over time
    4. Estimated 1RM over time
    5. Training/validation loss & accuracy curves
    6. Confusion matrix heatmap
    7. ROC curve comparison
    8. Class distribution
    9. Actual vs predicted plateau labels

Usage:
    python src/visualization.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_config


# Style configuration
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")


def plot_athlete_progression(
    df: pd.DataFrame,
    athlete_id: int,
    exercise: str,
    save_dir: str,
) -> None:
    """Plot a single athlete's strength progression with plateau periods highlighted.

    Args:
        df: DataFrame with all features and plateau labels.
        athlete_id: Which athlete to plot.
        exercise: Which exercise to plot.
        save_dir: Directory to save the figure.
    """
    athlete_data = df[
        (df["athlete_id"] == athlete_id) & (df["exercise"] == exercise)
    ].sort_values("week")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Athlete {athlete_id} — {exercise.replace('_', ' ').title()} Progression",
                 fontsize=16, fontweight="bold")

    weeks = athlete_data["week"].values

    # --- Plot 1: Estimated 1RM ---
    ax1 = axes[0]
    ax1.plot(weeks, athlete_data["estimated_1rm"], "b-o", markersize=3, label="Estimated 1RM", linewidth=1.5)
    if "estimated_1rm_rolling" in athlete_data.columns:
        ax1.plot(weeks, athlete_data["estimated_1rm_rolling"], "r-", linewidth=2.5,
                 label=f"Rolling 3-week avg", alpha=0.8)

    # Shade plateau periods
    if "plateau_label" in athlete_data.columns:
        plateau_mask = athlete_data["plateau_label"].values == 1
        for i in range(len(weeks)):
            if plateau_mask[i]:
                ax1.axvspan(weeks[i] - 0.5, weeks[i] + 0.5, alpha=0.15, color="red")
        # Add legend entry for plateau shading
        plateau_patch = mpatches.Patch(color="red", alpha=0.15, label="Plateau period")
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(plateau_patch)
        ax1.legend(handles=handles, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    ax1.set_ylabel("Estimated 1RM (kg)")
    ax1.set_title("Strength Progression")

    # --- Plot 2: Training Volume ---
    ax2 = axes[1]
    ax2.bar(weeks, athlete_data["volume_kg"], color="steelblue", alpha=0.7, label="Weekly Volume")
    ax2.set_ylabel("Volume (kg)")
    ax2.set_title("Training Volume")
    ax2.legend(loc="upper left")

    # --- Plot 3: RPE ---
    ax3 = axes[2]
    ax3.plot(weeks, athlete_data["rpe"], "g-s", markersize=3, label="RPE", linewidth=1.5)
    ax3.set_ylabel("RPE")
    ax3.set_xlabel("Week")
    ax3.set_title("Rate of Perceived Exertion")
    ax3.set_ylim(4, 11)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    filename = f"athlete_{athlete_id}_{exercise}_progression.png"
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_training_curves(save_dir: str, models_dir: str = "models") -> None:
    """Plot training/validation loss and accuracy curves.

    Args:
        save_dir: Directory to save figures.
        models_dir: Directory containing training_history.json.
    """
    history_path = os.path.join(models_dir, "training_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Training Curves", fontsize=16, fontweight="bold")

    # Loss curves
    ax1.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


def plot_confusion_matrices(save_dir: str, metrics_dir: str = "results/metrics") -> None:
    """Plot confusion matrices for all models.

    Args:
        save_dir: Directory to save figures.
        metrics_dir: Directory containing test_results.json.
    """
    results_path = os.path.join(metrics_dir, "test_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Only plot models with confusion matrices
    models_to_plot = {k: v for k, v in results.items() if "confusion_matrix" in v}

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(6 * len(models_to_plot), 5))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=16, fontweight="bold")

    if len(models_to_plot) == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, models_to_plot.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Plateau", "Plateau"],
                    yticklabels=["No Plateau", "Plateau"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        display_name = name.replace("_", " ").title()
        ax.set_title(f"{display_name}\n(Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_roc_curves(save_dir: str, metrics_dir: str = "results/metrics") -> None:
    """Plot ROC curves comparing models.

    Args:
        save_dir: Directory to save figures.
        metrics_dir: Directory with roc_*.json files and test_results.json.
    """
    results_path = os.path.join(metrics_dir, "test_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curve Comparison (Test Set)", fontsize=16, fontweight="bold")

    colors = {"lr": "darkorange", "lstm": "blue"}
    names = {"lr": "Logistic Regression", "lstm": "LSTM"}

    for key in ["lr", "lstm"]:
        roc_path = os.path.join(metrics_dir, f"roc_{key}.json")
        if os.path.exists(roc_path):
            with open(roc_path, "r") as f:
                roc_data = json.load(f)
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]

            # Get AUC from results
            result_key = "logistic_regression" if key == "lr" else key
            auc_val = results[result_key]["roc_auc"]

            ax.plot(fpr, tpr, color=colors[key], linewidth=2,
                    label=f"{names[key]} (AUC = {auc_val:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), bbox_inches="tight")
    plt.close()
    print("  Saved roc_curves.png")


def plot_class_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Plot the class distribution of plateau labels.

    Args:
        df: DataFrame with plateau_label and target_plateau_next_week.
        save_dir: Directory to save figures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=16, fontweight="bold")

    # plateau_label
    valid = df["plateau_label"].dropna()
    counts = valid.value_counts().sort_index()
    ax1.bar(["No Plateau (0)", "Plateau (1)"], counts.values,
            color=["steelblue", "coral"])
    ax1.set_title("plateau_label")
    ax1.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 500, f"{v}\n({v/len(valid)*100:.1f}%)", ha="center", fontsize=10)

    # target_plateau_next_week
    valid_target = df["target_plateau_next_week"].dropna()
    counts_target = valid_target.value_counts().sort_index()
    ax2.bar(["No Plateau (0)", "Plateau (1)"], counts_target.values,
            color=["steelblue", "coral"])
    ax2.set_title("target_plateau_next_week")
    ax2.set_ylabel("Count")
    for i, v in enumerate(counts_target.values):
        ax2.text(i, v + 500, f"{v}\n({v/len(valid_target)*100:.1f}%)", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved class_distribution.png")


def plot_data_overview(df: pd.DataFrame, save_dir: str) -> None:
    """Plot overview statistics of the dataset.

    Args:
        df: Raw or processed DataFrame.
        save_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # 1. Weight distribution by exercise
    ax = axes[0, 0]
    for exercise in df["exercise"].unique():
        ex_data = df[df["exercise"] == exercise]
        ax.hist(ex_data["weight_kg"], bins=40, alpha=0.5,
                label=exercise.replace("_", " ").title())
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Frequency")
    ax.set_title("Weight Distribution by Exercise")
    ax.legend()

    # 2. RPE distribution
    ax = axes[0, 1]
    ax.hist(df["rpe"], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("RPE")
    ax.set_ylabel("Frequency")
    ax.set_title("RPE Distribution")

    # 3. Average 1RM by week (all athletes, one exercise)
    ax = axes[1, 0]
    for exercise in df["exercise"].unique():
        ex_data = df[df["exercise"] == exercise]
        weekly_avg = ex_data.groupby("week")["estimated_1rm"].mean()
        ax.plot(weekly_avg.index, weekly_avg.values, linewidth=2,
                label=exercise.replace("_", " ").title())
    ax.set_xlabel("Week")
    ax.set_ylabel("Average Estimated 1RM (kg)")
    ax.set_title("Average Strength Progression (All Athletes)")
    ax.legend()

    # 4. Volume distribution
    ax = axes[1, 1]
    ax.hist(df["volume_kg"], bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Volume (kg)")
    ax.set_ylabel("Frequency")
    ax.set_title("Training Volume Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "data_overview.png"), bbox_inches="tight")
    plt.close()
    print("  Saved data_overview.png")


def plot_model_comparison_bar(save_dir: str, metrics_dir: str = "results/metrics") -> None:
    """Bar chart comparing model metrics side by side.

    Args:
        save_dir: Directory to save figures.
        metrics_dir: Directory containing test_results.json.
    """
    results_path = os.path.join(metrics_dir, "test_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = list(results.keys())
    display_names = [n.replace("_", " ").title() for n in model_names]

    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Model Comparison (Test Set)", fontsize=16, fontweight="bold")

    for i, (name, display) in enumerate(zip(model_names, display_names)):
        values = [results[name].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=display, color=colors[i % len(colors)])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), bbox_inches="tight")
    plt.close()
    print("  Saved model_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(config_path: str = "config/config.yaml"):
    """Generate all visualization plots.

    Args:
        config_path: Path to config file.
    """
    cfg = load_config(config_path)
    save_dir = cfg["paths"]["figures"]
    os.makedirs(save_dir, exist_ok=True)

    # Load processed data for data-related plots
    processed_path = os.path.join(cfg["preprocessing"]["processed_dir"], "workout_features.csv")
    df = pd.read_csv(processed_path)

    print("Generating plots...")

    # 1. Data overview
    print("\n1. Data overview plots:")
    plot_data_overview(df, save_dir)

    # 2. Class distribution
    print("\n2. Class distribution:")
    plot_class_distribution(df, save_dir)

    # 3. Example athlete progressions (pick athletes with and without plateaus)
    print("\n3. Athlete progression plots:")
    # Find an athlete with plateau and one without
    plateau_athletes = df[df["plateau_label"] == 1]["athlete_id"].unique()
    non_plateau_athletes = df[df["plateau_label"] == 0]["athlete_id"].unique()

    # Pick first athlete with plateau on bench press
    for aid in plateau_athletes[:3]:
        athlete_bench = df[(df["athlete_id"] == aid) & (df["exercise"] == "bench_press")]
        if athlete_bench["plateau_label"].sum() > 5:
            plot_athlete_progression(df, aid, "bench_press", save_dir)
            break

    # Pick first athlete without plateau on squat
    for aid in non_plateau_athletes[:20]:
        athlete_squat = df[(df["athlete_id"] == aid) & (df["exercise"] == "squat")]
        if athlete_squat["plateau_label"].sum() == 0:
            plot_athlete_progression(df, aid, "squat", save_dir)
            break

    # 4. Training curves (only if model has been trained)
    history_path = os.path.join(cfg["paths"]["models"], "training_history.json")
    if os.path.exists(history_path):
        print("\n4. Training curves:")
        plot_training_curves(save_dir, cfg["paths"]["models"])
    else:
        print("\n4. Training curves: SKIPPED (model not trained yet)")

    # 5. Confusion matrices and ROC curves (only if evaluation has been run)
    test_results_path = os.path.join(cfg["paths"]["metrics"], "test_results.json")
    if os.path.exists(test_results_path):
        print("\n5. Confusion matrices:")
        plot_confusion_matrices(save_dir, cfg["paths"]["metrics"])
        print("\n6. ROC curves:")
        plot_roc_curves(save_dir, cfg["paths"]["metrics"])
        print("\n7. Model comparison bar chart:")
        plot_model_comparison_bar(save_dir, cfg["paths"]["metrics"])
    else:
        print("\n5-7. Evaluation plots: SKIPPED (evaluation not run yet)")

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    generate_all_plots()
