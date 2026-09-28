# RNN-Based Workout Progression Predictor

## Problem Statement

**Implement an LSTM network to forecast plateau phases in strength training by analyzing multi-week progressive overload data.**

When athletes follow a strength-training program, they periodically experience *plateaus* — periods where their strength gains stall despite continued training. Early identification of an approaching plateau allows coaches and athletes to proactively adjust programming (e.g., deload weeks, exercise variation, volume manipulation).

This project builds an LSTM-based binary classifier that takes a 6-week sliding window of training data and predicts whether the athlete will be in a plateau the following week.

---

## Project Objective

Build a complete, reproducible deep-learning pipeline that:
1. Generates realistic synthetic workout data for 1,000 athletes over 52 weeks.
2. Engineers temporal features and defines plateau labels using transparent heuristics.
3. Trains an LSTM model to predict future plateau onset.
4. Compares the LSTM against simple baselines (majority class, logistic regression).
5. Evaluates all models on a held-out test set with comprehensive metrics.

---

## Dataset

### Source
The dataset is **synthetically generated** using `src/data_generation.py`. No real athlete data is used.

### Scale
- **1,000 athletes** × **52 weeks** × **3 exercises** = **156,000 rows**
- Exercises: Bench Press, Squat, Deadlift

### Features (raw)
| Field | Description |
|-------|-------------|
| `athlete_id` | Unique athlete identifier (1–1000) |
| `week` | Training week (1–52) |
| `exercise` | Exercise name |
| `sets` | Number of sets performed |
| `reps` | Repetitions per set |
| `weight_kg` | Working weight in kilograms |
| `rpe` | Rate of Perceived Exertion (5–10 scale) |
| `volume_kg` | Total volume = sets × reps × weight |
| `estimated_1rm` | Epley formula: weight × (1 + reps/30) |

### Data Realism
- Each athlete has a unique starting strength and progression rate.
- ~40% of athlete-exercise combinations experience a plateau (onset week 20–42).
- RPE drifts upward as athletes approach their plateau.
- Weight increments in 2.5 kg steps (standard barbell plates).

> **Important**: This is synthetic data. It captures simplified progressive-overload dynamics but does not reflect the full complexity of real athlete training.

---

## Plateau Definition

A **plateau** is defined as a period where an athlete's strength progression shows insufficient improvement:

1. Calculate a **3-week rolling average** of estimated 1RM.
2. Compare the current 3-week block against the previous 3-week block.
3. If improvement is **< 0.5%**, the week is labeled as a plateau.

The prediction target is **`target_plateau_next_week`** — a shifted version of the plateau label. This ensures the model predicts the *future* state, not the current one.

The threshold (0.5%) and rolling window (3 weeks) are configurable in `config/config.yaml`.

---

## Feature Engineering

### Engineered Features
| Feature | Description |
|---------|-------------|
| `estimated_1rm_rolling` | 3-week rolling mean of estimated 1RM |
| `volume_rolling` | 3-week rolling mean of volume |
| `estimated_1rm_change_3w` | Percentage change in rolling 1RM vs. previous block |
| `volume_change_3w` | Percentage change in rolling volume vs. previous block |
| `plateau_label` | Current plateau status (binary) |
| `target_plateau_next_week` | Next week's plateau status (prediction target) |

### Input Features for the Model
```
sets, reps, weight_kg, rpe, volume_kg,
estimated_1rm, estimated_1rm_change_3w, volume_change_3w
```

Note: `plateau_label` is NOT used as an input feature to avoid leaking the target.

---

## Sequence Generation

The LSTM receives **sliding windows of 6 consecutive weeks** per athlete per exercise:

```
Weeks  7–12 features → predict plateau at Week 13
Weeks  8–13 features → predict plateau at Week 14
...
```

Each sequence has shape **(6, 8)** — 6 time steps, 8 features.

Sequences **never cross athlete or exercise boundaries**.

---

## Train / Validation / Test Split

### Athlete-Level Split
| Split | Athletes | Fraction |
|-------|----------|----------|
| Train | 700 | 70% |
| Validation | 150 | 15% |
| Test | 150 | 15% |

**Why athlete-level?** If we split randomly, the model could see weeks 1–30 of an athlete during training and then be tested on weeks 31–52 of the *same* athlete. This leaks information about individual progression patterns and inflates performance metrics.

All sequences from a given athlete remain in the same split.

---

## LSTM Architecture

```
Input (6, 8)
    ↓
LSTM(64, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dropout(0.2)
    ↓
Dense(16, relu)
    ↓
Dense(1, sigmoid)
```

- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (lr=0.001)
- **Class weights**: Applied to handle ~77/23 class imbalance

### Callbacks
- **EarlyStopping**: patience=10, restores best weights
- **ReduceLROnPlateau**: reduces LR by 0.5× after 5 epochs of no improvement
- **ModelCheckpoint**: saves the best model by validation loss

---

## Training Procedure

1. Generate synthetic data → `data/raw/workout_data.csv`
2. Engineer features → `data/processed/workout_features.csv`
3. Split by athlete → train/val/test DataFrames
4. Fit StandardScaler on training data only → apply to all splits
5. Build 6-week sliding-window sequences
6. Train baselines (majority class, logistic regression)
7. Train LSTM with callbacks and class weights
8. Evaluate all models on the held-out test set

---

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted plateaus, how many were correct?
- **Recall**: Of actual plateaus, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability across all thresholds
- **Confusion Matrix**: Detailed error breakdown

---

## How to Run

### Prerequisites
```bash
# Activate the virtual environment
# Windows:
.\IBMpy\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Full Pipeline
```bash
# 1. Generate synthetic data
python src/data_generation.py

# 2. Engineer features and labels
python src/feature_engineering.py

# 3. Train all models (baselines + LSTM)
python src/train.py

# 4. Evaluate on test set
python src/evaluate.py

# 5. Generate all plots
python src/visualization.py
```

All outputs are saved to `models/`, `results/figures/`, and `results/metrics/`.

---

## Project Structure

```
project_root/
├── config/
│   └── config.yaml              # All configurable parameters
├── data/
│   ├── raw/
│   │   └── workout_data.csv     # Generated synthetic data
│   └── processed/
│       ├── workout_features.csv # Engineered features
│       ├── scaler.pkl           # Fitted StandardScaler
│       ├── X_test.npy           # Test sequences
│       └── y_test.npy           # Test targets
├── models/
│   ├── best_lstm.keras          # Best LSTM model
│   ├── logistic_regression.pkl  # Baseline LR model
│   └── training_history.json    # LSTM training history
├── results/
│   ├── figures/                 # All plots (PNG)
│   └── metrics/                 # Evaluation results (JSON)
├── src/
│   ├── __init__.py
│   ├── data_generation.py       # Synthetic data generator
│   ├── feature_engineering.py   # Feature engineering + labeling
│   ├── preprocessing.py         # Splitting + scaling
│   ├── sequence_builder.py      # Sliding window sequences
│   ├── models.py                # LSTM architecture
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Test-set evaluation
│   └── visualization.py         # Plot generation
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Limitations

1. **Synthetic data** — The dataset is simulated and cannot capture the full complexity of real athlete training (periodization, injury, fatigue accumulation, psychological factors).

2. **Heuristic labels** — Plateau labels are generated using a rule-based threshold (< 0.5% 1RM improvement). This is a simplification; real plateaus have nuanced characteristics.

3. **Limited features** — Real-world training involves sleep quality, nutrition, stress, recovery, program design, and injury history, none of which are modeled.

4. **No clinical validation** — LSTM performance on synthetic data does not prove effectiveness for real sports-science applications.

5. **Single threshold** — The 0.5% plateau threshold is somewhat arbitrary. Different thresholds produce different class distributions and may alter model performance.

6. **Exercise independence** — Each exercise is treated as an independent time series. In reality, exercises influence each other (e.g., squat strength affects deadlift).

---

## Future Improvements

1. Use **real longitudinal athlete data** from training logs or wearable devices.
2. Add **external features**: sleep, nutrition, heart-rate variability, training program metadata.
3. Experiment with **Transformer/attention-based architectures** for sequence modeling.
4. Implement **multi-step forecasting** (predict plateau 2–4 weeks ahead).
5. Add **exercise interaction modeling** (multi-variate time series across exercises).
6. Explore **hyperparameter optimization** with Optuna or similar frameworks.
7. Build a **deployment interface** (web app / API) for real-time predictions.

---

## Reproducibility

- All random seeds are fixed (`random_seed: 42` in config.yaml).
- Scaler is fitted only on training data and saved to disk.
- Model checkpointing saves the best model by validation loss.
- All parameters are centralized in `config/config.yaml`.

---

*This project was developed as an academic deep-learning exercise. It demonstrates the application of LSTM networks to temporal pattern recognition in a sports-science context using synthetic data.*
