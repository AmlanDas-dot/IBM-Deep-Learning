"""
Synthetic Workout Dataset Generator
====================================
Generates realistic multi-week progressive-overload data for multiple athletes.

Each athlete has:
- A talent / starting-strength level
- A per-exercise progression rate (kg/week)
- Natural week-to-week variation
- Some athletes will plateau (progression rate drops to ~0 after a trigger week)
- RPE fluctuations that correlate with fatigue and proximity to plateau

Usage:
    python src/data_generation.py          # uses config/config.yaml
    python src/data_generation.py --seed 42 --athletes 1000 --weeks 52
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def epley_1rm(weight: float, reps: int) -> float:
    """Estimate 1-rep-max using the Epley formula.

    1RM = weight × (1 + reps / 30)
    """
    if reps <= 0 or weight <= 0:
        return 0.0
    return weight * (1 + reps / 30.0)


# ---------------------------------------------------------------------------
# Exercise-specific defaults (realistic starting weights & rep schemes)
# ---------------------------------------------------------------------------

EXERCISE_DEFAULTS = {
    "bench_press": {
        "start_weight_mean": 60,   # kg
        "start_weight_std": 15,
        "sets_range": (3, 5),
        "reps_range": (5, 10),
        "progression_mean": 0.75,  # kg/week average
        "progression_std": 0.25,
    },
    "squat": {
        "start_weight_mean": 80,
        "start_weight_std": 20,
        "sets_range": (3, 5),
        "reps_range": (5, 8),
        "progression_mean": 1.0,
        "progression_std": 0.30,
    },
    "deadlift": {
        "start_weight_mean": 100,
        "start_weight_std": 25,
        "sets_range": (3, 5),
        "reps_range": (3, 6),
        "progression_mean": 1.0,
        "progression_std": 0.35,
    },
}


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_athlete_exercise(
    athlete_id: int,
    exercise: str,
    num_weeks: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate weekly training data for one athlete on one exercise.

    Progression model:
        - Linear progression with noise for the first N weeks.
        - Some athletes plateau: their progression rate drops dramatically
          after a randomly chosen onset week.
        - RPE starts moderate (~6-7) and drifts upward as the athlete
          approaches their limit, with random fluctuation.
    """
    defaults = EXERCISE_DEFAULTS[exercise]

    # --- Athlete-specific parameters ---
    start_weight = max(
        20.0,
        rng.normal(defaults["start_weight_mean"], defaults["start_weight_std"]),
    )
    base_progression = max(
        0.1,
        rng.normal(defaults["progression_mean"], defaults["progression_std"]),
    )

    # Decide if this athlete will plateau on this exercise
    will_plateau = rng.random() < 0.40  # ~40% chance per exercise
    plateau_onset = rng.integers(20, 42) if will_plateau else num_weeks + 10

    # Typical rep/set scheme (stays mostly stable, small variation per week)
    typical_sets = rng.integers(defaults["sets_range"][0], defaults["sets_range"][1] + 1)
    typical_reps = rng.integers(defaults["reps_range"][0], defaults["reps_range"][1] + 1)

    rows = []
    current_weight = start_weight

    for week in range(1, num_weeks + 1):
        # --- Progression logic ---
        if week < plateau_onset:
            # Normal progressive overload
            weekly_gain = base_progression * rng.uniform(0.5, 1.5)
        elif week == plateau_onset:
            # Transition week — progression drops sharply
            weekly_gain = base_progression * rng.uniform(0.0, 0.2)
        else:
            # Post-plateau: very small random fluctuation around zero
            weekly_gain = rng.normal(0.0, 0.3)

        current_weight = max(20.0, current_weight + weekly_gain)
        # Round weight to nearest 2.5 kg (barbell increment)
        current_weight_rounded = round(current_weight / 2.5) * 2.5

        # --- Sets & reps with small weekly variation ---
        sets = int(np.clip(typical_sets + rng.integers(-1, 2), 2, 6))
        reps = int(np.clip(typical_reps + rng.integers(-1, 2), 1, 12))

        # --- RPE: increases as athlete nears plateau ---
        if week < plateau_onset - 5:
            base_rpe = 6.5
        elif week < plateau_onset:
            base_rpe = 7.0 + (week - (plateau_onset - 5)) * 0.3
        else:
            base_rpe = 8.5
        rpe = float(np.clip(base_rpe + rng.normal(0, 0.5), 5.0, 10.0))
        rpe = round(rpe * 2) / 2  # round to nearest 0.5

        # --- Derived fields ---
        volume_kg = sets * reps * current_weight_rounded
        est_1rm = epley_1rm(current_weight_rounded, reps)

        rows.append({
            "athlete_id": athlete_id,
            "week": week,
            "exercise": exercise,
            "sets": sets,
            "reps": reps,
            "weight_kg": current_weight_rounded,
            "rpe": rpe,
            "volume_kg": round(volume_kg, 1),
            "estimated_1rm": round(est_1rm, 2),
        })

    return pd.DataFrame(rows)


def generate_dataset(
    num_athletes: int = 1000,
    num_weeks: int = 52,
    exercises: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the full synthetic workout dataset.

    Args:
        num_athletes: Number of athletes to simulate.
        num_weeks: Number of weeks per athlete.
        exercises: List of exercise names.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            athlete_id, week, exercise, sets, reps, weight_kg,
            rpe, volume_kg, estimated_1rm
    """
    if exercises is None:
        exercises = list(EXERCISE_DEFAULTS.keys())

    rng = np.random.default_rng(seed)
    all_frames = []

    for athlete_id in range(1, num_athletes + 1):
        for exercise in exercises:
            df_ex = generate_athlete_exercise(athlete_id, exercise, num_weeks, rng)
            all_frames.append(df_ex)

        if athlete_id % 200 == 0:
            print(f"  Generated {athlete_id}/{num_athletes} athletes …")

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset = dataset.sort_values(["athlete_id", "exercise", "week"]).reset_index(drop=True)

    print(f"\nDataset shape: {dataset.shape}")
    print(f"Athletes: {dataset['athlete_id'].nunique()}")
    print(f"Exercises: {dataset['exercise'].nunique()}")
    print(f"Weeks per athlete/exercise: {dataset.groupby(['athlete_id', 'exercise'])['week'].count().unique()}")
    return dataset


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic workout data")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--athletes", type=int, default=None, help="Override num athletes")
    parser.add_argument("--weeks", type=int, default=None, help="Override num weeks")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = args.seed or cfg["random_seed"]
    num_athletes = args.athletes or cfg["dataset"]["num_athletes"]
    num_weeks = args.weeks or cfg["dataset"]["num_weeks"]
    exercises = cfg["dataset"]["exercises"]
    output_path = cfg["dataset"]["output_path"]

    print(f"Generating dataset: {num_athletes} athletes × {num_weeks} weeks × {len(exercises)} exercises")
    print(f"Random seed: {seed}")

    dataset = generate_dataset(
        num_athletes=num_athletes,
        num_weeks=num_weeks,
        exercises=exercises,
        seed=seed,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
