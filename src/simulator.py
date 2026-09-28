"""
Workout Simulator — Auto-Progression Engine
=============================================
Generates physiologically plausible workout progressions for the interactive UI.

Uses the SAME feature-engineering logic as the training pipeline:
    - Epley 1RM formula (from data_generation.py)
    - Rolling 3-week averages and percentage changes (from feature_engineering.py)
    - Plateau label definition (from feature_engineering.py)

The auto-progression algorithm examines recent history (last 3–6 workouts)
and applies constrained stochastic updates that produce realistic
longitudinal athlete records.

Usage:
    from src.simulator import WorkoutSimulator
    sim = WorkoutSimulator(exercise="bench_press", bodyweight=72.0)
    sim.add_workout(weight_kg=60, sets=3, reps=8, rpe=7.0)
    sim.auto_progress()  # generates the next plausible workout
"""

import numpy as np
import pandas as pd
import yaml
from typing import Optional

from data_generation import epley_1rm


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class WorkoutSimulator:
    """Manages a simulated athlete's workout history and auto-progression.

    Attributes:
        exercise: Current exercise name.
        bodyweight: Athlete bodyweight in kg.
        history: List of dicts, each representing one workout/day.
        rng: Numpy random generator for reproducibility.
    """

    def __init__(
        self,
        exercise: str = "bench_press",
        bodyweight: float = 72.0,
        seed: Optional[int] = None,
        config_path: str = "config/config.yaml",
    ):
        self.cfg = load_config(config_path)
        self.sim_cfg = self.cfg["simulator"]
        self.plateau_cfg = self.cfg["plateau"]
        self.exercise = exercise
        self.bodyweight = bodyweight
        self.history: list[dict] = []
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_workout(
        self,
        weight_kg: float,
        sets: int,
        reps: int,
        rpe: float,
    ) -> dict:
        """Add a manual workout entry.

        Derived fields (volume, estimated_1rm) are computed automatically.
        """
        entry = self._make_entry(
            weight_kg=weight_kg,
            sets=sets,
            reps=reps,
            rpe=rpe,
            is_rest_day=False,
        )
        self.history.append(entry)
        return entry

    def add_rest_day(self) -> dict:
        """Add a rest day.

        Keeps the athlete's reference weight but zeroes out sets/reps/RPE/volume.
        The estimated_1rm carries forward from the last workout to avoid
        visual discontinuity in the graph.
        """
        last_weight = self._last_working_weight()
        entry = self._make_entry(
            weight_kg=last_weight,
            sets=0,
            reps=0,
            rpe=0.0,
            is_rest_day=True,
            current_bw=self.bodyweight
        )
        self.history.append(entry)
        return entry

    def validate_progression(self, history: list[dict], candidate: dict) -> bool:
        """Sanity check to prevent unrealistic progression."""
        if not history:
            return True
            
        sc = self.sim_cfg
        prog = sc.get("progression", {})
        
        if candidate["is_rest_day"]:
            return True
            
        # 1. Bounds check
        if candidate["weight_kg"] < sc["min_weight_kg"] or candidate["weight_kg"] > sc["max_weight_kg"]:
            return False
        if candidate["sets"] < sc["min_sets"] or candidate["sets"] > sc["max_sets"]:
            return False
        if candidate["reps"] < sc["min_reps"] or candidate["reps"] > sc["max_reps"]:
            return False
        if candidate["rpe"] < sc["rpe_min"] or candidate["rpe"] > sc["rpe_max"]:
            return False
            
        recent = [w for w in history if not w["is_rest_day"]]
        if not recent:
            return True
            
        last = recent[-1]
        
        # 2. Maximum single jump check
        max_jump = prog.get("max_single_increment_kg", 2.5)
        if candidate["weight_kg"] - last["weight_kg"] > max_jump + 0.01:
            return False
            
        # 3. Minimum days between weight increases
        min_days = prog.get("min_days_between_weight_increases", 5)
        if candidate["weight_kg"] > last["weight_kg"] + 0.01:
            days_since_increase = 0
            for w in reversed(history):
                days_since_increase += 1
                if not w["is_rest_day"] and w["weight_kg"] < last["weight_kg"] - 0.01:
                    break
            else:
                pass
            
            if days_since_increase <= min_days:
                return False
                
        # 4. Cumulative progression over time limit
        first_weight = history[0]["weight_kg"]
        first_non_rest = [w for w in history if not w["is_rest_day"]]
        if first_non_rest:
            first_weight = first_non_rest[0]["weight_kg"]
            
        elapsed_weeks = len(history) / 7.0
        progression_rate = prog.get("progression_rate_percent_per_week", 1.5) / 100.0
        max_allowed_weight = first_weight * (1.0 + (progression_rate * elapsed_weeks))
        
        ex_defaults = sc.get("exercise_defaults", {}).get(self.exercise, {})
        increment = ex_defaults.get("increment", 2.5)
        
        # Allow the progression budget to accumulate naturally without arbitrary buffers.
        if candidate["weight_kg"] > max_allowed_weight + 0.01:
            return False
            
        # 5. Realistic bodyweight change
        if abs(candidate["bodyweight"] - last["bodyweight"]) > 0.5:
            return False
            
        return True

    def _generate_candidate(self) -> dict:
        sc = self.sim_cfg
        prog = sc.get("progression", {})
        recent = [w for w in self.history if not w["is_rest_day"]]
        ex_defaults = sc.get("exercise_defaults", {}).get(self.exercise, {})
        increment = ex_defaults.get("increment", 2.5)
        
        # Generate rest day logic
        if self.history and self.rng.random() < sc["rest_day_probability"]:
            if not self.history[-1]["is_rest_day"]:
                last_weight = self._last_working_weight()
                return self._make_entry(last_weight, 0, 0, 0.0, True, self.bodyweight)
                
        if not recent:
            weight = ex_defaults.get("start_weight", 60)
            sets = ex_defaults.get("typical_sets", 3)
            reps = ex_defaults.get("typical_reps", 8)
            rpe = round(self.rng.uniform(6.0, 7.5) * 2) / 2
            return self._make_entry(weight, sets, reps, rpe, False, self.bodyweight)

        last = recent[-1]
        weight = last["weight_kg"]
        sets = last["sets"]
        reps = last["reps"]
        rpe = last["rpe"]
        
        # Plateau detection
        is_stagnating = False
        if len(recent) >= 4:
            recent_1rms = [w["estimated_1rm"] for w in recent[-4:]]
            if (max(recent_1rms) - min(recent_1rms)) < 2.0:
                is_stagnating = True
                
        # Recovery Lock: If we recently regressed, prevent immediate bouncing
        is_recovering = False
        can_regress = True
        if len(recent) >= 2:
            for i in range(len(recent)-1, 0, -1):
                if recent[i]["weight_kg"] < recent[i-1]["weight_kg"]:
                    # Found the last regression
                    days_since_regression = recent[-1]["day"] - recent[i]["day"]
                    if days_since_regression < 10:
                        is_recovering = True
                    if days_since_regression < 21:
                        can_regress = False
                    break
                
        roll = self.rng.random()
        
        # Rep-based and Plateau-aware progression
        if is_stagnating and roll < prog.get("plateau_probability", 0.15):
            # True plateau behavior
            reps = max(sc["min_reps"], reps - int(self.rng.choice([0, 1])))
            rpe = min(sc["rpe_max"], rpe + 0.5)
        elif roll < prog.get("regression_probability", 0.03) and can_regress:
            # Small regression (only if we haven't regressed recently)
            weight -= increment
            reps = ex_defaults.get("typical_reps", 8)
            rpe = max(sc["rpe_min"], rpe - 0.5)
        elif reps >= ex_defaults.get("typical_reps", 8) and rpe <= 9.0 and roll < 0.6 and not is_recovering:
            # Earned weight increase
            weight += increment
            reps = max(sc["min_reps"], reps - 2)
            rpe = min(sc["rpe_max"], rpe + 0.5)
        elif reps < ex_defaults.get("typical_reps", 8) and roll < 0.8:
            # Build reps at current weight
            reps += 1
            rpe = min(sc["rpe_max"], rpe + 0.5)
        else:
            # Occasional small random variation
            if roll < 0.15 and not is_recovering:
                weight += increment
                reps = max(sc["min_reps"], reps - 2)
            elif roll < 0.3:
                reps += 1
                
        # Ensure weight is aligned to valid increments
        weight = round(weight / increment) * increment
        
        rpe_drift = self.rng.normal(0, 0.3)
        if is_stagnating:
            rpe_drift += 0.2
        rpe += rpe_drift
        rpe = round(np.clip(rpe, sc["rpe_min"], sc["rpe_max"]) * 2) / 2
        
        sets += int(self.rng.choice([-1, 0, 0, 0, 1]))
        sets = int(np.clip(sets, sc["min_sets"], sc["max_sets"]))
        reps = int(np.clip(reps, sc["min_reps"], sc["max_reps"]))
        
        bw_drift = self.rng.normal(0, sc["bodyweight_drift_std"])
        cand_bw = round(max(40.0, self.bodyweight + bw_drift), 1)
        
        return self._make_entry(weight, sets, reps, rpe, False, cand_bw)

    def auto_progress(self) -> dict:
        """Generate the next workout using constrained stochastic progression."""
        max_attempts = 50
        for _ in range(max_attempts):
            candidate = self._generate_candidate()
            if self.validate_progression(self.history, candidate):
                self.bodyweight = candidate["bodyweight"]
                self.history.append(candidate)
                return candidate
                
        # If all attempts fail, default to maintain
        if self.history and not self.history[-1]["is_rest_day"] and self.rng.random() < 0.5:
            return self.add_rest_day()
        else:
            last = self._last_working_weight()
            recent = [w for w in self.history if not w["is_rest_day"]]
            if recent:
                sets, reps, rpe = recent[-1]["sets"], recent[-1]["reps"], recent[-1]["rpe"]
            else:
                sets, reps, rpe = 3, 8, 7.0
            return self.add_workout(last, sets, reps, rpe)

    def clear(self):
        """Reset the workout history."""
        self.history.clear()

    def load_example(self):
        """Load a pre-built example that demonstrates plateau onset."""
        self.clear()
        # Crafted progression: steady gains → stagnation → plateau
        example = [
            (60.0, 3, 8, 6.5),
            (62.5, 3, 8, 6.5),
            (62.5, 3, 9, 7.0),
            (65.0, 3, 8, 7.0),
            (65.0, 4, 8, 7.0),
            (65.0, 3, 8, 7.5),
            None,  # rest day
            (65.0, 3, 8, 7.0),
            (67.5, 3, 7, 7.5),
            (67.5, 3, 8, 7.5),
            (67.5, 4, 7, 8.0),
            (70.0, 3, 6, 8.0),
            (70.0, 3, 7, 8.0),
            (70.0, 3, 6, 8.5),
            None,  # rest day
            (70.0, 3, 6, 8.5),
            (70.0, 3, 6, 8.5),
            (70.0, 3, 5, 9.0),
            (70.0, 3, 6, 9.0),
            (70.0, 3, 5, 9.0),
            (70.0, 3, 5, 9.5),
        ]
        for item in example:
            if item is None:
                self.add_rest_day()
            else:
                w, s, r, rp = item
                self.add_workout(weight_kg=w, sets=s, reps=r, rpe=rp)

    # ------------------------------------------------------------------
    # Feature engineering (matches training pipeline exactly)
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """Convert history to a DataFrame with all engineered features.

        Uses the SAME calculations as src/feature_engineering.py:
            - estimated_1rm via Epley
            - rolling 3-week means
            - pct_change(periods=3) for 1RM and volume changes
            - plateau_label based on improvement_threshold
        """
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history)
        rw = self.plateau_cfg["rolling_window"]
        threshold = self.plateau_cfg["improvement_threshold"]

        # For rolling calculations, only use non-rest-day rows, then reindex
        # This matches training where rest days weren't in the dataset.
        # But the simulator keeps rest days in-line, so we compute features
        # on non-rest rows and merge back.

        non_rest = df[~df["is_rest_day"]].copy().reset_index(drop=True)

        if len(non_rest) == 0:
            return df

        # Rolling means — same as feature_engineering.add_rolling_features
        non_rest["estimated_1rm_rolling"] = (
            non_rest["estimated_1rm"]
            .rolling(window=rw, min_periods=rw)
            .mean()
        )
        non_rest["volume_rolling"] = (
            non_rest["volume_kg"]
            .rolling(window=rw, min_periods=rw)
            .mean()
        )

        # Percentage change — same as feature_engineering.add_rolling_features
        non_rest["estimated_1rm_change_3w"] = (
            non_rest["estimated_1rm_rolling"]
            .pct_change(periods=rw)
        )
        non_rest["volume_change_3w"] = (
            non_rest["volume_rolling"]
            .pct_change(periods=rw)
        )

        # Plateau label — same as feature_engineering.add_plateau_labels
        non_rest["plateau_label"] = (
            non_rest["estimated_1rm_change_3w"]
            .fillna(threshold + 1) < threshold
        ).astype(int)

        # Merge computed features back into full history (rest days get NaN)
        feature_cols = [
            "estimated_1rm_rolling", "volume_rolling",
            "estimated_1rm_change_3w", "volume_change_3w", "plateau_label",
        ]
        # Map by original index
        for col in feature_cols:
            df[col] = np.nan
        non_rest_idx = df.index[~df["is_rest_day"]]
        for col in feature_cols:
            df.loc[non_rest_idx, col] = non_rest[col].values

        return df

    def get_lstm_features(self) -> Optional[np.ndarray]:
        """Extract the most recent 6-week feature sequence for LSTM input.

        Returns:
            numpy array of shape (1, 6, 8) or None if insufficient data.
            Uses the SAME feature order as config.yaml:
                sets, reps, weight_kg, rpe, volume_kg, estimated_1rm,
                estimated_1rm_change_3w, volume_change_3w
        """
        feature_cols = self.cfg["features"]["input_features"]
        seq_len = self.cfg["sequence"]["length"]

        df = self.get_dataframe()
        if df.empty:
            return None

        # Only non-rest rows with all features available
        non_rest = df[~df["is_rest_day"]].dropna(subset=feature_cols)

        if len(non_rest) < seq_len:
            return None

        # Take the last `seq_len` rows
        window = non_rest.iloc[-seq_len:]
        features = window[feature_cols].values.astype(np.float32)

        return features.reshape(1, seq_len, len(feature_cols))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        weight_kg: float,
        sets: int,
        reps: int,
        rpe: float,
        is_rest_day: bool,
        current_bw: float = None,
    ) -> dict:
        """Create a single history entry with derived fields."""
        day = len(self.history) + 1
        week = (day - 1) // 1 + 1  # each entry = 1 week in the model's terms

        if is_rest_day:
            volume = 0.0
            # Carry forward estimated 1RM from last non-rest workout
            est_1rm = self._last_estimated_1rm()
        else:
            volume = round(sets * reps * weight_kg, 1)
            est_1rm = round(epley_1rm(weight_kg, reps), 2)
            
        bw = current_bw if current_bw is not None else self.bodyweight

        return {
            "day": day,
            "week": week,
            "exercise": self.exercise,
            "bodyweight": round(bw, 1),
            "weight_kg": round(weight_kg, 1),
            "sets": int(sets),
            "reps": int(reps),
            "rpe": float(rpe),
            "volume_kg": volume,
            "estimated_1rm": est_1rm,
            "is_rest_day": is_rest_day,
        }

    def _last_working_weight(self) -> float:
        """Return the weight from the most recent non-rest workout."""
        for entry in reversed(self.history):
            if not entry["is_rest_day"]:
                return entry["weight_kg"]
        # Fallback to exercise default
        defaults = self.sim_cfg["exercise_defaults"].get(self.exercise, {})
        return defaults.get("start_weight", 60)

    def _last_estimated_1rm(self) -> float:
        """Return estimated_1rm from the most recent non-rest workout."""
        for entry in reversed(self.history):
            if not entry["is_rest_day"]:
                return entry["estimated_1rm"]
        return 0.0
