import sys
import os
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from simulator import WorkoutSimulator

def run_simulation(sim_id):
    sim = WorkoutSimulator(exercise="bench_press", bodyweight=72.9)
    sim.add_workout(weight_kg=72.5, sets=3, reps=8, rpe=7.0)
    
    for _ in range(60):
        sim.auto_progress()
        
    df = sim.get_dataframe()
    non_rest = df[~df["is_rest_day"]].copy().reset_index(drop=True)
    rest_days = df[df["is_rest_day"]]
    
    start_weight = non_rest.iloc[0]["weight_kg"]
    final_weight = non_rest.iloc[-1]["weight_kg"]
    total_increase = final_weight - start_weight
    pct_increase = (total_increase / start_weight) * 100
    
    increase_days = []
    regression_days = []
    intervals = []
    oscillations = 0
    min_weight = min(non_rest["weight_kg"])
    
    max_single_increase = 0
    max_7day_increase = 0
    max_14day_increase = 0
    
    non_rest_weights = non_rest["weight_kg"].tolist()
    non_rest_days = non_rest["day"].tolist()
    
    # Track increases, regressions, and oscillations
    for i in range(1, len(non_rest_weights)):
        diff = non_rest_weights[i] - non_rest_weights[i-1]
        if diff > 0:
            increase_days.append(non_rest_days[i])
            if diff > max_single_increase:
                max_single_increase = diff
                
            # Check for oscillation: W(i-2) > W(i-1) and W(i) == W(i-2)
            if i >= 2 and non_rest_weights[i-2] > non_rest_weights[i-1]:
                days_since_reg = non_rest_days[i] - non_rest_days[i-1]
                if days_since_reg < 7:
                    oscillations += 1
                    
        elif diff < 0:
            regression_days.append(non_rest_days[i])
            
    if len(increase_days) > 1:
        intervals = [increase_days[i] - increase_days[i-1] for i in range(1, len(increase_days))]
    elif len(increase_days) == 1:
        intervals = [increase_days[0] - non_rest_days[0]]
        
    # Calculate rolling increases
    for i in range(len(df)):
        c_day = df.iloc[i]["day"]
        c_weight = df.iloc[i]["weight_kg"]
        
        for j in range(i-1, -1, -1):
            p_day = df.iloc[j]["day"]
            p_weight = df.iloc[j]["weight_kg"]
            window = c_day - p_day
            
            if window <= 7:
                if (c_weight - p_weight) > max_7day_increase:
                    max_7day_increase = c_weight - p_weight
            if window <= 14:
                if (c_weight - p_weight) > max_14day_increase:
                    max_14day_increase = c_weight - p_weight
            if window > 14:
                break

    # Automated Assertions
    failures = []
    
    if max_single_increase > 2.5 + 0.01:
        failures.append(f"Single increase > 2.5kg ({max_single_increase}kg)")
        
    if any(interval < 5 for interval in intervals):
        failures.append("Two increases < 5 days apart")
        
    bw_range = max(df["bodyweight"]) - min(df["bodyweight"])
    if bw_range > 15.0:
        failures.append(f"Unrealistic bodyweight change ({bw_range}kg)")
        
    if oscillations > 0:
        failures.append(f"Repeated oscillations detected ({oscillations})")
        
    if len(regression_days) > len(df) * 0.10:
        failures.append(f"Too many regression days ({len(regression_days)} days)")
        
    if len(rest_days) == 0:
        failures.append("No rest days generated (invalid distribution)")
        
    # Chronological weight path
    weight_path = []
    last_w = None
    for w in non_rest["weight_kg"]:
        if w != last_w:
            weight_path.append(f"{w:.1f}")
            last_w = w
            
    # Check that counts match exactly
    num_path_increases = sum(1 for i in range(1, len(weight_path)) if float(weight_path[i]) > float(weight_path[i-1]))
    num_path_regressions = sum(1 for i in range(1, len(weight_path)) if float(weight_path[i]) < float(weight_path[i-1]))
    if num_path_increases != len(increase_days):
        failures.append(f"Path increases ({num_path_increases}) do not match counted increases ({len(increase_days)})")
    if num_path_regressions != len(regression_days):
        failures.append(f"Path regressions ({num_path_regressions}) do not match counted regressions ({len(regression_days)})")

    print(f"=== RUN {sim_id} ===")
    print(f"- Training days: {len(non_rest)}")
    print(f"- Rest days: {len(rest_days)}")
    print(f"- Starting weight: {start_weight:.1f} kg")
    print(f"- Final weight: {final_weight:.1f} kg")
    print(f"- Total increase: {total_increase:+.1f} kg")
    print(f"- Percentage increase: {pct_increase:+.1f}%")
    print(f"- Number of increases: {len(increase_days)}")
    print(f"- Actual increase days: {increase_days}")
    print(f"- Days between increases: {intervals}")
    print(f"- Number of regressions: {len(regression_days)}")
    print(f"- Regression days: {regression_days}")
    print(f"- Min observed weight: {min_weight:.1f} kg")
    print(f"- Max single increase: {max_single_increase:.1f} kg")
    print(f"- Max 7-day change: {max_7day_increase:.1f} kg")
    print(f"- Max 14-day change: {max_14day_increase:.1f} kg")
    print(f"- Oscillation count: {oscillations}")
    print(f"- Chronological weight path: {' -> '.join(weight_path)}")
    
    if failures:
        print("[FAIL] ASSERTION FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("[PASS] ALL ASSERTIONS PASSED")
    print("-" * 50)
    
    return len(failures) == 0

if __name__ == "__main__":
    print("Running 10 stochastic simulations (60 days each)...\n")
    all_passed = True
    for i in range(1, 11):
        if not run_simulation(i):
            all_passed = False
            
    if all_passed:
        print("\n[SUCCESS] ALL 10 SIMULATIONS PASSED")
    else:
        print("\n[WARNING] SOME SIMULATIONS FAILED ASSERTIONS")
