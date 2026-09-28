"""
Interactive Training Simulator — Streamlit Dashboard
======================================================
A polished demonstration UI for the RNN-Based Workout Progression Predictor.

Features:
    - Manual workout entry with live graph updates
    - Rest day support (zero volume without destroying reference weight)
    - Constrained stochastic auto-progression
    - LSTM plateau prediction using the trained model
    - Interactive Plotly charts with hover/zoom
    - Pre-built example athlete for demos

Launch:
    streamlit run src/app.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Path setup — ensure src/ imports work regardless of cwd
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from simulator import WorkoutSimulator
from data_generation import epley_1rm

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workout Plateau Predictor",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container { padding-top: 1.5rem; }
    /* Metric card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #475569;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #f1f5f9; font-weight: 600; }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #cbd5e1; }
    /* Subheader accent */
    .stSubheader { border-bottom: 2px solid #3b82f6; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_config_cached():
    config_path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_lstm_model():
    """Load the trained LSTM model. Returns None on failure."""
    try:
        cfg = load_config_cached()
        model_path = os.path.join(_PROJECT_ROOT, cfg["paths"]["models"], "best_lstm.keras")
        if not os.path.exists(model_path):
            return None
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler. Returns None on failure."""
    try:
        cfg = load_config_cached()
        scaler_path = os.path.join(
            _PROJECT_ROOT, cfg["preprocessing"]["processed_dir"], "scaler.pkl"
        )
        if not os.path.exists(scaler_path):
            return None
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------
cfg = load_config_cached()

if "simulator" not in st.session_state:
    st.session_state.simulator = WorkoutSimulator(
        exercise="bench_press",
        bodyweight=72.0,
        config_path=os.path.join(_PROJECT_ROOT, "config", "config.yaml"),
    )

sim: WorkoutSimulator = st.session_state.simulator


# ---------------------------------------------------------------------------
# Sidebar — Athlete Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Athlete Configuration")

    new_bw = st.number_input(
        "Bodyweight (kg)", min_value=40.0, max_value=200.0,
        value=sim.bodyweight, step=0.5, format="%.1f",
    )
    if new_bw != sim.bodyweight:
        sim.bodyweight = new_bw

    exercises = list(cfg["simulator"]["exercise_defaults"].keys())
    exercise_labels = [e.replace("_", " ").title() for e in exercises]
    ex_idx = exercises.index(sim.exercise) if sim.exercise in exercises else 0
    selected_label = st.selectbox("Exercise", exercise_labels, index=ex_idx)
    selected_exercise = exercises[exercise_labels.index(selected_label)]
    if selected_exercise != sim.exercise:
        sim.exercise = selected_exercise

    st.markdown("---")
    st.markdown("## Quick Actions")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Load Example", use_container_width=True):
            sim.exercise = "bench_press"
            sim.bodyweight = 72.0
            sim.load_example()
            st.rerun()
    with col_b:
        if st.button("Clear History", use_container_width=True, type="secondary"):
            sim.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("## Auto-Progression")
    auto_steps = st.select_slider(
        "Steps to generate", options=[1, 3, 5, 10, 20], value=1,
    )
    if st.button(f"Auto-Progress ({auto_steps} step{'s' if auto_steps > 1 else ''})",
                 use_container_width=True, type="primary"):
        for _ in range(auto_steps):
            sim.auto_progress()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Model: LSTM (64→32)<br>"
        "Sequence: 6 weeks × 8 features<br>"
        "Trained on 1,000 synthetic athletes</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content — Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h2 style='margin-bottom:0'>RNN-Based Workout Progression Predictor</h2>"
    "<p style='color:#94a3b8; margin-top:0'>Interactive Training Simulator</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Row 1 — Manual Workout Entry + Current Stats + LSTM Prediction
# ---------------------------------------------------------------------------
col_entry, col_stats, col_lstm = st.columns([2, 1.5, 1.5])

with col_entry:
    st.markdown("#### Add Workout")

    # Determine defaults from last workout
    defaults = cfg["simulator"]["exercise_defaults"].get(sim.exercise, {})
    last_non_rest = [w for w in sim.history if not w["is_rest_day"]]

    if last_non_rest:
        d_weight = last_non_rest[-1]["weight_kg"]
        d_sets = last_non_rest[-1]["sets"]
        d_reps = last_non_rest[-1]["reps"]
        d_rpe = last_non_rest[-1]["rpe"]
    else:
        d_weight = float(defaults.get("start_weight", 60))
        d_sets = int(defaults.get("typical_sets", 3))
        d_reps = int(defaults.get("typical_reps", 8))
        d_rpe = 7.0

    sc = cfg["simulator"]
    r1, r2 = st.columns(2)
    with r1:
        inp_weight = st.number_input(
            "Weight (kg)", min_value=sc["min_weight_kg"], max_value=sc["max_weight_kg"],
            value=d_weight, step=sc["weight_increment_kg"], format="%.1f",
        )
        inp_sets = st.number_input(
            "Sets", min_value=sc["min_sets"], max_value=sc["max_sets"],
            value=d_sets, step=1,
        )
    with r2:
        inp_reps = st.number_input(
            "Reps", min_value=sc["min_reps"], max_value=sc["max_reps"],
            value=d_reps, step=1,
        )
        inp_rpe = st.number_input(
            "RPE", min_value=sc["rpe_min"], max_value=sc["rpe_max"],
            value=d_rpe, step=0.5, format="%.1f",
        )

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("Add Workout", use_container_width=True, type="primary"):
            sim.add_workout(inp_weight, inp_sets, inp_reps, inp_rpe)
            st.rerun()
    with btn2:
        if st.button("Rest Day", use_container_width=True):
            sim.add_rest_day()
            st.rerun()

with col_stats:
    st.markdown("#### Current Workout Preview")
    preview_vol = round(inp_sets * inp_reps * inp_weight, 1)
    preview_1rm = round(epley_1rm(inp_weight, inp_reps), 1)

    m1, m2 = st.columns(2)
    m1.metric("Volume", f"{preview_vol:,.0f} kg")
    m2.metric("Est. 1RM", f"{preview_1rm:.1f} kg")
    m3, m4 = st.columns(2)
    m3.metric("Weight", f"{inp_weight:.1f} kg")
    m4.metric("RPE", f"{inp_rpe}")

    total_workouts = len([w for w in sim.history if not w["is_rest_day"]])
    total_rest = len([w for w in sim.history if w["is_rest_day"]])
    st.caption(f"{total_workouts} workouts + {total_rest} rest days logged")

with col_lstm:
    st.markdown("#### LSTM Prediction")

    lstm_model = load_lstm_model()
    scaler = load_scaler()

    if lstm_model is None or scaler is None:
        st.warning("Model or scaler not loaded.")
    elif len([w for w in sim.history if not w["is_rest_day"]]) < 6:
        non_rest_count = len([w for w in sim.history if not w["is_rest_day"]])
        remaining = 6 - non_rest_count
        st.info(f"Need **{remaining}** more workout{'s' if remaining > 1 else ''} for prediction (minimum 6 non-rest weeks).")
    else:
        # Get raw features from simulator
        raw_features = sim.get_lstm_features()
        if raw_features is None:
            st.info("Insufficient feature data (rolling windows need ≥6 valid weeks with computed changes).")
        else:
            # Scale using the SAME scaler fitted on training data
            original_shape = raw_features.shape
            flat = raw_features.reshape(-1, original_shape[-1])
            scaled = scaler.transform(flat)
            X_input = scaled.reshape(original_shape)

            prob = float(lstm_model.predict(X_input, verbose=0)[0][0])
            pct = prob * 100

            if prob < 0.3:
                status_icon = "🟢"
                status_text = "Progressing"
                status_color = "#22c55e"
            elif prob < 0.5:
                status_icon = "🟡"
                status_text = "Slowing"
                status_color = "#eab308"
            elif prob < 0.7:
                status_icon = "🟠"
                status_text = "Plateau Risk"
                status_color = "#f97316"
            else:
                status_icon = "🔴"
                status_text = "High Plateau Risk"
                status_color = "#ef4444"

            st.metric("Plateau Probability", f"{pct:.1f}%")
            st.markdown(
                f"<div style='text-align:center; font-size:1.4rem; padding:8px; "
                f"border-radius:8px; border:2px solid {status_color}; "
                f"background:{status_color}20'>"
                f"{status_icon} {status_text}</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2 — Live Progression Graph
# ---------------------------------------------------------------------------
if sim.history:
    st.markdown("#### Live Workout Progression")
    
    # --- PROGRESSION SUMMARY ---
    df = sim.get_dataframe()
    non_rest = df[~df["is_rest_day"]]
    
    if not non_rest.empty:
        start_w = non_rest.iloc[0]["weight_kg"]
        curr_w = non_rest.iloc[-1]["weight_kg"]
        inc = curr_w - start_w
        elapsed = len(df)
        weeks = max(1, elapsed) / 7.0
        avg_weekly = inc / weeks

        prog_cfg = cfg["simulator"].get("progression", {})
        max_weekly = prog_cfg.get("max_weekly_increase_percent", 0.75) / 100.0
        max_allowed_inc_per_week = start_w * max_weekly
        
        if avg_weekly > max_allowed_inc_per_week * 1.5:
            p_status, p_color = "🔴 Unrealistic", "#ef4444"
        elif avg_weekly > max_allowed_inc_per_week:
            p_status, p_color = "🟡 Fast", "#eab308"
        else:
            p_status, p_color = "🟢 Gradual", "#22c55e"
            
        with st.expander("Progression Summary", expanded=True):
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("Starting", f"{start_w:.1f} kg")
            s2.metric("Current", f"{curr_w:.1f} kg")
            s3.metric("Increase", f"+{inc:.1f} kg")
            s4.metric("Elapsed", f"{elapsed} days")
            s5.metric("Avg. Weekly", f"+{avg_weekly:.2f} kg/wk")
            s6.markdown(f"**Status**<br><span style='color:{p_color}'>{p_status}</span>", unsafe_allow_html=True)
            
    rest_days = df[df["is_rest_day"]]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Strength Progression", "Training Volume"),
    )

    # -- Weight line (non-rest only, connected) --
    fig.add_trace(go.Scatter(
        x=non_rest["day"], y=non_rest["weight_kg"],
        mode="lines+markers",
        name="Weight (kg)",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7, color="#3b82f6"),
        hovertemplate="Day %{x}<br>Weight: %{y:.1f} kg<extra></extra>",
    ), row=1, col=1)

    # -- Estimated 1RM line --
    fig.add_trace(go.Scatter(
        x=non_rest["day"], y=non_rest["estimated_1rm"],
        mode="lines+markers",
        name="Est. 1RM (kg)",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(size=5, color="#f59e0b"),
        hovertemplate="Day %{x}<br>Est. 1RM: %{y:.1f} kg<extra></extra>",
    ), row=1, col=1)

    # -- Rolling 1RM if available --
    if "estimated_1rm_rolling" in non_rest.columns:
        rolling_valid = non_rest.dropna(subset=["estimated_1rm_rolling"])
        if not rolling_valid.empty:
            fig.add_trace(go.Scatter(
                x=rolling_valid["day"], y=rolling_valid["estimated_1rm_rolling"],
                mode="lines",
                name="Rolling 3-wk 1RM",
                line=dict(color="#ef4444", width=2.5),
                hovertemplate="Day %{x}<br>Rolling 1RM: %{y:.1f} kg<extra></extra>",
            ), row=1, col=1)

    # -- Rest-day markers --
    if not rest_days.empty:
        # Place rest markers at the last known weight level
        rest_y = rest_days["weight_kg"]
        fig.add_trace(go.Scatter(
            x=rest_days["day"], y=rest_y,
            mode="markers+text",
            name="Rest Day",
            marker=dict(
                size=12, color="#64748b", symbol="x",
                line=dict(width=2, color="#94a3b8"),
            ),
            text=["REST"] * len(rest_days),
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            hovertemplate="Day %{x}<br>REST<extra></extra>",
        ), row=1, col=1)

    # -- Plateau markers (from heuristic, not LSTM) --
    if "plateau_label" in non_rest.columns:
        plateau_rows = non_rest[non_rest["plateau_label"] == 1]
        if not plateau_rows.empty:
            fig.add_trace(go.Scatter(
                x=plateau_rows["day"], y=plateau_rows["weight_kg"],
                mode="markers",
                name="Plateau (heuristic)",
                marker=dict(
                    size=14, color="rgba(239,68,68,0.3)", symbol="circle",
                    line=dict(width=2, color="#ef4444"),
                ),
                hovertemplate="Day %{x}<br>Plateau detected<extra></extra>",
            ), row=1, col=1)

    # -- Volume bars (non-rest only) --
    fig.add_trace(go.Bar(
        x=non_rest["day"], y=non_rest["volume_kg"],
        name="Volume (kg)",
        marker=dict(color="#3b82f6", opacity=0.5),
        hovertemplate="Day %{x}<br>Volume: %{y:,.0f} kg<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=40, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Day", row=2, col=1, gridcolor="#334155")
    fig.update_yaxes(title_text="kg", row=1, col=1, gridcolor="#334155")
    fig.update_yaxes(title_text="Volume (kg)", row=2, col=1, gridcolor="#334155")

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # History Table
    # ------------------------------------------------------------------
    st.markdown("#### Workout History")

    # Build display rows as strings to avoid dtype conflicts
    table_rows = []
    for i, row in df.iterrows():
        if row["is_rest_day"]:
            table_rows.append({
                "Day": int(row["day"]),
                "Weight (kg)": "—",
                "Sets": "—",
                "Reps": "—",
                "RPE": "—",
                "Volume (kg)": "—",
                "Est. 1RM (kg)": "—",
                "Status": "🔵 Rest",
            })
        else:
            if pd.notna(row.get("plateau_label")) and row["plateau_label"] == 1:
                status = "🔴 Plateau"
            elif pd.notna(row.get("estimated_1rm_change_3w")) and row["estimated_1rm_change_3w"] < 0.01:
                status = "🟡 Slowing"
            else:
                status = "🟢 Progressing"
            table_rows.append({
                "Day": int(row["day"]),
                "Weight (kg)": f"{row['weight_kg']:.1f}",
                "Sets": str(int(row["sets"])),
                "Reps": str(int(row["reps"])),
                "RPE": f"{row['rpe']:.1f}",
                "Volume (kg)": f"{row['volume_kg']:,.0f}",
                "Est. 1RM (kg)": f"{row['estimated_1rm']:.1f}",
                "Status": status,
            })

    table_display = pd.DataFrame(table_rows)

    st.dataframe(
        table_display.set_index("Day"),
        use_container_width=True,
        height=min(400, 35 * len(table_display) + 38),
    )


else:
    st.info(
        "No workout history yet. Use **Add Workout**, **Auto-Progress**, or **Load Example** to get started."
    )
