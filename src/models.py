"""
Model Definitions: Baselines + LSTM
=====================================
Contains:
    - Majority-class baseline
    - Logistic Regression baseline
    - LSTM model for plateau prediction

Usage:
    from src.models import build_lstm_model
    model = build_lstm_model(input_shape=(6, 8))
"""

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_rate: float = 0.2,
    dense_units: int = 16,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build the LSTM model for plateau prediction.

    Architecture:
        Input → LSTM(64, return_sequences=True) → Dropout(0.2)
              → LSTM(32) → Dropout(0.2)
              → Dense(16, relu) → Dense(1, sigmoid)

    Args:
        input_shape: (sequence_length, num_features), e.g. (6, 8).
        lstm_units_1: Units in first LSTM layer.
        lstm_units_2: Units in second LSTM layer.
        dropout_rate: Dropout rate after each LSTM layer.
        dense_units: Units in the hidden Dense layer.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # First LSTM layer — returns sequences for the second LSTM
        layers.LSTM(lstm_units_1, return_sequences=True, name="lstm_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # Second LSTM layer — returns only the final output
        layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),

        # Dense classification head
        layers.Dense(dense_units, activation="relu", name="dense_hidden"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="plateau_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_lstm_from_config(
    input_shape: tuple[int, int],
    config_path: str = "config/config.yaml",
) -> keras.Model:
    """Build the LSTM model using parameters from config.yaml.

    Args:
        input_shape: (sequence_length, num_features).
        config_path: Path to config file.

    Returns:
        Compiled Keras model.
    """
    cfg = load_config(config_path)["model"]

    return build_lstm_model(
        input_shape=input_shape,
        lstm_units_1=cfg["lstm_units_1"],
        lstm_units_2=cfg["lstm_units_2"],
        dropout_rate=cfg["dropout_rate"],
        dense_units=cfg["dense_units"],
        learning_rate=cfg["learning_rate"],
    )
