"""
Feature engineering functions for the Panama electricity load forecasting model.

Extracted verbatim from the training notebook so the Streamlit app builds
features using the EXACT SAME code path as training - not a hand-copied
reimplementation that could silently drift from what the model actually learned.
"""
import numpy as np
import pandas as pd

FORECAST_HORIZON = 24


def add_calendar_features(data):
    """Extract calendar structure from the datetime index."""
    out = data.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["day"] = idx.day
    out["month"] = idx.month
    out["year"] = idx.year
    out["dayofyear"] = idx.dayofyear
    out["weekofyear"] = idx.isocalendar().week.astype(int)
    out["quarter"] = idx.quarter

    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_working_day"] = ((idx.dayofweek < 5) &
                             (out["holiday"] == 0)).astype(int)
    out["time_index"] = np.arange(len(out))
    return out


def add_cyclical_features(data):
    """Encode cyclical variables as sine/cosine pairs."""
    out = data.copy()
    for col, period in [("hour", 24), ("dayofweek", 7),
                        ("month", 12), ("dayofyear", 365.25)]:
        out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


def add_lag_features(data, target_col, lags):
    """Past values of the target."""
    out = data.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(data, target_col, windows, shift=FORECAST_HORIZON):
    """Rolling statistics of the target, shifted by the forecast horizon so no
    window can contain information unavailable at prediction time."""
    out = data.copy()
    shifted = out[target_col].shift(shift)

    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
        out[f"roll_min_{window}"] = shifted.rolling(window).min()
        out[f"roll_max_{window}"] = shifted.rolling(window).max()

    for span in [24, 168]:
        out[f"ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()
    return out


def add_weather_features(data):
    """Domain-motivated weather transforms."""
    out = data.copy()
    out["temp_national"] = out[["T2M_toc", "T2M_san", "T2M_dav"]].mean(axis=1)
    out["humidity_national"] = out[["QV2M_toc", "QV2M_san", "QV2M_dav"]].mean(axis=1)

    COMFORT_TEMP_C = 24.0
    out["cooling_degree_hours"] = np.maximum(out["temp_national"] - COMFORT_TEMP_C, 0)
    out["temp_roll_mean_24"] = out["temp_national"].rolling(24, min_periods=1).mean()
    out["temp_change_24"] = out["temp_national"] - out["temp_national"].shift(24)
    return out


def build_features(raw_df, target_col, lags, rolling_windows):
    """Run the full feature pipeline in the same order used at training time.

    `raw_df` must have a DatetimeIndex and contain the target column plus the
    raw weather/calendar columns from the original dataset (temperatures
    already converted to Celsius, as the notebook does immediately after load).
    """
    out = raw_df.copy()
    out = add_calendar_features(out)
    out = add_cyclical_features(out)
    out = add_lag_features(out, target_col, lags)
    out = add_rolling_features(out, target_col, rolling_windows)
    out = add_weather_features(out)
    return out
