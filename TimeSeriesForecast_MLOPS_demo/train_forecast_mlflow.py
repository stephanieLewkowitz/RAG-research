"""
Claims / Call Center Volume Forecasting Demo
-------------------------------------------
Synthetic healthcare claims/call-volume forecasting pipeline using:
- generated time-series data
- RandomForestRegressor
- lag and rolling-window features
- MLflow experiment tracking
- saved plots and prediction artifacts

Run locally:
    python train_forecast_mlflow.py

View MLflow UI:
    mlflow ui --backend-store-uri ./mlruns

Then open:
    http://localhost:5000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# 1. Generate synthetic business data
# -----------------------------
def generate_synthetic_claims_data(n_days: int = 730) -> pd.DataFrame:
    """
    Generate synthetic daily claims/call center volume data.

    This simulates a benefits/claims environment with:
    - weekly seasonality
    - annual seasonality
    - gradual growth trend
    - random operational noise
    - occasional spikes, such as open enrollment or claims backlog
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    day_index = np.arange(n_days)
    day_of_week = dates.dayofweek

    # Base daily workload
    base_volume = 500

    # Claims/call centers are often busier on Mondays and quieter on weekends
    weekly_pattern = np.where(day_of_week == 0, 90, 0)  # Monday spike
    weekly_pattern += np.where(day_of_week == 4, 35, 0)  # Friday bump
    weekly_pattern -= np.where(day_of_week >= 5, 160, 0)  # weekend drop

    # Annual/seasonal utilization cycle
    annual_seasonality = 55 * np.sin(2 * np.pi * day_index / 365)

    # Gradual business growth / claim volume growth
    trend = 0.12 * day_index

    # Random noise
    noise = rng.normal(0, 35, n_days)

    # Operational spikes: e.g., open enrollment, billing cycle issue, weather event
    spikes = np.zeros(n_days)
    spike_days = rng.choice(n_days, size=12, replace=False)
    spikes[spike_days] = rng.integers(120, 280, size=len(spike_days))

    call_volume = base_volume + weekly_pattern + annual_seasonality + trend + noise + spikes
    call_volume = np.maximum(call_volume, 50).round().astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "call_volume": call_volume,
            "day_of_week": day_of_week,
            "month": dates.month,
            "is_weekend": (day_of_week >= 5).astype(int),
        }
    )

    return df


# -----------------------------
# 2. Feature engineering
# -----------------------------
def create_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features and rolling-window features from historical volume.
    These are common production features for operational forecasting.
    """
    df = df.copy()

    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = df["call_volume"].shift(lag)

    df["rolling_mean_7"] = df["call_volume"].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df["call_volume"].shift(1).rolling(window=14).mean()
    df["rolling_std_7"] = df["call_volume"].shift(1).rolling(window=7).std()

    # Calendar features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Drop early rows with missing lag values
    df = df.dropna().reset_index(drop=True)

    return df


# -----------------------------
# 3. Train/test split
# -----------------------------
def time_based_train_test_split(
    df: pd.DataFrame, test_days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use time-based split instead of random split.
    This prevents future data leaking into the training set.
    """
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    return train_df, test_df


# -----------------------------
# 4. Plotting helpers
# -----------------------------
def save_forecast_plot(test_df: pd.DataFrame, predictions: np.ndarray) -> str:
    plot_path = OUTPUT_DIR / "forecast_vs_actual.png"

    plt.figure(figsize=(12, 6))
    plt.plot(test_df["date"], test_df["call_volume"], label="Actual")
    plt.plot(test_df["date"], predictions, label="Predicted")
    plt.title("Claims Call Volume Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Daily Call Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return str(plot_path)


def save_feature_importance_plot(model: RandomForestRegressor, feature_cols: list[str]) -> str:
    plot_path = OUTPUT_DIR / "feature_importance.png"

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return str(plot_path)


# -----------------------------
# 5. Main training function with MLflow
# -----------------------------
def train_and_log_model() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("claims_call_volume_forecasting")

    # Hyperparameters
    n_estimators = 300
    max_depth = 12
    min_samples_leaf = 3
    test_days = 90

    raw_df = generate_synthetic_claims_data(n_days=730)
    feature_df = create_forecasting_features(raw_df)
    train_df, test_df = time_based_train_test_split(feature_df, test_days=test_days)

    target_col = "call_volume"
    drop_cols = ["date", target_col]
    feature_cols = [col for col in feature_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="random_forest_claims_forecast"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("test_days", test_days)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("target", target_col)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        forecast_plot = save_forecast_plot(test_df, predictions)
        feature_plot = save_feature_importance_plot(model, feature_cols)

        predictions_df = test_df[["date", "call_volume"]].copy()
        predictions_df["predicted_call_volume"] = predictions.round(2)
        predictions_df["residual"] = predictions_df["call_volume"] - predictions_df["predicted_call_volume"]

        predictions_path = OUTPUT_DIR / "predictions.csv"
        raw_data_path = OUTPUT_DIR / "synthetic_claims_call_volume.csv"
        predictions_df.to_csv(predictions_path, index=False)
        raw_df.to_csv(raw_data_path, index=False)

        mlflow.log_artifact(forecast_plot)
        mlflow.log_artifact(feature_plot)
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(raw_data_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        print("Training complete.")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.3f}")
        print("\nArtifacts saved to:", OUTPUT_DIR.resolve())
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        print("\nRun this to view results:")
        print("mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    train_and_log_model()
