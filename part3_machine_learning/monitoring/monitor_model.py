import numpy as np

from sklearn.model_selection import train_test_split

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

def load_model_bundle():
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "traffic_volume_rf_v1.joblib"
    )

    try:
        model_bundle = joblib.load(model_path)

        logger.info(
            "Model bundle loaded successfully from %s",
            model_path
        )

        return model_bundle

    except FileNotFoundError:
        logger.error(
            "Model file not found at %s",
            model_path,
            exc_info=True
        )

        return None

def load_monitoring_data():
    data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "part2_python"
        / "data"
        / "processed"
        / "traffic_features.csv"
    )

    try:
        df = pd.read_csv(
            data_path,
            parse_dates=["date_time"]
        )

        logger.info(
            "Monitoring data loaded successfully with %d rows.",
            len(df)
        )

        return df

    except FileNotFoundError:
        logger.error(
            "Monitoring data file not found at %s",
            data_path,
            exc_info=True
        )

        return None

def prepare_monitoring_features(df, feature_columns):
    df = df.copy()

    # Recreate Part 3 features
    df["Day_of_Week_Num"] = df["date_time"].dt.dayofweek

    df["Day_sin"] = np.sin(
        2 * np.pi * df["Day_of_Week_Num"] / 7
    )

    df["Day_cos"] = np.cos(
        2 * np.pi * df["Day_of_Week_Num"] / 7
    )

    df["Holiday_Flag"] = (
        df["holiday"]
        .fillna("None")
        .ne("None")
        .astype(int)
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Monitoring data is missing required features: %s",
            missing_columns
        )
        return None, None

    X_monitor = df[feature_columns].copy()
    y_actual = df["traffic_volume"].copy()

    logger.info(
        "Monitoring features prepared with shape %s.",
        X_monitor.shape
    )

    return X_monitor, y_actual


def select_monitoring_sample(
    X_monitor,
    y_actual
):
    _, X_test, _, y_test = train_test_split(
        X_monitor,
        y_actual,
        test_size=0.20,
        random_state=42
    )

    logger.info(
        "Monitoring test sample prepared with %d rows.",
        len(X_test)
    )

    return X_test, y_test


def evaluate_monitoring_status(
    model,
    X_monitor,
    y_actual,
    baseline_mae,
    threshold_multiplier=1.20
):
    predictions = model.predict(
        X_monitor
    )

    current_mae = mean_absolute_error(
        y_actual,
        predictions
    )

    alert_threshold = (
        baseline_mae
        * threshold_multiplier
    )

    if current_mae <= alert_threshold:
        status = "PASS / Normal"
        logger.info(
            "Monitoring status is PASS. Current MAE: %.2f",
            current_mae
        )
    else:
        status = "ALERT / Requires investigation"
        logger.warning(
            "Monitoring alert triggered. Current MAE: %.2f",
            current_mae
        )

    return {
        "baseline_mae": baseline_mae,
        "alert_threshold": alert_threshold,
        "current_mae": current_mae,
        "status": status
    }

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    model_bundle = load_model_bundle()

    if model_bundle is None:
        return

    df = load_monitoring_data()

    if df is None:
        return

    X_monitor, y_actual = prepare_monitoring_features(
        df,
        model_bundle["feature_columns"]
    )

    if X_monitor is None:
        return

    X_test, y_test = select_monitoring_sample(
        X_monitor,
        y_actual
    )

    results = evaluate_monitoring_status(
        model=model_bundle["model"],
        X_monitor=X_test,
        y_actual=y_test,
        baseline_mae=model_bundle["MAE"]
    )

    print(
        f"Baseline MAE: "
        f"{results['baseline_mae']:.2f}"
    )

    print(
        f"Alert threshold: "
        f"{results['alert_threshold']:.2f}"
    )

    print(
        f"Current MAE: "
        f"{results['current_mae']:.2f}"
    )

    print(
        f"Monitoring status: "
        f"{results['status']}"
    )


if __name__ == "__main__":
    main()


