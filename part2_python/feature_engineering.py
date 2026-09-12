import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

def create_time_features(df):
    df["Hour"] = df["date_time"].dt.hour

    df["Day_of_Week"] = df["date_time"].dt.day_name()

    df["Weekend_Indicator"] = (
        df["date_time"].dt.dayofweek >= 5
    ).astype(int)

    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    logger.info(
        "Time features created: Hour, Day_of_Week, Weekend_Indicator, Hour_sin, Hour_cos."
    )

    return df


def create_weather_features(df):
    df["Rain_Indicator"] = (
        df["rain_1h"] > 0
    ).astype(int)

    weather_encoded = pd.get_dummies(
        df["weather_main"],
        prefix="Weather",
        dtype=int
    )

    df = pd.concat(
        [df, weather_encoded],
        axis=1
    )

    logger.info(
        "Weather features created, including Rain_Indicator and one-hot encoded weather categories."
    )

    return df

def create_scaled_features(df):
    df["Temp_Scaled"] = (
        (df["temp"] - df["temp"].min()) /
        (df["temp"].max() - df["temp"].min())
    )

    df["Clouds_Scaled"] = (
        (df["clouds_all"] - df["clouds_all"].min()) /
        (df["clouds_all"].max() - df["clouds_all"].min())
    )

    logger.info(
        "Scaled numerical features created: Temp_Scaled and Clouds_Scaled."
    )

    return df


def create_congestion_category(df):
    q1 = df["traffic_volume"].quantile(0.25)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug(
        "Congestion thresholds calculated: Q1=%.2f, Q3=%.2f",
        q1,
        q3
    )

    conditions = [
        df["traffic_volume"] <= q1,
        df["traffic_volume"] > q3
    ]

    choices = [
        "Low",
        "High"
    ]

    df["Congestion_Category"] = np.select(
        conditions,
        choices,
        default="Medium"
    )

    logger.info(
        "Congestion category created using traffic-volume quartiles."
    )

    return df

def engineer_features(df):
    logger.info(
        "Dataset shape before feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )

    df = create_time_features(df)
    df = create_weather_features(df)
    df = create_scaled_features(df)
    df = create_congestion_category(df)

    logger.info(
        "Dataset shape after feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )

    return df


def save_feature_data(df, output_path):
    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Feature-engineered dataset saved successfully to %s",
            output_path
        )

        return True

    except (OSError, PermissionError):
        logger.error(
            "Failed to save feature-engineered dataset to %s",
            output_path,
            exc_info=True
        )

        return False




