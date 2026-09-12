import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def plot_traffic_by_hour(df, output_path):
    hourly_traffic = (
        df
        .groupby("Hour")["traffic_volume"]
        .mean()
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o"
    )

    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.xticks(range(0, 24))

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to %s",
        output_path
    )

def plot_weekday_vs_weekend(df, output_path):
    weekend_traffic = (
        df
        .groupby("Weekend_Indicator")["traffic_volume"]
        .mean()
    )

    weekend_traffic.index = [
        "Weekday",
        "Weekend"
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        weekend_traffic.index,
        weekend_traffic.values
    )

    plt.title("Average Traffic Volume: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Average Traffic Volume")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to %s",
        output_path
    )


def plot_traffic_by_weather(df, output_path):
    weather_traffic = (
        df
        .groupby("weather_main")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        weather_traffic.index,
        weather_traffic.values
    )

    plt.title("Average Traffic Volume by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Average Traffic Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully to %s",
        output_path
    )


