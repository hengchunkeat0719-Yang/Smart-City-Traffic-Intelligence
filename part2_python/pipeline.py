import logging
from pathlib import Path

import pandas as pd

from feature_engineering import engineer_features, save_feature_data

from visualizations import (
    plot_traffic_by_hour,
    plot_weekday_vs_weekend,
    plot_traffic_by_weather
)

logger = logging.getLogger(__name__)

def configure_logging(log_file):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        mode="w"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if not root_logger.handlers:
       root_logger.addHandler(console_handler)
       root_logger.addHandler(file_handler)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Raw data loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError):
        logger.error(
            "Failed to load raw data from %s",
            file_path,
            exc_info=True
        )

        return None

def validate_schema(df):
    expected_columns = [
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "date_time",
        "traffic_volume"
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Schema validation failed. Missing columns: %s",
            missing_columns
        )
        return False

    logger.info("Schema validation passed. All expected columns are present.")
    return True

def remove_duplicates(df):
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().copy()

        logger.warning(
            "%d exact duplicate rows were removed.",
            duplicate_count
        )
    else:
        logger.info("No exact duplicate rows were found.")

    return df

def parse_datetime(df):
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_date_count = df["date_time"].isna().sum()

    if invalid_date_count > 0:
        df = df.dropna(subset=["date_time"]).copy()

        logger.warning(
            "%d rows were removed because date_time could not be parsed.",
            invalid_date_count
        )
    else:
        logger.info("All date_time values were parsed successfully.")

    return df

def clean_categorical_values(df):
    holiday_missing = df["holiday"].isna().sum()

    if holiday_missing > 0:
        df["holiday"] = df["holiday"].fillna("None")

        logger.warning(
            "%d missing holiday values were replaced with 'None'.",
            holiday_missing
        )

    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description"
    ]

    for column in categorical_columns:
        df[column] = df[column].str.strip()

    logger.info(
        "Categorical values were standardised for: %s",
        categorical_columns
    )

    return df

def clean_temperature(df):
    df["month"] = df["date_time"].dt.month

    valid_temp_df = df[df["temp"] > 0]

    monthly_temp_median = (
        valid_temp_df
        .groupby("month")["temp"]
        .median()
    )

    total_imputed = 0

    for month, median_temp in monthly_temp_median.items():
        mask = (
            (df["month"] == month) &
            (df["temp"] == 0)
        )

        affected_rows = mask.sum()

        if affected_rows > 0:
            df.loc[mask, "temp"] = median_temp
            total_imputed += affected_rows

    if total_imputed > 0:
        logger.warning(
            "%d zero-Kelvin temperature readings were imputed using monthly medians.",
            total_imputed
        )
    else:
        logger.info("No zero-Kelvin temperature readings were found.")

    return df

def clean_rainfall(df):
    valid_rain_df = df[df["rain_1h"] <= 9000]

    monthly_rain_median = (
        valid_rain_df
        .groupby("month")["rain_1h"]
        .median()
    )

    total_imputed = 0

    for month, median_rain in monthly_rain_median.items():
        mask = (
            (df["month"] == month) &
            (df["rain_1h"] > 9000)
        )

        affected_rows = mask.sum()

        if affected_rows > 0:
            df.loc[mask, "rain_1h"] = median_rain
            total_imputed += affected_rows

    if total_imputed > 0:
        logger.warning(
            "%d extreme rainfall readings were imputed using monthly medians.",
            total_imputed
        )
    else:
        logger.info("No extreme rainfall readings above 9000 mm were found.")

    return df

def validate_numeric_ranges(df):
    invalid_snow = (df["snow_1h"] < 0).sum()

    invalid_clouds = (
        (df["clouds_all"] < 0) |
        (df["clouds_all"] > 100)
    ).sum()

    invalid_traffic = (df["traffic_volume"] < 0).sum()

    if invalid_snow > 0:
        logger.warning(
            "%d negative snow_1h values were detected.",
            invalid_snow
        )
    else:
        logger.info("No invalid snow_1h values were found.")

    if invalid_clouds > 0:
        logger.warning(
            "%d clouds_all values outside the 0-100 range were detected.",
            invalid_clouds
        )
    else:
        logger.info("All clouds_all values are within the 0-100 range.")

    if invalid_traffic > 0:
        logger.warning(
            "%d negative traffic_volume values were detected.",
            invalid_traffic
        )
    else:
        logger.info("No negative traffic_volume values were found.")

    return df


def save_cleaned_data(df, output_path):
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
            "Cleaned dataset saved successfully to %s",
            output_path
        )

    except (OSError, PermissionError):
        logger.error(
            "Failed to save cleaned dataset to %s",
            output_path,
            exc_info=True
        )
        return False

    return True


def main():
    base_dir = Path(__file__).resolve().parent

    raw_file = (
        base_dir
        / "data"
        / "raw"
        / "Metro_Interstate_Traffic_Volume.csv"
    )

    output_file = (
        base_dir
        / "data"
        / "processed"
        / "cleaned_traffic_data.csv"
    )

    feature_output_file = (
    base_dir
    / "data"
    / "processed"
    / "traffic_features.csv"
    )

    figure_hourly = (
    base_dir
    / "figures"
    / "average_traffic_by_hour.png"
    )

    figure_weekend = (
    base_dir
    / "figures"
    / "weekday_vs_weekend_traffic.png"
    )

    figure_weather = (
    base_dir
    / "figures"
    / "traffic_by_weather_condition.png"
    )

    log_file = base_dir / "pipeline.log"

    configure_logging(log_file)

    logger.info("Traffic data pipeline started.")

    df = load_data(raw_file)

    if df is None:
        logger.error("Pipeline stopped because the raw dataset could not be loaded.")
        return

    if not validate_schema(df):
        logger.error("Pipeline stopped because schema validation failed.")
        return

    df = remove_duplicates(df)
    df = parse_datetime(df)
    df = clean_categorical_values(df)
    df = clean_temperature(df)
    df = clean_rainfall(df)
    df = validate_numeric_ranges(df)

    if not save_cleaned_data(df, output_file):
        logger.error("Pipeline stopped because the cleaned dataset could not be saved.")
        return

    feature_df = engineer_features(df.copy())

    if not save_feature_data(
        feature_df,
        feature_output_file
    ):
        logger.error(
            "Pipeline stopped because the feature-engineered dataset could not be saved."
        )
        return

    plot_traffic_by_hour(
        feature_df,
        figure_hourly
    )

    plot_weekday_vs_weekend(
        feature_df,
        figure_weekend
    )

    plot_traffic_by_weather(
        feature_df,
        figure_weather
    )

    logger.info(
        "Traffic data pipeline completed successfully: %d rows, %d columns.",
        feature_df.shape[0],
        feature_df.shape[1]
    )


if __name__ == "__main__":
    main()


    
