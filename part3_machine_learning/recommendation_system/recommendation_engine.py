import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def load_data():
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
            "Traffic data loaded successfully with %d rows.",
            len(df)
        )

        return df

    except FileNotFoundError:
        logger.error(
            "Traffic data file not found at %s",
            data_path,
            exc_info=True
        )
        return None

def build_recommendation_table(df):
    df = df.copy()

    df["Day_Type"] = df["Weekend_Indicator"].map({
        0: "Weekday",
        1: "Weekend"
    })

    practical_df = df[
        df["Hour"].between(6, 22)
    ]

    profile = (
        practical_df
        .groupby(
            [
                "Day_Type",
                "weather_main",
                "Hour"
            ]
        )["traffic_volume"]
        .agg(
            Average_Traffic="mean",
            Observations="count"
        )
        .reset_index()
    )

    reliable_profile = profile[
        profile["Observations"] >= 20
    ].copy()

    logger.info(
        "Recommendation table created with %d reliable combinations.",
        len(reliable_profile)
    )

    return reliable_profile

def recommend_travel_time(profile, day_type, weather):
    matches = profile[
        (profile["Day_Type"] == day_type)
        & (profile["weather_main"] == weather)
    ].sort_values(
        "Average_Traffic"
    )

    if matches.empty:
        logger.warning(
            "No reliable recommendation found for %s and %s.",
            day_type,
            weather
        )

        return (
            "No reliable historical recommendation "
            "is available for this combination."
        )

    best = matches.iloc[0]

    start_hour = int(best["Hour"])
    end_hour = start_hour + 1
    avg_traffic = best["Average_Traffic"]

    logger.info(
        "Recommendation generated for %s and %s.",
        day_type,
        weather
    )

    return (
        f"For a {day_type.lower()} journey in {weather.lower()} weather, "
        f"consider travelling between "
        f"{start_hour:02d}:00 and {end_hour:02d}:00, "
        f"when historical traffic volume averages about "
        f"{avg_traffic:.0f} vehicles."
    )

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    df = load_data()

    if df is None:
        return

    profile = build_recommendation_table(df)

    recommendation = recommend_travel_time(
        profile,
        "Weekday",
        "Clear"
    )

    print(recommendation)


if __name__ == "__main__":
    main()



