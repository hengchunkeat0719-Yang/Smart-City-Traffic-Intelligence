import argparse
import logging
from pathlib import Path

import pandas as pd


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
        mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

def load_processed_data(file_path):
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=["date_time"]
        )

        logger.info(
            "Processed dataset loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load processed dataset from %s",
            file_path,
            exc_info=True
        )

        return None


def show_high_traffic_periods(df):
    logger.info("Command invoked: high-traffic")

    hourly_traffic = (
        df
        .groupby("Hour")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\nTop 5 High-Traffic Hours:")
    print(hourly_traffic)


def compare_weekday_weekend(df):
    logger.info("Command invoked: compare-weekend")

    comparison = (
        df
        .groupby("Weekend_Indicator")["traffic_volume"]
        .mean()
    )

    weekday_average = comparison.get(0, 0)
    weekend_average = comparison.get(1, 0)

    print("\nAverage Traffic Volume:")
    print(f"Weekday: {weekday_average:.2f}")
    print(f"Weekend: {weekend_average:.2f}")


def query_traffic_by_datetime(df, date_time_text):
    logger.info(
        "Command invoked: query-time with argument: %s",
        date_time_text
    )

    try:
        requested_time = pd.to_datetime(date_time_text)

    except (ValueError, TypeError):
        logger.error(
            "Invalid date/time supplied: %s",
            date_time_text
        )

        print(
            "Invalid date/time. "
            "Please use a format such as 2017-01-01 08:00:00."
        )
        return

    result = df[
        df["date_time"] == requested_time
    ]

    if result.empty:
        print("\nNo traffic record was found for that date/time.")
        return

    print(
        result[
            [
                "date_time",
                "traffic_volume",
                "weather_main",
                "temp",
                "Congestion_Category"
            ]
        ]
    )

def create_parser():
    parser = argparse.ArgumentParser(
        description="Smart City Traffic Analytics CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    subparsers.add_parser(
        "high-traffic",
        help="Show the top five high-traffic hours."
    )

    subparsers.add_parser(
        "compare-weekend",
        help="Compare average weekday and weekend traffic."
    )

    query_parser = subparsers.add_parser(
        "query-time",
        help="Query traffic for a specific date and time."
    )

    query_parser.add_argument(
        "datetime",
        help='Date/time such as "2017-01-01 08:00:00".'
    )

    return parser

def main():
    base_dir = Path(__file__).resolve().parent.parent
    
    log_file = base_dir / "cli_app" / "traffic_cli.log"

    configure_logging(log_file)

    data_file = (
        base_dir
        / "data"
        / "processed"
        / "traffic_features.csv"
    )

    df = load_processed_data(data_file)

    if df is None:
        print("Unable to load the processed traffic dataset.")
        return

    parser = create_parser()
    args = parser.parse_args()

    if args.command == "high-traffic":
        show_high_traffic_periods(df)

    elif args.command == "compare-weekend":
        compare_weekday_weekend(df)

    elif args.command == "query-time":
        query_traffic_by_datetime(
            df,
            args.datetime
        )


if __name__ == "__main__":
    main()

