# Part 2 – Python Traffic Analytics Pipeline

## Project Overview

This project is Part 2 of the Smart City Traffic Intelligence capstone.

The objective is to build a reproducible Python-based traffic analytics pipeline that cleans and validates traffic data, creates machine-learning-ready features, generates traffic visualisations, records processing events through logging, and provides a command-line application for querying traffic information.

## Project Structure

part2_python/
│
├── pipeline.py
├── feature_engineering.py
├── visualizations.py
├── pipeline.log
│
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv
│   └── processed/
│       ├── cleaned_traffic_data.csv
│       └── traffic_features.csv
│
├── figures/
│   ├── average_traffic_by_hour.png
│   ├── weekday_vs_weekend_traffic.png
│   └── traffic_by_weather_condition.png
│
├── cli_app/
│   ├── traffic_cli.py
│   └── traffic_cli.log
│
├── notebooks/
│   └── part2_development.ipynb
│
└── README.md

## Data Pipeline

The main pipeline is implemented in `pipeline.py`.

It performs the following steps:

1. Loads the raw traffic CSV.
2. Validates the expected data schema.
3. Removes exact duplicate rows.
4. Parses and validates date/time values.
5. Standardises categorical variables.
6. Imputes invalid 0 Kelvin temperature readings using monthly medians.
7. Imputes extreme rainfall readings using monthly medians.
8. Validates numerical ranges.
9. Saves the cleaned dataset.
10. Performs feature engineering.
11. Saves the feature-engineered dataset.
12. Generates three Matplotlib visualisations.

## Feature Engineering

`feature_engineering.py` creates features including:

- Hour of day
- Day of week
- Weekend indicator
- Cyclical hour encoding using sine and cosine
- Rain indicator
- One-hot encoded weather categories
- Scaled temperature
- Scaled cloud coverage
- Data-driven congestion category

The congestion category is created using the 25th and 75th percentiles of traffic volume.

## Visualisations

`visualizations.py` generates:

- Average traffic volume by hour
- Average weekday versus weekend traffic
- Average traffic volume by weather condition

The generated figures are saved in the `figures/` directory.

## Running the Pipeline

From the `part2_python` directory, run:

```powershell
python pipeline.py
```
