import numpy as np
import pandas as pd

from pydantic import BaseModel
import logging
from pathlib import Path

import joblib
from fastapi import FastAPI

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart City Traffic Prediction API"
)

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "traffic_volume_rf_v1.joblib"
)

model_bundle = joblib.load(model_path)

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]
model_version = model_bundle["version"]

logger.info(
    "Traffic model %s loaded successfully.",
    model_version
)

class TrafficInput(BaseModel):
    hour: int
    day_of_week: int
    weekend: int
    holiday: int
    temp: float
    rain_1h: float
    snow_1h: float
    clouds_all: float
    weather_main: str

def build_feature_row(data: TrafficInput):
    row = {
        column: 0
        for column in feature_columns
    }

    row["Hour"] = data.hour
    row["Day_of_Week_Num"] = data.day_of_week
    row["Weekend_Indicator"] = data.weekend

    row["Hour_sin"] = np.sin(
        2 * np.pi * data.hour / 24
    )

    row["Hour_cos"] = np.cos(
        2 * np.pi * data.hour / 24
    )

    row["Day_sin"] = np.sin(
        2 * np.pi * data.day_of_week / 7
    )

    row["Day_cos"] = np.cos(
        2 * np.pi * data.day_of_week / 7
    )

    row["Holiday_Flag"] = data.holiday

    row["temp"] = data.temp
    row["rain_1h"] = data.rain_1h
    row["snow_1h"] = data.snow_1h
    row["clouds_all"] = data.clouds_all

    weather_column = f"Weather_{data.weather_main}"

    if weather_column in row:
        row[weather_column] = 1

    return pd.DataFrame(
        [row],
        columns=feature_columns
    )
@app.post("/predict")
def predict_traffic(data: TrafficInput):
    feature_row = build_feature_row(data)

    prediction = model.predict(
        feature_row
    )[0]

    logger.info(
        "Traffic prediction generated using model %s.",
        model_version
    )

    return {
        "model_version": model_version,
        "predicted_traffic_volume": round(
            float(prediction),
            2
        )
    }

    
@app.get("/")
def home():
    return {
        "message": "Traffic Prediction API is running",
        "model_version": model_version
    }

