# Part 3 — Machine Learning and AI: Intelligent Mobility Solution

## Project Overview

This project extends the traffic analytics and Python data-processing work completed in Parts 1 and 2 into an intelligent mobility solution.

Part 3 applies supervised machine learning, unsupervised learning, neural networks, SHAP explainability, MLflow experiment tracking, a travel-timing recommendation system, FastAPI deployment simulation, model monitoring, and Responsible AI considerations.

The main regression objective is to predict traffic volume using time, weather, holiday, and cyclical features. The Random Forest Regressor was selected as the deployment model after achieving an MAE of approximately **271.26 vehicles** and an R² of **0.9442**.

---

## Accident-Risk Data Note

No real accident dataset was provided for this capstone. Therefore, the classification task uses a documented proxy `high_risk` label based on High/Severe congestion occurring together with severe or low-visibility weather conditions.

This proxy is used only to demonstrate the machine-learning classification workflow and must not be interpreted as a prediction of actual accidents.

---

## Project Structure

```text
part3_machine_learning/
├── notebooks/
│   └── part3_ml_development.ipynb
│
├── models/
│   └── traffic_volume_rf_v1.joblib
│
├── mlflow/
│   └── mlflow.db
│
├── deployment/
│   └── app.py
│
├── recommendation_system/
│   └── recommendation_engine.py
│
├── monitoring/
│   └── monitor_model.py
│
├── reports/
│   ├── bias_and_fairness_report.pdf
│   └── final_capstone_report.pdf
│
└── README.md
```

---

## Task 1 — Supervised Machine Learning

Two supervised-learning problems were developed.

### Classification

The classification task predicts the proxy `high_risk` traffic condition.

Two algorithms were compared:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9653 | 0.7736 | 0.9798 | 0.8646 | 0.9937 |
| Random Forest Classifier | 0.9838 | 0.9252 | 0.9320 | 0.9286 | 0.9965 |

The Random Forest Classifier achieved stronger overall classification performance, while Logistic Regression achieved slightly higher recall.

### Regression

The regression task predicts traffic volume.

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 813.47 | 0.7260 |
| Random Forest Regressor | 271.26 | 0.9442 |

The Random Forest Regressor achieved the best regression performance and was selected for deployment.

---

## Task 2 — Unsupervised Machine Learning

### K-Means Clustering

K-means clustering was applied using:

- Hour
- Weather severity
- Traffic volume

Three traffic-condition clusters were identified:

1. **Low-Traffic Overnight**
   - Average hour: 2.93
   - Average traffic volume: 816.25

2. **High-Traffic Mild Weather**
   - Average hour: 15.12
   - Average traffic volume: 4,254.11

3. **High-Traffic Adverse Weather**
   - Average hour: 13.79
   - Average traffic volume: 4,091.23

### Association Rule Mining

Association rules were generated using time of day, weekday/weekend status, weather, and congestion level.

Examples of important rules include:

- **Overnight + Weekend → Low Congestion**
  - Confidence: 88.61%
  - Lift: 3.54

- **Weekday + Afternoon → Severe Congestion**
  - Confidence: 71.33%
  - Lift: 2.86

These rules provide interpretable information about when different congestion conditions are more likely to occur.

---

## Task 3 — Neural Network and Explainable AI

A feed-forward neural network was developed for traffic-volume prediction using two hidden layers:

- 64 neurons
- 32 neurons
- ReLU activation
- Adam optimiser
- Early stopping

Performance:

- MAE: **294.53 vehicles**
- R²: **0.9403**

The neural network performed strongly but was slightly weaker than the Random Forest Regressor.

### SHAP Explainability

SHAP was applied to the Random Forest Regressor trained on the same traffic-volume prediction problem.

The most influential features included:

1. `Hour_cos`
2. `Hour`
3. `Hour_sin`
4. `Day_of_Week_Num`
5. `Weekend_Indicator`
6. `Day_sin`
7. `temp`
8. `Day_cos`
9. `clouds_all`

The results indicate that recurring time-of-day and day-of-week patterns are major drivers of predicted traffic volume.

Cyclical variables such as `Hour_cos` and `Hour_sin` represent time patterns mathematically and should not be interpreted as direct causal factors.

---

## Task 4 — Advanced AI Technique: MLflow

MLflow was used for experiment tracking.

The following regression experiments were recorded:

- Linear Regression
- Random Forest Regressor
- Neural Network

MLflow was used to record:

- Model parameters
- Performance metrics
- Experiment runs
- Model information

A local SQLite database is stored at:

```text
mlflow/mlflow.db
```

The Random Forest Regressor achieved the strongest regression performance and was selected as the deployment model.

A shorter local artifact path was used during development because the project directory is deeply nested on Windows and MLflow encountered Windows path-length limitations.

---

## Task 5 — Traffic Recommendation System

The recommendation engine provides **travel-timing recommendations** based on historical traffic patterns.

Because the dataset represents one traffic corridor rather than multiple alternative routes, the recommendation system focuses on **when to travel instead of which route to take**.

Recommendations consider:

- Weekday or weekend
- Weather condition
- Hour of day
- Historical average traffic volume

Practical travel hours were restricted to:

```text
06:00 – 22:00
```

A minimum of 20 historical observations was required for a day/weather/hour combination before it was used to generate a recommendation.

Example:

> For a weekday journey in clear weather, consider travelling between 22:00 and 23:00, when historical traffic volume averages about 2,223 vehicles.

### Run the Recommendation Engine

From the `part3_machine_learning` folder:

```powershell
python recommendation_system/recommendation_engine.py
```

---

## Task 6 — MLOps, Deployment and Monitoring

### Model Versioning

The project documents three model candidates:

| Version | Model | MAE | R² | Status |
|---|---|---:|---:|---|
| v0 | Linear Regression | 813.47 | 0.7260 | Baseline |
| v1 | Random Forest Regressor | 271.26 | 0.9442 | Selected for Deployment |
| v2 | Neural Network | 294.53 | 0.9403 | Candidate |

The selected model is stored as:

```text
models/traffic_volume_rf_v1.joblib
```

### FastAPI Deployment

A FastAPI application was created to simulate model deployment.

The API accepts traffic-related information including:

- Hour
- Day of week
- Weekend indicator
- Holiday indicator
- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather condition

It returns a predicted traffic volume.

### Start the FastAPI Application

From the `part3_machine_learning` folder:

```powershell
python -m uvicorn deployment.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page provides FastAPI's interactive testing interface.

Prediction endpoint:

```text
POST /predict
```

Example response:

```json
{
  "model_version": "v1",
  "predicted_traffic_volume": 5662.9
}
```

---

## Model Monitoring

The monitoring component compares model error against the original deployment baseline.

Baseline MAE:

```text
271.26
```

Alert threshold:

```text
325.51
```

The threshold was defined as 20% above the baseline MAE for this capstone simulation.

Monitoring status:

```text
PASS / Normal
```

If the MAE exceeds the threshold, the system reports:

```text
ALERT / Requires investigation
```

For example, a simulated MAE of 350 correctly triggers an alert.

### Run Model Monitoring

From the `part3_machine_learning` folder:

```powershell
python monitoring/monitor_model.py
```

---

## Task 7 — Responsible AI, Bias, Fairness and Sustainability

The project evaluates several Responsible AI considerations.

### Data Coverage

The traffic dataset does not provide equal coverage across every year and represents only one traffic corridor.

Therefore, results should not automatically be generalised to an entire transport network.

### Proxy Label Risk

The `high_risk` variable is a constructed proxy and is not based on real accident records.

Strong classification performance does not demonstrate that the system can predict real accidents.

A production accident-risk system would require validated accident or road-safety data.

### Uneven Model Performance

Model accuracy may vary across:

- Different hours
- Weekdays and weekends
- Congestion conditions
- Common and rare weather conditions

Rare conditions such as fog, squall or severe weather may have fewer observations and therefore require additional evaluation.

### Human Oversight

Traffic predictions should support rather than replace human decision-making.

Traffic planners or transport operations staff should review:

- Model performance
- Monitoring alerts
- Unusual weather
- Road incidents
- Unexpected traffic patterns

before using predictions for operational decisions.

### Sustainability

Computational cost should also be considered when selecting AI models.

The Random Forest Regressor slightly outperformed the neural network while providing a simpler modelling and deployment approach.

Using an appropriately sized model can reduce:

- Computation time
- Energy consumption
- Infrastructure requirements
- Unnecessary retraining

More detailed discussion is available in:

```text
reports/bias_and_fairness_report.pdf
```

---

## Data Source

Part 3 uses the processed traffic dataset generated during Part 2:

```text
part2_python/data/processed/traffic_features.csv
```

The Part 3 scripts therefore expect the Part 2 folder to remain within the same repository.

---

## Main Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- mlxtend
- SHAP
- MLflow
- FastAPI
- Uvicorn
- joblib
- Jupyter Notebook

---

## How to Review the Project

For a complete review of Part 3:

1. Open `notebooks/part3_ml_development.ipynb` for model development and analysis.
2. Review `reports/final_capstone_report.pdf` for the complete methodology and findings.
3. Review `reports/bias_and_fairness_report.pdf` for Responsible AI considerations.
4. Run `recommendation_system/recommendation_engine.py` to test the recommendation system.
5. Run `monitoring/monitor_model.py` to test model monitoring.
6. Start `deployment/app.py` through Uvicorn to test the prediction API.

---

## Important Limitation

This project is an educational capstone and deployment simulation.

The models have not been validated for live transport operations. Real-world implementation would require additional data validation, chronological testing, production monitoring, security controls, human oversight and testing using current mobility data.