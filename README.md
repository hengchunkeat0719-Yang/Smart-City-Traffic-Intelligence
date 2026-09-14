# Smart City Traffic Intelligence

## Project Overview

This is my AI, Machine Learning, and Data Science capstone project.

The project uses the **Metro Interstate Traffic Volume** dataset to study traffic patterns and explore how data analytics and AI can support smarter mobility decisions.

I completed the project in three parts:

1. **Part 1 — Data Analytics**
   - Understand the dataset
   - Analyse traffic patterns
   - Use SQL and Power BI
   - Study statistics and probability

2. **Part 2 — Python Data Processing**
   - Clean the data using Python
   - Create new features
   - Build a reusable data pipeline
   - Create charts
   - Add logging
   - Build a simple command-line application

3. **Part 3 — Machine Learning and AI**
   - Build classification and regression models
   - Use clustering and association rules
   - Build a neural network
   - Use SHAP for explainability
   - Track models with MLflow
   - Build a traffic recommendation system
   - Create a FastAPI deployment simulation
   - Monitor model performance
   - Review bias, fairness, governance, and sustainability

---

## Repository Structure

```text
Smart-City-Traffic-Intelligence/
├── part1_data_analytics/
│   └── Part 1 analysis, SQL, Power BI and reports
│
├── part2_python/
│   ├── pipeline.py
│   ├── feature_engineering.py
│   ├── visualizations.py
│   ├── pipeline.log
│   ├── data/
│   ├── figures/
│   ├── cli_app/
│   ├── notebooks/
│   ├── report/
│   └── README.md
│
├── part3_machine_learning/
│   ├── notebooks/
│   ├── models/
│   ├── mlflow/
│   ├── deployment/
│   ├── recommendation_system/
│   ├── monitoring/
│   ├── reports/
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## Tools and Technologies Used

During this project, I used:

- Python
- pandas
- NumPy
- scikit-learn
- SQL
- Power BI
- Jupyter Notebook
- mlxtend
- SHAP
- MLflow
- FastAPI
- Uvicorn
- joblib
- Git
- GitHub

---

# Part 1 — Data Analytics

The main purpose of Part 1 was to understand the traffic data before building any machine learning models.

I explored:

- Data quality
- Traffic volume
- Mean, median, standard deviation and variance
- Traffic by hour
- Traffic by weather
- Probability
- SQL queries
- Power BI visualisations

One important finding was that traffic volume changes a lot depending on the time of day.

For example, peak-hour traffic is much higher than overnight traffic.

---

# Part 2 — Python Data Processing

In Part 2, I created a Python pipeline so that the data-processing steps can be repeated automatically.

The pipeline performs tasks such as:

- Loading the raw dataset
- Removing duplicate records
- Handling unusual or missing values
- Creating new features
- Creating traffic charts
- Saving processed datasets
- Writing log messages

Some of the new features include:

- Hour
- Day of week
- Weekend indicator
- Cyclical hour features
- Weather indicators
- Congestion category

---

## How to Run the Python Pipeline

First, open a terminal inside:

```text
part2_python/
```

Then run:

```powershell
python pipeline.py
```

The pipeline will process the raw traffic data and create cleaned and engineered datasets.

Processed data is saved in:

```text
part2_python/data/processed/
```

Charts are saved in:

```text
part2_python/figures/
```

---

## Mini-Application

I also created a simple command-line application in Part 2.

To run it, open a terminal in the `part2_python` folder and run:

```powershell
python cli_app/traffic_cli.py
```

The command-line application allows a user to interact with traffic information in a simple way.

---

# Logging

Logging is used to record what happens when the Python programs run.

For example, the logs can show:

- When the program starts
- When data is loaded
- When processing is completed
- When a warning happens
- When an error occurs

The main pipeline log is stored at:

```text
part2_python/pipeline.log
```

The command-line application log is stored at:

```text
part2_python/cli_app/traffic_cli.log
```

Part 3 scripts also use Python logging for model loading, recommendations, monitoring, warnings and errors.

MLflow experiment information is stored in:

```text
part3_machine_learning/mlflow/mlflow.db
```

---

# Part 3 — Machine Learning and AI

In Part 3, I used the processed data from Part 2 to build machine learning and AI solutions.

The main tasks include:

- Classification
- Regression
- K-means clustering
- Association rule mining
- Neural network
- SHAP explainability
- MLflow
- Traffic recommendation system
- FastAPI deployment
- Model monitoring
- Responsible AI

---

# Machine Learning Models

## Classification

The classification task predicts a proxy traffic-risk condition called:

```text
high_risk
```

I used a proxy because the dataset does not contain real accident records.

The proxy is based on:

- High or Severe congestion
- Together with severe or low-visibility weather

I compared two classification models:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9653 | 0.7736 | 0.9798 | 0.8646 | 0.9937 |
| Random Forest Classifier | 0.9838 | 0.9252 | 0.9320 | 0.9286 | 0.9965 |

The **Random Forest Classifier** gave the strongest overall result.

However, these results should not be treated as real accident prediction because the target is only a proxy.

---

## Traffic Volume Regression

I also built models to predict traffic volume.

I compared:

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 813.47 | 0.7260 |
| Random Forest Regressor | 271.26 | 0.9442 |
| Neural Network | 294.53 | 0.9403 |

The **Random Forest Regressor** performed best.

It had:

- Lower prediction error
- Higher R²
- Better overall performance than Linear Regression and the Neural Network

The selected model is saved in:

```text
part3_machine_learning/models/traffic_volume_rf_v1.joblib
```

---

# Neural Network

I created a feed-forward neural network using `MLPRegressor`.

The model uses:

- Feature standardisation
- First hidden layer: 64 neurons
- Second hidden layer: 32 neurons
- ReLU activation
- Adam optimiser
- Early stopping

The neural network achieved:

```text
MAE = 294.53
R² = 0.9403
```

This result was strong, but the Random Forest Regressor still performed slightly better.

---

# Explainable AI with SHAP

Machine learning models can sometimes be difficult to understand.

To make the model easier to explain, I used **SHAP**.

SHAP helps show which features have the biggest influence on predictions.

The SHAP analysis showed that time-related features were very important, including:

- Hour
- Hour cyclical features
- Day of week
- Weekend indicator

This means the model learned that traffic volume follows strong daily and weekly patterns.

---

# Unsupervised Learning

## K-Means Clustering

I used K-means clustering to group similar traffic conditions.

The clustering used:

- Hour
- Weather severity
- Traffic volume

Three main traffic groups were identified:

1. **Low-Traffic Overnight**
2. **High-Traffic Mild Weather**
3. **High-Traffic Adverse Weather**

This helped me understand different traffic patterns without using a target variable.

---

## Association Rule Mining

Association rules were used to find relationships between:

- Time of day
- Weekday or weekend
- Weather
- Congestion level

For example:

```text
Overnight + Weekend → Low Congestion
```

This rule had:

```text
Confidence = 88.61%
Lift = 3.54
```

Another example was:

```text
Weekday + Afternoon → Severe Congestion
```

This means severe congestion was more likely during weekday afternoons.

---

# Traffic Recommendation System

The dataset only represents one traffic corridor.

Because of this, I did not build a route recommendation system.

Instead, I built a **travel-time recommendation system**.

The system recommends when traffic may be lower based on:

- Weekday or weekend
- Weather condition
- Hour
- Historical traffic volume

For example:

> For a weekday journey in clear weather, consider travelling between 22:00 and 23:00, when historical traffic volume averages about 2,223 vehicles.

To run the recommendation engine:

```powershell
python recommendation_system/recommendation_engine.py
```

Run this command from:

```text
part3_machine_learning/
```

---

# MLflow Experiment Tracking

I used MLflow to keep track of machine learning experiments.

MLflow records information such as:

- Model name
- Parameters
- MAE
- R²
- Experiment runs

I tracked:

- Linear Regression
- Random Forest Regressor
- Neural Network

This made it easier to compare the models and decide which one should be used for deployment.

MLflow information is stored in:

```text
part3_machine_learning/mlflow/mlflow.db
```

---

# FastAPI Deployment Simulation

I created a simple API using FastAPI.

The API allows traffic and weather information to be sent to the trained model.

The model then returns a predicted traffic volume.

To start the API, open a terminal in:

```text
part3_machine_learning/
```

Run:

```powershell
python -m uvicorn deployment.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides an interactive testing page.

The prediction endpoint is:

```text
POST /predict
```

An example response is:

```json
{
  "model_version": "v1",
  "predicted_traffic_volume": 5662.9
}
```

---

# Model Monitoring

After a model is deployed, its performance should continue to be checked.

For this project, I created a simple monitoring system.

The original Random Forest MAE is:

```text
271.26
```

I set a monitoring alert threshold at 20% above the baseline:

```text
325.51
```

If the MAE stays below the threshold, the system reports:

```text
PASS / Normal
```

If the MAE becomes higher than the threshold, the system reports:

```text
ALERT / Requires investigation
```

To run the monitoring script:

```powershell
python monitoring/monitor_model.py
```

---

# How to Reproduce the Main Analysis

A new user can reproduce the project by following these steps:

1. Clone or download this GitHub repository.
2. Make sure the raw Metro Interstate Traffic Volume dataset is available in the Part 2 raw-data folder.
3. Run the Part 2 pipeline:

```powershell
python pipeline.py
```

4. Check the processed data and figures.
5. Open:

```text
part3_machine_learning/notebooks/part3_ml_development.ipynb
```

6. Run the notebook cells to reproduce the machine learning analysis.
7. Run the recommendation engine.
8. Run the monitoring script.
9. Start the FastAPI application to test the deployed model.

---

# Important Assumptions and Limitations

This project is an educational capstone project.

Some important limitations are:

### 1. Single traffic corridor

The dataset represents one traffic corridor.

The results should not automatically be applied to an entire city or transport network.

### 2. No real accident data

The dataset does not contain real accident information.

The `high_risk` target is only a proxy based on congestion and weather conditions.

Therefore, this project should **not** be described as a real accident-prediction system.

### 3. Historical data

The models learn from historical traffic patterns.

Future traffic may change because of:

- Road accidents
- Road closures
- New infrastructure
- Special events
- Extreme weather
- Changes in travel behaviour

### 4. Monitoring threshold

The 20% monitoring threshold is an assumption created for this capstone.

It is not an official transport-industry standard.

### 5. Recommendation sample size

The recommendation system requires at least 20 historical observations for a day, weather, and hour combination.

This is also a project assumption.

### 6. Real-world deployment

The models have not been validated for real traffic-management decisions.

A real production system would require:

- More current data
- More testing
- Security controls
- Continuous monitoring
- Human oversight
- Model governance

---

# Responsible AI

Responsible AI is important because traffic predictions could influence real people.

In this project, I considered:

- Data coverage
- Sampling bias
- Proxy-label risk
- Uneven model errors
- Human oversight
- Governance
- Sustainability

The model should be used as **decision support**, not as an automatic decision-maker.

A human traffic planner or transport operations team should review important predictions and alerts before taking action.

---

# Reports

The main Part 3 reports are available in:

```text
part3_machine_learning/reports/
```

The two main reports are:

```text
final_capstone_report.pdf
bias_and_fairness_report.pdf
```

More detailed instructions are also available in:

```text
part2_python/README.md
part3_machine_learning/README.md
```

---

# What I Learned

Through this capstone, I learned how a data project can develop from basic analysis into a complete AI workflow.

The overall process was:

```text
Raw Data
   ↓
Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Explainable AI
   ↓
Recommendation System
   ↓
Deployment
   ↓
Monitoring
   ↓
Responsible AI
```

This project helped me understand that building an AI solution is not only about creating an accurate model.

It also requires good data preparation, clear evaluation, explainability, deployment, monitoring, documentation, and responsible use.
