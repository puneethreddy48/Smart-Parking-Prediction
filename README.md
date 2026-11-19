# 🅿️ Smart Parking Occupancy Prediction Using Machine Learning

## 📌 Project Overview

Smart Parking Occupancy Prediction is a machine learning-based application designed to predict parking occupancy and estimate available parking spaces using historical parking data.

The project combines temporal patterns, historical weather information, and calendar-based features to improve parking occupancy prediction.

An interactive dashboard built using Streamlit allows users to explore parking locations, generate occupancy predictions, visualize parking patterns, and monitor model performance.

## 🎯 Objectives

- Predict parking occupancy using machine learning.
- Estimate available parking spaces for different parking locations.
- Analyze temporal, weather, and calendar-related parking patterns.
- Compare different machine learning models through an ablation study.
- Improve model transparency using SHAP explainability.
- Monitor prediction performance and identify potential concept drift.

## ✨ Key Features

- **Parking Overview:** Displays parking capacity and occupancy information.
- **Interactive Parking Map:** Visualizes parking locations.
- **Parking Prediction:** Predicts occupancy based on the selected parking location and time.
- **Weather Integration:** Incorporates historical weather information into prediction models.
- **Calendar Features:** Considers weekends, school holidays, half-term periods, and Christmas periods.
- **Parking Analytics:** Visualizes occupancy trends and parking usage patterns.
- **Ablation Study:** Compares four feature configurations to understand the contribution of different inputs.
- **SHAP Explainability:** Analyzes feature importance and explains model predictions.
- **Concept Drift Monitoring:** Tracks prediction performance over time.
- **Model Retraining:** Includes a retraining workflow for updating the trained model.

## 🧠 Proposed Methodology

The project uses four experimental configurations to evaluate the impact of different feature groups.

| Experiment | Features |
|---|---|
| E1 | Temporal Features |
| E2 | Temporal + Weather Features |
| E3 | Temporal + Calendar Features |
| E4 | Temporal + Weather + Calendar Features |

The experiments compare Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

This approach helps evaluate how additional contextual information affects parking occupancy prediction.

## 🤖 Machine Learning Models

Three machine learning algorithms are evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📊 Model Performance

The selected E4 Random Forest model achieved the following results using chronological train-test evaluation.

| Metric | E4 Random Forest |
|---|---:|
| MAE | 24.0014 |
| RMSE | 55.7091 |
| R² Score | 0.994059 |

**Selected Model:** E4 Temporal + Weather + Calendar Random Forest Regressor.

The E3 configuration achieved a slightly lower MAE in the experiments. E4 was selected for its inclusion of both weather and calendar context, rather than because it achieved the lowest MAE.

**Baseline Comparison**

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Persistence Baseline | 70.1128 | 172.7265 | 0.942886 |
| E4 Random Forest | 24.0014 | 55.7091 | 0.994059 |

*Note: These are historical dataset evaluation results. They do not guarantee real-time prediction accuracy.*

## 🔍 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) is used to understand the contribution of input features to model predictions.

The explainability analysis helps identify influential features and improves the interpretability of the prediction model.

## 📉 Concept Drift Monitoring

The project includes a concept drift monitoring component to track changes in prediction performance.

It records parking observations and prediction errors to help identify potential performance degradation.

A separate retraining workflow is provided to support model updates when required.

## 📁 Dataset Information

The project uses historical parking occupancy data containing:

- Parking location identifiers
- Parking capacity
- Occupancy values
- Date and time information
- Historical weather information
- Calendar and holiday information

**Dataset Summary:**

| Description | Value |
|---|---:|
| Total Records | 35,459 |
| Parking Locations | 30 |
| Dataset Start | 04 October 2016 |
| Dataset End | 19 December 2016 |

The model uses a chronological 80:20 train-test split to preserve the time order of observations and reduce information leakage from future records.

Weather and calendar features provide additional contextual information for the prediction process.

*Note: The dataset contains historical information and does not represent live parking sensor data.*

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Plotly, Matplotlib |
| Interactive Maps | Folium, Streamlit-Folium |
| Model Storage | Joblib |
| Version Control | Git, GitHub, Git LFS |

## 📂 Project Structure

```text
Smart-Parking-Prediction/
│
├── app.py
├── train_model.py
├── concept_drift.py
├── retrain_model.py
├── add_holiday_features.py
├── get_weather_data.py
├── merge_datasets.py
├── requirements.txt
├── README.md
├── .gitattributes
│
├── dataset/
│   ├── cleaned_parking.csv
│   ├── enhanced_parking.csv
│   ├── holiday_enhanced_parking.csv
│   └── historical_weather_birmingham.csv
│
├── models/
│   ├── shap_analysis.py
│   ├── validate_model.py
│   └── experiments/
│       ├── E1_Temporal_Only_*.pkl
│       ├── E2_Temporal_Weather_*.pkl
│       ├── E3_Temporal_Calendar_*.pkl
│       └── E4_Temporal_Weather_Calendar_*.pkl
│
├── monitoring_data/
│   ├── drift_history.csv
│   ├── parking_observations.csv
│   └── parking_observations_backup.csv
│
└── outputs/
    ├── ablation_model_comparison.csv
    ├── contextual_ablation_random_forest.csv
    └── actual_vs_predicted.png
```

## ▶️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Smart-Parking-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Smart-Parking-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download Git LFS Model Files

```bash
git lfs install
git lfs pull
```

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 Future Enhancements

- Integration with live parking sensors.
- Real-time weather data integration.
- Deployment on a cloud platform.
- Improved prediction through additional historical data.
- Advanced statistical concept drift detection.
- Automated model retraining and evaluation.

## 👨‍💻 Project Information

**Project Title:** Smart Parking Occupancy Prediction Using Machine Learning

**Domain:** Artificial Intelligence and Machine Learning

**Application:** Streamlit Web Dashboard

**Primary Model:** Random Forest Regressor

**Research Focus:** Temporal, Weather and Calendar-Based Parking Occupancy Prediction

---

⭐ If you find this project useful, consider giving the repository a star!