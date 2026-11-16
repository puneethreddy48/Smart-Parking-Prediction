# 🅿️ Smart Parking Availability Prediction Using Machine Learning

## 📌 Project Overview

Smart Parking Availability Prediction is a machine learning-based application that predicts parking occupancy and estimates available parking spaces using historical parking data.

The application provides an interactive dashboard built with Streamlit, allowing users to explore parking locations, view predictions, analyze parking patterns, and evaluate model performance.

## 🎯 Objectives

- Predict parking occupancy using machine learning.
- Estimate the number of available parking spaces.
- Visualize parking locations through an interactive map.
- Analyze historical parking usage patterns.
- Compare machine learning models based on their performance.

## ✨ Key Features

- **Parking Overview:** Displays parking capacity and occupancy information.
- **Interactive Parking Map:** Shows parking locations on a map.
- **Parking Prediction:** Predicts occupancy based on selected location, date, and time.
- **Parking Analytics:** Visualizes occupancy trends, hourly patterns, and location-wise usage.
- **Model Performance:** Compares the performance of different machine learning algorithms.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Interactive Map | Folium, Streamlit-Folium |
| Model Storage | Joblib |

## 🤖 Machine Learning Models

The project compares three machine learning algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The best-performing model is selected based on evaluation metrics such as MAE, RMSE, and R².

## 📊 Model Performance

The recorded test results from the current training run:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 65.10 | 142.29 | 0.9616 |
| Decision Tree | 29.80 | 64.57 | 0.9921 |
| Random Forest | 23.24 | 51.24 | 0.9950 |

**Selected Model:** Random Forest Regressor

*Note: These are results from the historical dataset test split, not a guarantee of real-time prediction accuracy.*

## 📁 Project Structure

```text
Smart-Parking-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── cleaned_parking.csv
│   └── parking_locations_demo.csv
│
├── models/
│   ├── best_parking_model.pkl
│   └── best_model_name.txt
│
└── outputs/
    └── model_comparison.csv
```

## ▶️ Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Smart-Parking-Prediction.git
```

**2. Navigate to the project folder**

```bash
cd Smart-Parking-Prediction
```

**3. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
streamlit run app.py
```

## 📌 Dataset Information

The project uses historical parking occupancy records containing parking location identifiers, capacity, occupancy, and update timestamps.

The historical data is used for model training and evaluation. It should not be interpreted as live parking availability.

## 🚀 Deployment

The application is designed for deployment using **Streamlit Community Cloud**.

## 👨‍💻 Project Information

**Project Title:** Smart Parking Availability Prediction Using Machine Learning

**Domain:** Artificial Intelligence and Machine Learning

**Dashboard:** Streamlit

---

⭐ If you find this project useful, consider giving the repository a star!