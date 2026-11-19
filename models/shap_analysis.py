
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "holiday_enhanced_parking.csv"

# IMPORTANT: Use the exact E4 model loaded by app.py
MODEL_PATH = (
    BASE_DIR / "models" / "experiments" /
    "E4_Temporal_Weather_Calendar_Random_Forest.pkl"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. LOAD DATASET
# =====================================================

print("Loading E4 contextual parking dataset...")

df = pd.read_csv(DATA_PATH)

df["LastUpdated"] = pd.to_datetime(
    df["LastUpdated"], errors="coerce"
)

df = df.dropna(subset=["LastUpdated"])

df = df.sort_values(
    ["LastUpdated", "SystemCodeNumber"]
).reset_index(drop=True)

# =====================================================
# 3. CREATE CYCLICAL TEMPORAL FEATURES
# =====================================================

df["Time_sin"] = np.sin(
    2 * np.pi * (df["Hour"] * 60 + df["Minute"]) / 1440
)

df["Time_cos"] = np.cos(
    2 * np.pi * (df["Hour"] * 60 + df["Minute"]) / 1440
)

df["Weekday_sin"] = np.sin(
    2 * np.pi * df["Weekday"] / 7
)

df["Weekday_cos"] = np.cos(
    2 * np.pi * df["Weekday"] / 7
)

df["Month_sin"] = np.sin(
    2 * np.pi * (df["Month"] - 1) / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * (df["Month"] - 1) / 12
)

# =====================================================
# 4. E4 FEATURES - ALL 23 FEATURES
# =====================================================

features = [
    # Temporal features (15)
    "SystemCodeNumber",
    "Capacity",
    "Hour",
    "Minute",
    "Day",
    "Month",
    "Weekday",
    "IsWeekend",
    "Previous_Occupancy",
    "Time_sin",
    "Time_cos",
    "Weekday_sin",
    "Weekday_cos",
    "Month_sin",
    "Month_cos",

    # Weather features (5)
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",

    # Calendar features (3)
    "IsSchoolHoliday",
    "IsHalfTerm",
    "IsChristmasPeriod"
]

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing E4 features in dataset: {missing}"
    )

print("Total E4 input features:", len(features))

# =====================================================
# 5. CHRONOLOGICAL TRAINING SAMPLE
# =====================================================

timestamps = np.sort(df["LastUpdated"].unique())

split_index = int(len(timestamps) * 0.8)

if split_index >= len(timestamps):
    raise ValueError("Insufficient timestamps for chronological split.")

split_time = timestamps[split_index]

train_df = df[
    df["LastUpdated"] < split_time
].copy()

sample_size = min(300, len(train_df))

if sample_size == 0:
    raise ValueError("No training records available for SHAP.")

X_sample = train_df[features].sample(
    n=sample_size,
    random_state=42
)

print("SHAP training sample:", len(X_sample))

# =====================================================
# 6. LOAD THE EXACT E4 MODEL
# =====================================================

print("\nLoading E4 Random Forest model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"E4 model not found: {MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

if not isinstance(pipeline, Pipeline):
    raise TypeError(
        "The E4 model is not a Scikit-learn Pipeline."
    )

print("Pipeline structure:")

for name, step in pipeline.steps:
    print(name, ":", type(step).__name__)

preprocessor = Pipeline(pipeline.steps[:-1])

random_forest = pipeline.steps[-1][1]

if not hasattr(random_forest, "estimators_"):
    raise TypeError(
        "The final estimator is not a fitted Random Forest."
    )

print("\nE4 Random Forest loaded successfully.")

# =====================================================
# 7. APPLY PREPROCESSING
# =====================================================

print("\nApplying E4 preprocessing...")

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [
        f"Feature_{i}"
        for i in range(X_transformed.shape[1])
    ]

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

print("Transformed feature count:", X_transformed.shape[1])

# =====================================================
# 8. SHAP EXPLAINER
# =====================================================

print("\nCreating E4 SHAP explainer...")

explainer = shap.TreeExplainer(random_forest)

shap_values = explainer.shap_values(X_transformed)

if hasattr(shap_values, "values"):
    shap_values = shap_values.values

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

if shap_values.shape != X_transformed.shape:
    raise ValueError(
        f"SHAP shape {shap_values.shape} does not match "
        f"input shape {X_transformed.shape}."
    )

print("E4 SHAP analysis completed.")

# =====================================================
# 9. GENERATE SHAP SUMMARY PLOT
# =====================================================

print("\nGenerating SHAP summary plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    show=False,
    max_display=15
)

plt.title(
    "SHAP Summary - E4 Smart Parking Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 10. GENERATE SHAP FEATURE IMPORTANCE
# =====================================================

print("Generating SHAP feature importance...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title(
    "SHAP Feature Importance - E4 Smart Parking Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 11. SAVE SHAP IMPORTANCE CSV
# =====================================================

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_transformed.columns,
    "Mean_SHAP_Importance": importance
})

importance_df = importance_df.sort_values(
    "Mean_SHAP_Importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "shap_importance.csv",
    index=False
)

# =====================================================
# 12. FINAL OUTPUT
# =====================================================

print("\n" + "=" * 55)
print("E4 SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 55)

print("\nTop 10 E4 Important Features:")

print(
    importance_df.head(10).to_string(index=False)
)

print("\nGenerated files:")

print(OUTPUT_DIR / "shap_summary.png")
print(OUTPUT_DIR / "shap_feature_importance.png")
print(OUTPUT_DIR / "shap_importance.csv")

print("\nYou can now run your Streamlit application.")
