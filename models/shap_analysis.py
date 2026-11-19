
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATA_PATH = "dataset/holiday_enhanced_parking.csv"
MODEL_PATH = "models/enhanced_best_model.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])

df = df.sort_values(
    ["LastUpdated", "SystemCodeNumber"]
).reset_index(drop=True)

# ==========================================
# 3. CREATE TEMPORAL FEATURES
# ==========================================

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

# ==========================================
# 4. SELECT MODEL FEATURES
# ==========================================

features = [
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
    "Month_cos"
]

# ==========================================
# 5. CHRONOLOGICAL TRAINING DATA ONLY
# ==========================================

timestamps = np.sort(df["LastUpdated"].unique())

split_index = int(len(timestamps) * 0.8)

split_time = timestamps[split_index]

train_df = df[df["LastUpdated"] < split_time].copy()

# Sample training records for SHAP
sample_size = min(300, len(train_df))

X_sample = train_df[features].sample(
    n=sample_size,
    random_state=42
)

print("Training sample size:", len(X_sample))

# ==========================================
# 6. LOAD TRAINED PIPELINE
# ==========================================

print("\nLoading trained model...")

pipeline = joblib.load(MODEL_PATH)

if not isinstance(pipeline, Pipeline):
    raise TypeError(
        "Expected a Scikit-learn Pipeline. "
        "Check the saved model."
    )

print("\nPipeline structure:")

for name, step in pipeline.steps:
    print(name, ":", type(step).__name__)

# ==========================================
# 7. SEPARATE PREPROCESSOR AND RANDOM FOREST
# ==========================================

preprocessor = Pipeline(pipeline.steps[:-1])

random_forest = pipeline.steps[-1][1]

print("\nFinal estimator:", type(random_forest).__name__)

if not hasattr(random_forest, "estimators_"):
    raise TypeError(
        "The final pipeline step does not appear "
        "to be a fitted Random Forest."
    )

# ==========================================
# 8. TRANSFORM DATA
# ==========================================

print("\nApplying preprocessing...")

X_transformed = preprocessor.transform(X_sample)

# Convert sparse matrix into dense array
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

# Get transformed feature names
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

print("Transformed features:", X_transformed.shape[1])

# ==========================================
# 9. SHAP TREE EXPLAINER
# ==========================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(random_forest)

shap_values = explainer.shap_values(X_transformed)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_values = shap_values[0]

if hasattr(shap_values, "values"):
    shap_values = shap_values.values

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

print("SHAP analysis generated successfully.")

# ==========================================
# 10. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    show=False,
    max_display=15
)

plt.title(
    "SHAP Summary - Smart Parking Prediction",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 11. SHAP FEATURE IMPORTANCE
# ==========================================

print("Generating feature importance plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_transformed,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title(
    "SHAP Feature Importance - Smart Parking",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 12. SAVE FEATURE IMPORTANCE DATA
# ==========================================

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_transformed.columns,
    "Mean_SHAP_Importance": importance
})

importance_df = importance_df.sort_values(
    by="Mean_SHAP_Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "shap_importance.csv"),
    index=False
)

# ==========================================
# 13. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 50)
print("SHAP ANALYSIS COMPLETED!")
print("=" * 50)

print("\nTop 10 Important Features:")

print(importance_df.head(10).to_string(index=False))

print("\nGenerated Files:")

print("1. outputs/shap_summary.png")
print("2. outputs/shap_feature_importance.png")
print("3. outputs/shap_importance.csv")

print("\nAll operations completed successfully!")
