import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# SMART PARKING: CONTROLLED CONTEXTUAL-FEATURE ABLATION STUDY
# ============================================================
# Experiments:
# E1: Temporal only
# E2: Temporal + Weather
# E3: Temporal + Calendar context
# E4: Temporal + Weather + Calendar context
#
# IMPORTANT:
# - All experiments use the same chronological train/test split.
# - All models use the same settings across experiments.
# - Calendar flags are treated as supplied proxy features, not
#   proof of actual public holidays, events, or exam schedules.
# - No local-event or exam feature is fabricated.
# - The app's existing enhanced_best_model.pkl is kept compatible
#   with its current temporal-only prediction inputs by saving the
#   Temporal_Only Random Forest there.
# ============================================================

DATA_PATH = "dataset/holiday_enhanced_parking.csv"
EXPERIMENT_DIR = "models/experiments"
OUTPUT_DIR = "outputs"

TARGET = "Occupancy"
TIMESTAMP = "LastUpdated"

# ----------------------------
# 1. LOAD AND VALIDATE DATA
# ----------------------------

print("\nLoading holiday-enhanced parking dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_base_columns = [
    "SystemCodeNumber",
    "Capacity",
    "Occupancy",
    "LastUpdated",
    "Hour",
    "Minute",
    "Day",
    "Month",
    "Weekday",
    "IsWeekend",
    "Previous_Occupancy"
]

weather_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m"
]

calendar_features = [
    "IsSchoolHoliday",
    "IsHalfTerm",
    "IsChristmasPeriod"
]

required_columns = (
    required_base_columns
    + weather_features
    + calendar_features
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns in dataset: "
        + ", ".join(missing_columns)
    )

df[TIMESTAMP] = pd.to_datetime(
    df[TIMESTAMP],
    errors="coerce"
)

df = df.dropna(
    subset=[TIMESTAMP, TARGET, "Capacity"]
).copy()

df = df.sort_values(
    [TIMESTAMP, "SystemCodeNumber"]
).reset_index(drop=True)

# Check target and capacity values.
invalid_target_rows = (
    (df[TARGET] < 0)
    | (df[TARGET] > df["Capacity"])
)

if invalid_target_rows.any():
    print(
        "Warning:",
        int(invalid_target_rows.sum()),
        "rows have occupancy outside [0, capacity]. Removing them."
    )
    df = df.loc[~invalid_target_rows].copy()

# Ensure calendar flags are numeric 0/1 values.
for col in calendar_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[col].isna().any():
        raise ValueError(
            f"{col} contains missing/non-numeric values. "
            "Review the contextual dataset before training."
        )

    unexpected = set(df[col].unique()) - {0, 1}
    if unexpected:
        raise ValueError(
            f"{col} must contain only 0/1 values. "
            f"Unexpected values: {unexpected}"
        )

print("Dataset shape after validation:", df.shape)
print("Date range:", df[TIMESTAMP].min(), "to", df[TIMESTAMP].max())

# ----------------------------
# 2. CREATE CYCLICAL FEATURES
# ----------------------------

print("\nCreating cyclical temporal features...")

df["Time_sin"] = np.sin(
    2 * np.pi *
    (df["Hour"] * 60 + df["Minute"]) / 1440
)

df["Time_cos"] = np.cos(
    2 * np.pi *
    (df["Hour"] * 60 + df["Minute"]) / 1440
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

# ----------------------------
# 3. DEFINE ABLATION GROUPS
# ----------------------------

temporal_features = [
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

experiments = {
    "E1_Temporal_Only": temporal_features,

    "E2_Temporal_Weather": (
        temporal_features + weather_features
    ),

    "E3_Temporal_Calendar": (
        temporal_features + calendar_features
    ),

    "E4_Temporal_Weather_Calendar": (
        temporal_features
        + weather_features
        + calendar_features
    )
}

print("\nAblation experiments:")
for name, features in experiments.items():
    print(f"{name}: {len(features)} input features")

# ----------------------------
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# ----------------------------

print("\nCreating one shared chronological train/test split...")

unique_times = np.sort(
    df[TIMESTAMP].dropna().unique()
)

if len(unique_times) < 2:
    raise ValueError(
        "Not enough unique timestamps for chronological splitting."
    )

split_index = int(len(unique_times) * 0.80)
split_index = min(max(split_index, 1), len(unique_times) - 1)

split_time = unique_times[split_index]

train_df = df[
    df[TIMESTAMP] < split_time
].copy()

test_df = df[
    df[TIMESTAMP] >= split_time
].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Chronological split produced an empty partition.")

print("Split timestamp:", split_time)
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Training end:", train_df[TIMESTAMP].max())
print("Testing start:", test_df[TIMESTAMP].min())

assert (
    train_df[TIMESTAMP].max()
    < test_df[TIMESTAMP].min()
), "Chronological split overlap detected."

print("Chronological split verified.")

# ----------------------------
# 5. DEFINE MODEL CONFIGURATIONS
# ----------------------------

model_templates = {
    "Linear_Regression": LinearRegression(),

    "Decision_Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random_Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# ----------------------------
# 6. PREPROCESSING PIPELINE
# ----------------------------

def create_pipeline(features, estimator):
    categorical_features = ["SystemCodeNumber"]

    if "weather_code" in features:
        categorical_features.append("weather_code")

    numerical_features = [
        col for col in features
        if col not in categorical_features
    ]

    transformers = [
        (
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clone(estimator))
    ])

# ----------------------------
# 7. CREATE OUTPUT DIRECTORIES
# ----------------------------

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 8. TRAIN ALL EXPERIMENTS
# ----------------------------

results = []
saved_model_paths = {}

y_train = train_df[TARGET]
y_test = test_df[TARGET]
test_capacity = test_df["Capacity"].to_numpy()

for experiment_name, features in experiments.items():

    print("\n" + "=" * 72)
    print("EXPERIMENT:", experiment_name)
    print("=" * 72)

    X_train = train_df[features]
    X_test = test_df[features]

    for model_name, estimator in model_templates.items():

        print("\nTraining:", model_name)

        pipeline = create_pipeline(
            features,
            estimator
        )

        pipeline.fit(X_train, y_train)

        raw_predictions = pipeline.predict(X_test)

        # Occupancy cannot be negative or exceed the lot capacity.
        predictions = np.clip(
            raw_predictions,
            0,
            test_capacity
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        model_path = os.path.join(
            EXPERIMENT_DIR,
            f"{experiment_name}_{model_name}.pkl"
        )

        joblib.dump(pipeline, model_path)

        saved_model_paths[
            (experiment_name, model_name)
        ] = model_path

        results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Feature_Count": len(features),
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "Model_Path": model_path
        })

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 6))
        print("Saved:", model_path)

# ----------------------------
# 9. SAVE FULL MODEL COMPARISON
# ----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "ablation_model_comparison.csv"),
    index=False
)

print("\n" + "=" * 80)
print("FULL ABLATION RESULTS: ALL MODELS")
print("=" * 80)

print(
    results_df.sort_values(
        ["Model", "MAE"]
    ).to_string(index=False)
)

# ----------------------------
# 10. RANDOM FOREST CONTEXTUAL ABLATION
# ----------------------------

rf_results = results_df[
    results_df["Model"] == "Random_Forest"
].copy()

baseline_row = rf_results[
    rf_results["Experiment"] == "E1_Temporal_Only"
]

if baseline_row.empty:
    raise RuntimeError("Random Forest temporal baseline is missing.")

baseline_mae = float(baseline_row.iloc[0]["MAE"])
baseline_rmse = float(baseline_row.iloc[0]["RMSE"])

rf_results["MAE_Change_vs_Baseline"] = (
    rf_results["MAE"] - baseline_mae
)

rf_results["RMSE_Change_vs_Baseline"] = (
    rf_results["RMSE"] - baseline_rmse
)

rf_results["MAE_Improvement_Percent"] = (
    (baseline_mae - rf_results["MAE"])
    / baseline_mae
) * 100

rf_results["RMSE_Improvement_Percent"] = (
    (baseline_rmse - rf_results["RMSE"])
    / baseline_rmse
) * 100

rf_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "contextual_ablation_random_forest.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("RANDOM FOREST CONTEXTUAL ABLATION")
print("=" * 80)

display_columns = [
    "Experiment",
    "MAE",
    "RMSE",
    "R2 Score",
    "MAE_Improvement_Percent",
    "RMSE_Improvement_Percent"
]

print(
    rf_results[display_columns]
    .round(4)
    .to_string(index=False)
)

# ----------------------------
# 11. CREATE ABLATION COMPARISON PLOTS
# ----------------------------

plot_df = rf_results.sort_values("Experiment").copy()

plt.figure(figsize=(11, 6))
plt.bar(plot_df["Experiment"], plot_df["MAE"])
plt.title("Random Forest: MAE Across Contextual Ablation Experiments")
plt.xlabel("Experiment")
plt.ylabel("MAE (parking spaces)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "contextual_ablation_mae.png"),
    dpi=300
)
plt.close()

plt.figure(figsize=(11, 6))
plt.bar(plot_df["Experiment"], plot_df["RMSE"])
plt.title("Random Forest: RMSE Across Contextual Ablation Experiments")
plt.xlabel("Experiment")
plt.ylabel("RMSE (parking spaces)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "contextual_ablation_rmse.png"),
    dpi=300
)
plt.close()

# ----------------------------
# 12. SAVE APP-COMPATIBLE MODEL
# ----------------------------
# The existing Streamlit prediction page currently supplies the
# temporal feature set. Therefore, keep enhanced_best_model.pkl
# as the temporal-only Random Forest so the current app does not
# unexpectedly require weather/calendar inputs.
#
# Contextual models remain saved separately under models/experiments.
# After the app's prediction pipeline is updated to supply contextual
# values, a contextual model can be selected for deployment.

deployment_key = (
    "E1_Temporal_Only",
    "Random_Forest"
)

deployment_model_path = saved_model_paths[deployment_key]
deployment_model = joblib.load(deployment_model_path)

joblib.dump(
    deployment_model,
    "models/enhanced_best_model.pkl"
)

with open(
    "models/enhanced_best_model_name.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "E1_Temporal_Only - Random_Forest "
        "(app-compatible temporal baseline)"
    )

# Also save the best measured experiment for research reference.
best_research_row = results_df.loc[
    results_df["MAE"].idxmin()
]

best_research_summary = {
    "experiment": str(best_research_row["Experiment"]),
    "model": str(best_research_row["Model"]),
    "mae": float(best_research_row["MAE"]),
    "rmse": float(best_research_row["RMSE"]),
    "r2_score": float(best_research_row["R2 Score"]),
    "model_path": str(best_research_row["Model_Path"])
}

joblib.dump(
    best_research_summary,
    os.path.join(OUTPUT_DIR, "best_research_experiment.pkl")
)

# ----------------------------
# 13. FINAL SUMMARY
# ----------------------------

print("\n" + "=" * 80)
print("TRAINING AND ABLATION STUDY COMPLETED")
print("=" * 80)

print("\nBest measured experiment by MAE:")
print("Experiment:", best_research_summary["experiment"])
print("Model:", best_research_summary["model"])
print("MAE:", round(best_research_summary["mae"], 4))
print("RMSE:", round(best_research_summary["rmse"], 4))
print("R2 Score:", round(best_research_summary["r2_score"], 6))

print("\nApp-compatible model:")
print("E1_Temporal_Only - Random_Forest")
print("Saved to: models/enhanced_best_model.pkl")

print("\nGenerated outputs:")
print("- outputs/ablation_model_comparison.csv")
print("- outputs/contextual_ablation_random_forest.csv")
print("- outputs/contextual_ablation_mae.png")
print("- outputs/contextual_ablation_rmse.png")
print("- outputs/best_research_experiment.pkl")

print("\nAll experiment pipelines:")
print("- models/experiments/")

print(
    "\nResearch note: calendar columns are supplied proxy flags. "
    "Do not describe them as verified public holidays, local events, "
    "or exam schedules unless independently validated."
)
