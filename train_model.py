
import os
import joblib
import numpy as np
import pandas as pd

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

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading cleaned parking dataset...")

df = pd.read_csv("dataset/cleaned_parking.csv")

df["LastUpdated"] = pd.to_datetime(
    df["LastUpdated"],
    errors="coerce"
)

df = df.dropna(subset=["LastUpdated"])

df = df.sort_values("LastUpdated").reset_index(drop=True)

# ==========================================
# 2. CREATE CYCLICAL TIME FEATURES
# ==========================================

print("Creating time features...")

# Time of day
df["Time_sin"] = np.sin(
    2 * np.pi * (df["Hour"] * 60 + df["Minute"]) / 1440
)

df["Time_cos"] = np.cos(
    2 * np.pi * (df["Hour"] * 60 + df["Minute"]) / 1440
)

# Day of week
df["Weekday_sin"] = np.sin(
    2 * np.pi * df["Weekday"] / 7
)

df["Weekday_cos"] = np.cos(
    2 * np.pi * df["Weekday"] / 7
)

# Month of year
df["Month_sin"] = np.sin(
    2 * np.pi * (df["Month"] - 1) / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * (df["Month"] - 1) / 12
)

# ==========================================
# 3. FEATURES AND TARGET
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

X = df[features].copy()
y = df["Occupancy"].copy()

# ==========================================
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 5. PREPROCESSING
# ==========================================

categorical_features = ["SystemCodeNumber"]

numerical_features = [
    col for col in features
    if col != "SystemCodeNumber"
]

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "location",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numbers",
                SimpleImputer(strategy="median"),
                numerical_features
            )
        ]
    )

# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []
trained_models = {}

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", algorithm)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    predictions = np.clip(
        predictions,
        0,
        X_test["Capacity"].to_numpy()
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# ==========================================
# 10. SAVE MODEL AND RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_parking_model.pkl"
)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

with open(
    "models/best_model_name.txt",
    "w"
) as file:
    file.write(best_model_name)

print("\nModel saved successfully.")
print("Comparison saved successfully.")
print("\nMODEL TRAINING COMPLETED!")
