
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset/holiday_enhanced_parking.csv")
df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])

# Sort chronologically
df = df.sort_values(["LastUpdated", "SystemCodeNumber"])

# Create the same chronological split
timestamps = np.sort(df["LastUpdated"].unique())
split_index = int(len(timestamps) * 0.8)
split_time = timestamps[split_index]

test = df[df["LastUpdated"] >= split_time].copy()

# Create temporal features
test["Time_sin"] = np.sin(2 * np.pi * (test["Hour"] * 60 + test["Minute"]) / 1440)
test["Time_cos"] = np.cos(2 * np.pi * (test["Hour"] * 60 + test["Minute"]) / 1440)
test["Weekday_sin"] = np.sin(2 * np.pi * test["Weekday"] / 7)
test["Weekday_cos"] = np.cos(2 * np.pi * test["Weekday"] / 7)
test["Month_sin"] = np.sin(2 * np.pi * (test["Month"] - 1) / 12)
test["Month_cos"] = np.cos(2 * np.pi * (test["Month"] - 1) / 12)

# Load best model
model = joblib.load("models/enhanced_best_model.pkl")

# Temporal-only features
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

X_test = test[features]
y_test = test["Occupancy"]

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL VALIDATION RESULTS")
print("------------------------")
print("Test records:", len(test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Actual vs predicted graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel("Actual Occupancy")
plt.ylabel("Predicted Occupancy")
plt.title("Actual vs Predicted Parking Occupancy")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()
