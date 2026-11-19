
import json
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MONITORING_DIR = BASE_DIR / "monitoring_data"

OBSERVATIONS_FILE = MONITORING_DIR / "parking_observations.csv"

DRIFT_HISTORY_FILE = MONITORING_DIR / "drift_history.csv"


# --------------------------------------------------
# CSV COLUMNS
# --------------------------------------------------

OBSERVATION_COLUMNS = [
    "Timestamp",
    "SystemCodeNumber",
    "Predicted_Occupancy",
    "Actual_Occupancy",
    "Features_JSON"
]

DRIFT_COLUMNS = [
    "Timestamp",
    "Baseline_MAE",
    "Recent_MAE",
    "Records",
    "Status"
]


# --------------------------------------------------
# INITIALIZE FILES
# --------------------------------------------------

def initialize_files():

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    if not OBSERVATIONS_FILE.exists():

        pd.DataFrame(
            columns=OBSERVATION_COLUMNS
        ).to_csv(OBSERVATIONS_FILE, index=False)

    if not DRIFT_HISTORY_FILE.exists():

        pd.DataFrame(
            columns=DRIFT_COLUMNS
        ).to_csv(DRIFT_HISTORY_FILE, index=False)


# --------------------------------------------------
# SAVE OBSERVATION
# --------------------------------------------------

def save_observation(
    parking_id,
    predicted_occupancy,
    actual_occupancy,
    feature_values=None
):

    initialize_files()

    values = [
        predicted_occupancy,
        actual_occupancy
    ]

    if not np.isfinite(values).all():

        raise ValueError("Occupancy values must be finite.")

    if predicted_occupancy < 0 or actual_occupancy < 0:

        raise ValueError("Occupancy cannot be negative.")

    # Store the exact model inputs used for prediction.
    if feature_values is None:
        feature_values = {}

    if not isinstance(feature_values, dict):
        raise ValueError("feature_values must be a dictionary.")

    features_json = json.dumps(
        feature_values,
        default=str
    )

    new_record = pd.DataFrame([{

        "Timestamp": datetime.now().isoformat(),

        "SystemCodeNumber": str(parking_id),

        "Predicted_Occupancy": float(predicted_occupancy),

        "Actual_Occupancy": float(actual_occupancy),

        "Features_JSON": features_json

    }])

    new_record.to_csv(
        OBSERVATIONS_FILE,
        mode="a",
        header=False,
        index=False
    )

    return True


# --------------------------------------------------
# DETECT PERFORMANCE DEGRADATION
# --------------------------------------------------

def detect_drift(
    window_size=50,
    threshold=1.5
):

    initialize_files()

    data = pd.read_csv(OBSERVATIONS_FILE)

    if data.empty:

        return {
            "status": "Insufficient Data",
            "baseline_mae": None,
            "recent_mae": None,
            "drift_detected": False,
            "records": 0
        }

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        errors="coerce"
    )

    data["Predicted_Occupancy"] = pd.to_numeric(
        data["Predicted_Occupancy"],
        errors="coerce"
    )

    data["Actual_Occupancy"] = pd.to_numeric(
        data["Actual_Occupancy"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Timestamp",
            "Predicted_Occupancy",
            "Actual_Occupancy"
        ]
    )

    data = data.sort_values("Timestamp")

    required_records = window_size * 2

    if len(data) < required_records:

        return {
            "status": "Insufficient Data",
            "baseline_mae": None,
            "recent_mae": None,
            "drift_detected": False,
            "records": len(data)
        }

    data["Absolute_Error"] = np.abs(
        data["Actual_Occupancy"]
        - data["Predicted_Occupancy"]
    )

    # Reference performance window
    baseline_errors = (
        data["Absolute_Error"].iloc[:window_size]
    )

    # Latest performance window
    recent_errors = (
        data["Absolute_Error"].iloc[-window_size:]
    )

    baseline_mae = float(baseline_errors.mean())

    recent_mae = float(recent_errors.mean())

    if baseline_mae == 0:

        drift_detected = recent_mae > 0

    else:

        drift_detected = (
            recent_mae > baseline_mae * threshold
        )

    status = (
        "Potential Performance Degradation"
        if drift_detected
        else "Stable"
    )

    result = {
        "status": status,
        "baseline_mae": baseline_mae,
        "recent_mae": recent_mae,
        "drift_detected": drift_detected,
        "records": len(data)
    }

    # Save monitoring history
    history_record = pd.DataFrame([{

        "Timestamp": datetime.now().isoformat(),

        "Baseline_MAE": baseline_mae,

        "Recent_MAE": recent_mae,

        "Records": len(data),

        "Status": status

    }])

    history_record.to_csv(
        DRIFT_HISTORY_FILE,
        mode="a",
        header=False,
        index=False
    )

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    initialize_files()

    print("\nCONCEPT DRIFT MONITORING")
    print("=" * 40)

    result = detect_drift()

    print("Status:", result["status"])
    print("Records:", result["records"])
    print("Baseline MAE:", result["baseline_mae"])
    print("Recent MAE:", result["recent_mae"])
    print("Potential degradation:", result["drift_detected"])

    print("\nMonitoring files initialized.")
