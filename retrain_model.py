
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "cleaned_parking.csv"
OBSERVATIONS_PATH = (
    BASE_DIR / "monitoring_data" / "parking_observations.csv"
)

MODEL_PATH = BASE_DIR / "models" / "enhanced_best_model.pkl"

CANDIDATE_PATH = BASE_DIR / "models" / "candidate_model.pkl"


# Minimum records required
MINIMUM_NEW_RECORDS = 100


def main():

    print("\nAUTOMATIC MODEL RETRAINING")
    print("=" * 45)

    # Load original dataset
    original_df = pd.read_csv(DATA_PATH)

    # Load new observations
    if not OBSERVATIONS_PATH.exists():

        print("No monitoring observations found.")
        return

    new_df = pd.read_csv(OBSERVATIONS_PATH)

    new_df = new_df.dropna(
        subset=[
            "SystemCodeNumber",
            "Actual_Occupancy"
        ]
    )

    if len(new_df) < MINIMUM_NEW_RECORDS:

        print(
            f"Insufficient new records: {len(new_df)}"
        )

        print(
            f"Required: {MINIMUM_NEW_RECORDS}"
        )

        return

    # Only genuinely observed records should be included.
    new_df["SystemCodeNumber"] = (
        new_df["SystemCodeNumber"].astype(str)
    )

    original_df["SystemCodeNumber"] = (
        original_df["SystemCodeNumber"].astype(str)
    )

    # Retrieve capacity for each parking location
    capacity_lookup = (
        original_df.groupby("SystemCodeNumber")["Capacity"]
        .first()
    )

    new_df["Capacity"] = (
        new_df["SystemCodeNumber"].map(capacity_lookup)
    )

    new_df = new_df.dropna(subset=["Capacity"])

    # Validate actual occupancy
    new_df = new_df[
        (new_df["Actual_Occupancy"] >= 0)
        & (new_df["Actual_Occupancy"] <= new_df["Capacity"])
    ].copy()

    if len(new_df) < MINIMUM_NEW_RECORDS:

        print("Not enough valid observations after validation.")
        return

    # Prepare chronological records
    new_df["Timestamp"] = pd.to_datetime(
        new_df["Timestamp"],
        errors="coerce"
    )

    new_df = new_df.dropna(subset=["Timestamp"])

    new_df = new_df.sort_values("Timestamp")

    # Use the recorded prediction time only as a provisional
    # timestamp until prediction timestamps are stored separately.
    # Do not claim this is a validated temporal feature pipeline.
    print(
        "WARNING: This candidate training process needs "
        "time-aligned feature records before deployment."
    )

    print(
        "Actual occupancy records are available, but the current "
        "monitoring CSV does not store all model input features."
    )

    print(
        "Candidate training is paused to prevent training on "
        "incomplete or misleading features."
    )

    print(
        "Next: extend observation logging to save the full "
        "prediction input row and observation timestamp."
    )


if __name__ == "__main__":

    main()
