
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "dataset" / "parking_locations.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "parking_locations_demo.csv"

df = pd.read_csv(INPUT_FILE)

# Approximate Birmingham city-centre map area.
# These generated points are for DEMONSTRATION ONLY.
# They are not actual parking facility coordinates.

rng = np.random.default_rng(42)

center_lat = 52.4775
center_lon = -1.8980

# Spread points across a wider area.
lat_radius = 0.011
lon_radius = 0.017

# Preserve existing provisional coordinates.
existing = (
    df["Latitude"].notna()
    & df["Longitude"].notna()
)

df["LocationType"] = "Approximate demo marker - not actual location"

df.loc[existing, "LocationType"] = "Provisional - verify"

# Generate well-spaced demonstration points.
generated_points = []

for index in df.index[~existing]:

    for attempt in range(10000):

        lat = center_lat + rng.uniform(
            -lat_radius, lat_radius
        )

        lon = center_lon + rng.uniform(
            -lon_radius, lon_radius
        )

        # Keep generated points separated from one another.
        sufficiently_far = all(
            np.sqrt(
                (lat - old_lat) ** 2
                + (lon - old_lon) ** 2
            ) >= 0.0025
            for old_lat, old_lon in generated_points
        )

        if sufficiently_far:
            generated_points.append((lat, lon))

            df.loc[index, "Latitude"] = lat
            df.loc[index, "Longitude"] = lon

            break

    else:
        raise RuntimeError(
            "Could not generate enough separated map points."
        )

df.to_csv(OUTPUT_FILE, index=False)

print("\nUpdated demo map successfully!")
print("Total parking IDs:", len(df))
print("Provisional coordinates:", existing.sum())
print("Approximate demo markers:", (~existing).sum())
print("Saved to:", OUTPUT_FILE)

print("\nParking marker details:")
print(
    df[
        [
            "SystemCodeNumber",
            "Capacity",
            "Latitude",
            "Longitude",
            "LocationType"
        ]
    ].to_string(index=False)
)
