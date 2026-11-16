
import requests
import pandas as pd
from pathlib import Path
from pyproj import Transformer

URL = (
    "https://services1.arcgis.com/vBu1s2ZA7rxzmHBD/"
    "ArcGIS/rest/services/"
    "Car_Park__CPZ_and_Railways_WFL1/FeatureServer/2/query"
)

params = {
    "where": "1=1",
    "outFields": "ID,LABEL,POSTCODE",
    "returnGeometry": "true",
    "outSR": "27700",
    "f": "json",
    "resultRecordCount": 2000
}

print("Downloading Birmingham car park boundaries...")

response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()
data = response.json()

if "error" in data:
    raise RuntimeError(data["error"])

features = data.get("features", [])

if not features:
    raise RuntimeError("No locations returned.")

transformer = Transformer.from_crs(
    "EPSG:27700", "EPSG:4326", always_xy=True
)

rows = []

for feature in features:
    attrs = feature.get("attributes", {})
    rings = (feature.get("geometry") or {}).get("rings", [])

    if not rings:
        continue

    # Calculate an approximate centre from the outer boundary.
    ring = rings[0]
    x = sum(point[0] for point in ring[:-1]) / max(len(ring) - 1, 1)
    y = sum(point[1] for point in ring[:-1]) / max(len(ring) - 1, 1)

    longitude, latitude = transformer.transform(x, y)

    rows.append({
        "Name": attrs.get("LABEL"),
        "ID": attrs.get("ID"),
        "Postcode": attrs.get("POSTCODE"),
        "Latitude": latitude,
        "Longitude": longitude
    })

result = pd.DataFrame(rows)

output_path = (
    Path(__file__).resolve().parent
    / "dataset"
    / "official_birmingham_locations.csv"
)

result.to_csv(output_path, index=False)

print("\nCompleted!")
print("Locations:", len(result))
print("Saved:", output_path)
print(result.to_string(index=False))
