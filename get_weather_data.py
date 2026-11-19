
import requests
import pandas as pd
from pathlib import Path

LATITUDE = 52.4862
LONGITUDE = -1.8904

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2016-10-04",
    "end_date": "2016-12-19",
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
    "timezone": "Europe/London"
}

print("Downloading weather data...")

try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(data)

    weather = pd.DataFrame(data["hourly"])
    weather["time"] = pd.to_datetime(weather["time"])

    print("\nMissing values before saving:")
    print(weather.isnull().sum())

    output = Path("dataset/historical_weather_birmingham.csv")
    weather.to_csv(output, index=False)

    print("\nDataset shape:", weather.shape)
    print("\nFirst five records:")
    print(weather.head())

    print("\nSaved successfully:", output)

except Exception as e:
    print("Download failed:", e)
