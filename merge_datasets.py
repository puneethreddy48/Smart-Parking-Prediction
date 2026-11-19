
import pandas as pd

# Load parking dataset
parking = pd.read_csv("dataset/cleaned_parking.csv")

# Load weather dataset
weather = pd.read_csv("dataset/historical_weather_birmingham.csv")

# Convert timestamps
parking["LastUpdated"] = pd.to_datetime(parking["LastUpdated"])
weather["time"] = pd.to_datetime(weather["time"])

# Create hourly matching timestamps
parking["WeatherTime"] = parking["LastUpdated"].dt.floor("h")

# Check weather timestamp uniqueness
if weather["time"].duplicated().any():
    raise ValueError("Duplicate weather timestamps found!")

# Merge parking and weather data
enhanced = parking.merge(
    weather,
    left_on="WeatherTime",
    right_on="time",
    how="left",
    validate="many_to_one",
    indicator=True
)

# Check matching results
print("PARKING RECORDS:", len(parking))
print("ENHANCED RECORDS:", len(enhanced))

print("\nWEATHER MATCH STATUS:")
print(enhanced["_merge"].value_counts())

# Remove temporary merge columns
enhanced.drop(
    columns=["WeatherTime", "time", "_merge"],
    inplace=True
)

# Check missing values
print("\nMISSING VALUES:")
print(enhanced.isnull().sum())

# Save enhanced dataset
enhanced.to_csv(
    "dataset/enhanced_parking.csv",
    index=False
)

print("\nFINAL DATASET SHAPE:", enhanced.shape)
print("\nFIRST FIVE RECORDS:")
print(enhanced.head())

print("\nEnhanced dataset saved successfully!")
