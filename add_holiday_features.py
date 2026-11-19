
import pandas as pd

# Load enhanced dataset
df = pd.read_csv("dataset/enhanced_parking.csv")

# Convert timestamp
df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
df["Date"] = df["LastUpdated"].dt.date

# Default values
df["IsSchoolHoliday"] = 0
df["IsChristmasPeriod"] = 0
df["IsHalfTerm"] = 0

# Autumn half-term: 24–28 October 2016
half_term_start = pd.Timestamp("2016-10-24").date()
half_term_end = pd.Timestamp("2016-10-28").date()

# Christmas school holiday begins 16 December
christmas_school_start = pd.Timestamp("2016-12-16").date()

# Birmingham Christmas Market: 17 November–29 December
market_start = pd.Timestamp("2016-11-17").date()
market_end = pd.Timestamp("2016-12-29").date()

df.loc[
    df["Date"].between(half_term_start, half_term_end),
    "IsHalfTerm"
] = 1

df.loc[
    df["Date"].between(half_term_start, half_term_end)
    | (df["Date"] >= christmas_school_start),
    "IsSchoolHoliday"
] = 1

df.loc[
    df["Date"].between(market_start, market_end),
    "IsChristmasPeriod"
] = 1

# Remove temporary Date column
df.drop(columns=["Date"], inplace=True)

# Save new dataset
df.to_csv("dataset/holiday_enhanced_parking.csv", index=False)

print("Holiday features added successfully!")
print(df[[
    "IsSchoolHoliday",
    "IsHalfTerm",
    "IsChristmasPeriod"
]].sum())

print("\nSaved: dataset/holiday_enhanced_parking.csv")
