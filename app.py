
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px

from pathlib import Path
from streamlit_folium import st_folium
from datetime import date, time, datetime

from concept_drift import save_observation, detect_drift


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "cleaned_parking.csv"
CONTEXT_DATA_PATH = BASE_DIR / "dataset" / "holiday_enhanced_parking.csv"
MAP_PATH = BASE_DIR / "dataset" / "parking_locations_demo.csv"

MODEL_PATH = (
    BASE_DIR / "models" / "experiments" /
    "E4_Temporal_Weather_Calendar_Random_Forest.pkl"
)

COMPARISON_PATH = (
    BASE_DIR / "outputs" / "ablation_model_comparison.csv"
)

OBSERVATION_PATH = (
    BASE_DIR / "monitoring_data" / "parking_observations.csv"
)

BEST_MODEL_NAME = (
    "E4 Temporal + Weather + Calendar - Random Forest"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #14213d, #123c54);
    border: 1px solid #24445f;
    padding: 34px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 15px;
}

.section-title {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: #151f32;
    border: 1px solid #29364b;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stButton > button {
    background: #0ea5e9;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 650;
}

.stButton > button:hover {
    background: #0284c7;
    color: white;
}

hr {
    border-color: #29364b;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_context_dataset():
    if CONTEXT_DATA_PATH.exists():
        return pd.read_csv(CONTEXT_DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_map_data():
    if MAP_PATH.exists():
        return pd.read_csv(MAP_PATH)
    return pd.DataFrame()


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_dataset()
    context_df = load_context_dataset()
    map_df = load_map_data()
    comparison = load_comparison()
    model = load_model()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.error(
        "Check that the E4 model exists inside models/experiments."
    )
    st.stop()


required_columns = [
    "SystemCodeNumber",
    "Capacity",
    "Occupancy",
    "Hour",
    "Minute",
    "Day",
    "Month",
    "Weekday",
    "IsWeekend",
    "Previous_Occupancy",
    "LastUpdated"
]

missing_columns = [
    c for c in required_columns if c not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


context_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "IsSchoolHoliday",
    "IsHalfTerm",
    "IsChristmasPeriod"
]

if context_df.empty:
    st.error(
        "Contextual dataset not found: "
        "dataset/holiday_enhanced_parking.csv"
    )
    st.stop()

missing_context = [
    c for c in context_features if c not in context_df.columns
]

if missing_context:
    st.error(
        f"Missing contextual dataset columns: {missing_context}"
    )
    st.stop()


df["SystemCodeNumber"] = df["SystemCodeNumber"].astype(str)

parking_options = sorted(
    df["SystemCodeNumber"].dropna().unique().tolist()
)

total_locations = len(parking_options)


# =========================================================
# WEATHER DEFAULTS AND WEATHER CODE OPTIONS
# =========================================================

weather_numeric_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]

weather_defaults = {}

for column in weather_numeric_features:
    values = pd.to_numeric(
        context_df[column], errors="coerce"
    ).dropna()

    if values.empty:
        weather_defaults[column] = 0.0
    else:
        weather_defaults[column] = float(values.median())


weather_codes = pd.to_numeric(
    context_df["weather_code"],
    errors="coerce"
).dropna().unique().tolist()

weather_codes = sorted(
    [float(x) for x in weather_codes]
)

if not weather_codes:
    weather_codes = [0.0]

calendar_defaults = {}

for column in [
    "IsSchoolHoliday",
    "IsHalfTerm",
    "IsChristmasPeriod"
]:
    values = pd.to_numeric(
        context_df[column], errors="coerce"
    ).dropna()

    if values.empty:
        calendar_defaults[column] = 0
    else:
        calendar_defaults[column] = int(
            values.mode().iloc[0]
        )


# =========================================================
# OVERVIEW DATA
# =========================================================

latest_by_location = (
    df.assign(
        LastUpdated=pd.to_datetime(
            df["LastUpdated"], errors="coerce"
        )
    )
    .dropna(subset=["LastUpdated"])
    .sort_values("LastUpdated")
    .groupby("SystemCodeNumber")
    .tail(1)
)

total_capacity = int(latest_by_location["Capacity"].sum())
latest_occupancy = int(latest_by_location["Occupancy"].sum())
latest_available = max(total_capacity - latest_occupancy, 0)


# =========================================================
# CONTEXT INPUT FUNCTION
# =========================================================

def context_input_section(key_prefix="prediction"):

    st.subheader("🌦️ Weather Information")

    st.caption(
        "Enter the expected weather conditions for the selected "
        "prediction date and time. These values are model inputs."
    )

    w1, w2, w3 = st.columns(3)

    with w1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=float(weather_defaults["temperature_2m"]),
            step=0.5,
            key=f"{key_prefix}_temperature"
        )

    with w2:
        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(weather_defaults["relative_humidity_2m"]),
            step=1.0,
            key=f"{key_prefix}_humidity"
        )

    with w3:
        rainfall = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=500.0,
            value=max(
                0.0,
                float(weather_defaults["precipitation"])
            ),
            step=0.1,
            key=f"{key_prefix}_rainfall"
        )

    w4, w5 = st.columns(2)

    with w4:
        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=max(
                0.0,
                float(weather_defaults["wind_speed_10m"])
            ),
            step=0.5,
            key=f"{key_prefix}_wind"
        )

    with w5:
        selected_weather_code = st.selectbox(
            "Weather Code",
            weather_codes,
            index=0,
            format_func=lambda x: str(int(x))
            if float(x).is_integer() else str(x),
            key=f"{key_prefix}_weather_code"
        )

    st.caption(
        "Select a weather code present in the training dataset. "
        "The code is treated as a categorical feature by the model."
    )

    st.divider()

    st.subheader("📅 Calendar Information")

    st.caption(
        "These are the three calendar features used in E4."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        school_holiday = st.checkbox(
            "School Holiday",
            value=bool(calendar_defaults["IsSchoolHoliday"]),
            key=f"{key_prefix}_school"
        )

    with c2:
        half_term = st.checkbox(
            "Half-Term Holiday",
            value=bool(calendar_defaults["IsHalfTerm"]),
            key=f"{key_prefix}_half"
        )

    with c3:
        christmas = st.checkbox(
            "Christmas Period",
            value=bool(calendar_defaults["IsChristmasPeriod"]),
            key=f"{key_prefix}_christmas"
        )

    return {
        "temperature_2m": float(temperature),
        "relative_humidity_2m": float(humidity),
        "precipitation": float(rainfall),
        "weather_code": float(selected_weather_code),
        "wind_speed_10m": float(wind_speed),
        "IsSchoolHoliday": int(school_holiday),
        "IsHalfTerm": int(half_term),
        "IsChristmasPeriod": int(christmas)
    }


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def make_prediction(
    parking_id,
    selected_date,
    selected_time,
    previous_occupancy,
    context_values
):

    parking_id = str(parking_id)

    location_data = df[
        df["SystemCodeNumber"] == parking_id
    ]

    if location_data.empty:
        raise ValueError("Parking location not found.")

    capacity = float(location_data["Capacity"].iloc[0])
    previous_occupancy = float(previous_occupancy)

    if not 0 <= previous_occupancy <= capacity:
        raise ValueError(
            "Occupancy must be between 0 and capacity."
        )

    hour = selected_time.hour
    minute = selected_time.minute
    day = selected_date.day
    month = selected_date.month
    weekday = selected_date.weekday()

    is_weekend = 1 if weekday >= 5 else 0

    minute_of_day = hour * 60 + minute

    # E4 temporal features
    input_features = {
        "SystemCodeNumber": parking_id,
        "Capacity": capacity,
        "Hour": hour,
        "Minute": minute,
        "Day": day,
        "Month": month,
        "Weekday": weekday,
        "IsWeekend": is_weekend,
        "Previous_Occupancy": previous_occupancy,

        "Time_sin": np.sin(
            2 * np.pi * minute_of_day / 1440
        ),

        "Time_cos": np.cos(
            2 * np.pi * minute_of_day / 1440
        ),

        "Weekday_sin": np.sin(
            2 * np.pi * weekday / 7
        ),

        "Weekday_cos": np.cos(
            2 * np.pi * weekday / 7
        ),

        "Month_sin": np.sin(
            2 * np.pi * (month - 1) / 12
        ),

        "Month_cos": np.cos(
            2 * np.pi * (month - 1) / 12
        )
    }

    # Add all E4 weather and calendar features
    input_features.update(context_values)

    # Preserve the exact feature names used in training
    input_data = pd.DataFrame([input_features])

    prediction = float(model.predict(input_data)[0])

    predicted_occupancy = int(
        np.clip(round(prediction), 0, capacity)
    )

    available = max(
        int(capacity) - predicted_occupancy, 0
    )

    occupancy_percentage = (
        predicted_occupancy / capacity * 100
        if capacity > 0 else 0
    )

    return {
        "capacity": int(capacity),
        "occupancy": predicted_occupancy,
        "available": available,
        "occupancy_percentage": occupancy_percentage,
        "features": input_features
    }


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🅿️ Smart Parking AI</h1>
    <p>
        Intelligent Parking Availability Prediction System
        | Powered by Machine Learning
    </p>
    <p>
        Context-aware parking prediction using temporal,
        weather and calendar information.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

nav_items = [
    "Overview",
    "Parking Map",
    "Prediction",
    "Decision Support",
    "Analytics",
    "Model Performance",
    "Explainable AI",
    "Concept Drift"
]

nav_cols = st.columns(4)

for i, item in enumerate(nav_items):
    with nav_cols[i % 4]:
        anchor = item.lower().replace(" ", "-")

        st.markdown(
            f'<a href="#{anchor}" '
            f'style="display:block;text-align:center;'
            f'padding:11px 4px;background:#17243a;'
            f'border:1px solid #29364b;border-radius:9px;'
            f'color:#7dd3fc;text-decoration:none;'
            f'font-size:12px;font-weight:600;margin-bottom:8px;">'
            f'{item}</a>',
            unsafe_allow_html=True
        )


# =========================================================
# OVERVIEW
# =========================================================

st.markdown(
    '<div id="overview" class="section-title">'
    '📊 Project Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Parking Locations", total_locations)
c2.metric("Total Capacity", f"{total_capacity:,}")
c3.metric("Historical Occupancy", f"{latest_occupancy:,}")
c4.metric("Available Spaces", f"{latest_available:,}")

st.caption(
    "Overview values represent the latest historical snapshot "
    "for each location, not live sensor readings."
)


# =========================================================
# PARKING MAP
# =========================================================

st.markdown(
    '<div id="parking-map" class="section-title">'
    '📍 Interactive Parking Map</div>',
    unsafe_allow_html=True
)

if not map_df.empty:

    latitude_col = next(
        (
            c for c in ["Latitude", "latitude", "Lat", "lat"]
            if c in map_df.columns
        ),
        None
    )

    longitude_col = next(
        (
            c for c in [
                "Longitude", "longitude", "Lon", "lon", "Long"
            ]
            if c in map_df.columns
        ),
        None
    )

    id_col = next(
        (
            c for c in [
                "SystemCodeNumber", "ParkingID", "Location", "Name"
            ]
            if c in map_df.columns
        ),
        None
    )

    if latitude_col and longitude_col and id_col:

        valid_map = map_df.dropna(
            subset=[latitude_col, longitude_col, id_col]
        ).copy()

        valid_map[latitude_col] = pd.to_numeric(
            valid_map[latitude_col], errors="coerce"
        )

        valid_map[longitude_col] = pd.to_numeric(
            valid_map[longitude_col], errors="coerce"
        )

        valid_map = valid_map.dropna(
            subset=[latitude_col, longitude_col]
        )

        parking_map = folium.Map(
            location=[52.479, -1.902],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        for _, row in valid_map.iterrows():

            parking_id = str(row[id_col])

            folium.Marker(
                location=[
                    float(row[latitude_col]),
                    float(row[longitude_col])
                ],
                tooltip=parking_id,
                popup=f"Parking Location: {parking_id}",
                icon=folium.Icon(
                    color="blue",
                    icon="parking",
                    prefix="fa"
                )
            ).add_to(parking_map)

        map_col, details_col = st.columns([2, 1])

        with map_col:
            st_folium(
                parking_map,
                height=430,
                width=None,
                key="parking_map"
            )

        with details_col:

            map_selection = st.selectbox(
                "Select Parking Location",
                parking_options,
                key="map_location_select"
            )

            selected_info = df[
                df["SystemCodeNumber"] == map_selection
            ]

            if not selected_info.empty:
                st.metric(
                    "Parking Capacity",
                    int(selected_info["Capacity"].iloc[0])
                )

                st.write(f"Parking ID: {map_selection}")

            st.session_state["map_selected_location"] = (
                map_selection
            )

        st.caption(
            "Map coordinates may be approximate demo locations. "
            "Verify them before real-world navigation."
        )

    else:
        st.warning(
            "Map CSV must contain a location ID, latitude and longitude."
        )

else:
    st.warning("Parking map file was not found.")


# =========================================================
# PREDICTION
# =========================================================

st.markdown(
    '<div id="prediction" class="section-title">'
    '🔮 Context-Aware Parking Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Predict parking availability using temporal, weather "
    "and calendar information."
)

p1, p2, p3 = st.columns(3)

with p1:

    default_location = st.session_state.get(
        "map_selected_location", parking_options[0]
    )

    selected_location = st.selectbox(
        "Parking Location",
        parking_options,
        index=(
            parking_options.index(default_location)
            if default_location in parking_options else 0
        ),
        key="prediction_location"
    )

with p2:

    selected_date = st.date_input(
        "Select Date",
        value=date.today(),
        key="prediction_date"
    )

with p3:

    selected_time = st.time_input(
        "Select Time",
        value=time(10, 0),
        key="prediction_time"
    )


location_history = df[
    df["SystemCodeNumber"] == selected_location
].copy()

location_history["LastUpdated"] = pd.to_datetime(
    location_history["LastUpdated"],
    errors="coerce"
)

location_history = location_history.dropna(
    subset=["LastUpdated"]
).sort_values("LastUpdated")

default_occupancy = int(
    location_history["Occupancy"].iloc[-1]
)

capacity = int(
    location_history["Capacity"].iloc[0]
)

previous_occupancy = st.number_input(
    "Latest Known Occupancy",
    min_value=0,
    max_value=capacity,
    value=min(default_occupancy, capacity),
    step=1,
    key=f"prediction_occupancy_{selected_location}"
)

st.info(
    "The default occupancy comes from historical data. "
    "It is not a live sensor reading."
)

prediction_context = context_input_section(
    "prediction_context"
)


if st.button(
    "🚀 Predict Parking Availability",
    use_container_width=True
):

    try:

        result = make_prediction(
            selected_location,
            selected_date,
            selected_time,
            previous_occupancy,
            prediction_context
        )

        st.session_state["prediction_result"] = result

        st.session_state["prediction_saved_location"] = (
            selected_location
        )

        st.session_state["prediction_saved_date"] = (
            selected_date.isoformat()
        )

        st.session_state["prediction_saved_time"] = (
            selected_time.strftime("%H:%M:%S")
        )

        st.session_state["prediction_created_at"] = (
            datetime.now().isoformat()
        )

        st.success("Context-aware prediction generated successfully!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


if st.session_state.get("prediction_result") is not None:

    result = st.session_state["prediction_result"]

    st.subheader("Prediction Results")

    r1, r2, r3 = st.columns(3)

    r1.metric("Available Spaces", result["available"])
    r2.metric("Predicted Occupancy", result["occupancy"])
    r3.metric("Total Capacity", result["capacity"])

    st.progress(
        min(result["occupancy_percentage"] / 100, 1.0)
    )

    st.write(
        f"Estimated Occupancy: "
        f"{result['occupancy_percentage']:.1f}%"
    )

    if result["available"] == 0:
        st.error("No spaces are predicted to be available.")

    elif result["occupancy_percentage"] >= 80:
        st.warning("High parking occupancy is predicted.")

    else:
        st.success("Parking spaces may be available.")

    st.subheader("Context Used for Prediction")

    used_features = result["features"]

    a, b, c = st.columns(3)

    a.metric(
        "Temperature",
        f"{used_features['temperature_2m']:.1f} °C"
    )

    b.metric(
        "Rainfall",
        f"{used_features['precipitation']:.1f} mm"
    )

    c.metric(
        "Humidity",
        f"{used_features['relative_humidity_2m']:.0f}%"
    )

    st.write(
        "School Holiday:",
        "Yes" if used_features["IsSchoolHoliday"] else "No"
    )

    st.write(
        "Half-Term:",
        "Yes" if used_features["IsHalfTerm"] else "No"
    )

    st.write(
        "Christmas Period:",
        "Yes" if used_features["IsChristmasPeriod"] else "No"
    )

    st.caption(
        "Predictions depend on the supplied weather and calendar "
        "values. They are estimates based on historical training data, "
        "not live availability guarantees."
    )


# =========================================================
# DECISION SUPPORT
# =========================================================

st.markdown(
    '<div id="decision-support" class="section-title">'
    '🧭 Smart Parking Decision Support</div>',
    unsafe_allow_html=True
)

st.write(
    "Compare parking locations and arrival times using "
    "the E4 temporal, weather and calendar model."
)

st.info(
    "The same weather and calendar context is applied to all "
    "locations and arrival-time candidates in this analysis."
)

decision_history = df.copy()

decision_history["LastUpdated"] = pd.to_datetime(
    decision_history["LastUpdated"],
    errors="coerce"
)

decision_history = (
    decision_history.dropna(subset=["LastUpdated"])
    .sort_values("LastUpdated")
    .groupby("SystemCodeNumber")
    .tail(1)
    .copy()
)

decision_input = decision_history[
    ["SystemCodeNumber", "Capacity", "Occupancy"]
].copy()

decision_input = decision_input.rename(
    columns={
        "SystemCodeNumber": "Parking Location",
        "Occupancy": "Latest Known Occupancy"
    }
)

decision_input["Capacity"] = (
    decision_input["Capacity"].astype(int)
)

decision_input["Latest Known Occupancy"] = (
    decision_input["Latest Known Occupancy"].round().astype(int)
)

decision_input = decision_input.sort_values(
    "Parking Location"
).reset_index(drop=True)


with st.expander(
    "Edit latest occupancy for each parking location"
):

    edited_occupancy = st.data_editor(
        decision_input,
        use_container_width=True,
        hide_index=True,
        disabled=["Parking Location", "Capacity"],
        key="decision_occupancy_editor"
    )


d1, d2, d3 = st.columns(3)

with d1:
    decision_date = st.date_input(
        "Decision Support Date",
        value=date.today(),
        key="decision_date"
    )

with d2:
    decision_start = st.time_input(
        "Arrival Window Start",
        value=time(9, 0),
        key="decision_start"
    )

with d3:
    decision_end = st.time_input(
        "Arrival Window End",
        value=time(17, 0),
        key="decision_end"
    )


interval = st.selectbox(
    "Evaluate arrival times every",
    [15, 30, 60],
    index=1,
    format_func=lambda x: f"{x} minutes",
    key="decision_interval"
)


decision_context = context_input_section(
    "decision_context"
)


if st.button(
    "🔎 Find Best Parking Location and Time",
    use_container_width=True
):

    occupancy_values = pd.to_numeric(
        edited_occupancy["Latest Known Occupancy"],
        errors="coerce"
    )

    capacities = pd.to_numeric(
        edited_occupancy["Capacity"],
        errors="coerce"
    )

    invalid = (
        occupancy_values.isna()
        | (occupancy_values < 0)
        | (occupancy_values > capacities)
    )

    if invalid.any():
        st.error(
            "Check occupancy values. They must be within capacity."
        )

    elif decision_end <= decision_start:
        st.error(
            "End time must be later than start time."
        )

    else:

        start_minutes = (
            decision_start.hour * 60 + decision_start.minute
        )

        end_minutes = (
            decision_end.hour * 60 + decision_end.minute
        )

        candidate_minutes = list(
            range(start_minutes, end_minutes + 1, interval)
        )

        if candidate_minutes[-1] != end_minutes:
            candidate_minutes.append(end_minutes)

        recommendation_records = []

        total_predictions = (
            len(edited_occupancy) * len(candidate_minutes)
        )

        progress_bar = st.progress(0)
        completed = 0

        for _, row in edited_occupancy.iterrows():

            parking_id = str(row["Parking Location"])

            current_occupancy = int(
                row["Latest Known Occupancy"]
            )

            for minute_value in candidate_minutes:

                arrival_time = time(
                    minute_value // 60,
                    minute_value % 60
                )

                try:

                    prediction_result = make_prediction(
                        parking_id,
                        decision_date,
                        arrival_time,
                        current_occupancy,
                        decision_context
                    )

                    recommendation_records.append({

                        "Parking Location": parking_id,

                        "Arrival Time": arrival_time.strftime("%H:%M"),

                        "Capacity": prediction_result["capacity"],

                        "Predicted Occupancy": (
                            prediction_result["occupancy"]
                        ),

                        "Available Spaces": (
                            prediction_result["available"]
                        ),

                        "Occupancy (%)": round(
                            prediction_result["occupancy_percentage"],
                            1
                        )
                    })

                except Exception as error:

                    st.warning(
                        f"Could not evaluate {parking_id}: {error}"
                    )

                completed += 1

                progress_bar.progress(
                    completed / total_predictions
                )

        progress_bar.empty()

        if recommendation_records:

            recommendation_df = pd.DataFrame(
                recommendation_records
            )

            recommendation_df = recommendation_df.sort_values(
                by=["Available Spaces", "Occupancy (%)"],
                ascending=[False, True]
            ).reset_index(drop=True)

            st.session_state["decision_recommendations"] = (
                recommendation_df
            )

            st.success(
                "Decision-support analysis completed!"
            )

        else:
            st.error(
                "No recommendations could be generated."
            )


if "decision_recommendations" in st.session_state:

    recommendation_df = (
        st.session_state["decision_recommendations"]
    )

    st.subheader("🏆 Recommended Parking Options")

    best_option = recommendation_df.iloc[0]

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Best Parking Location",
        best_option["Parking Location"]
    )

    b2.metric(
        "Best Arrival Time",
        best_option["Arrival Time"]
    )

    b3.metric(
        "Available Spaces",
        int(best_option["Available Spaces"])
    )

    st.success(
        f"Recommended: {best_option['Parking Location']} "
        f"at {best_option['Arrival Time']} with an estimated "
        f"{int(best_option['Available Spaces'])} available spaces."
    )

    st.subheader("Top 10 Parking Recommendations")

    st.dataframe(
        recommendation_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    availability_chart = px.line(
        recommendation_df,
        x="Arrival Time",
        y="Available Spaces",
        color="Parking Location",
        markers=True,
        title="Predicted Availability by Arrival Time"
    )

    availability_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#151f32",
        font_color="#e2e8f0",
        height=420
    )

    st.plotly_chart(
        availability_chart,
        use_container_width=True
    )

    st.caption(
        "The same latest-known occupancy and contextual weather "
        "values are used for each evaluated time slot. "
        "These are model estimates, not live guarantees."
    )


# =========================================================
# ANALYTICS
# =========================================================

st.markdown(
    '<div id="analytics" class="section-title">'
    '📈 Parking Analytics</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2)

with chart1:

    hourly_data = (
        df.groupby("Hour", as_index=False)["Occupancy"]
        .mean()
        .sort_values("Hour")
    )

    fig_hour = px.line(
        hourly_data,
        x="Hour",
        y="Occupancy",
        markers=True,
        title="Average Occupancy by Hour"
    )

    fig_hour.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#151f32",
        font_color="#e2e8f0",
        height=350
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )


with chart2:

    location_data = (
        df.groupby("SystemCodeNumber", as_index=False)["Occupancy"]
        .mean()
        .sort_values("Occupancy", ascending=False)
        .head(10)
    )

    fig_location = px.bar(
        location_data,
        x="SystemCodeNumber",
        y="Occupancy",
        title="Top 10 Locations by Average Occupancy"
    )

    fig_location.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#151f32",
        font_color="#e2e8f0",
        height=350
    )

    st.plotly_chart(
        fig_location,
        use_container_width=True
    )


chart3, chart4 = st.columns(2)

with chart3:

    weekday_data = (
        df.groupby("Weekday", as_index=False)["Occupancy"]
        .mean()
    )

    weekday_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    weekday_data["Day"] = weekday_data["Weekday"].map(
        weekday_names
    )

    fig_weekday = px.bar(
        weekday_data,
        x="Day",
        y="Occupancy",
        title="Average Occupancy by Day"
    )

    fig_weekday.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#151f32",
        font_color="#e2e8f0",
        height=350
    )

    st.plotly_chart(
        fig_weekday,
        use_container_width=True
    )


with chart4:

    occupancy_data = pd.DataFrame({
        "Status": ["Occupied", "Available"],
        "Spaces": [latest_occupancy, latest_available]
    })

    fig_pie = px.pie(
        occupancy_data,
        names="Status",
        values="Spaces",
        hole=0.55,
        title="Latest Historical Parking Distribution"
    )

    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        font_color="#e2e8f0",
        height=350
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown(
    '<div id="model-performance" class="section-title">'
    '🧠 Model Performance</div>',
    unsafe_allow_html=True
)

st.success(f"Deployment Model: {BEST_MODEL_NAME}")

if not comparison.empty:

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    model_col = next(
        (
            c for c in [
                "Model",
                "model",
                "Model_Name",
                "model_name",
                "Algorithm"
            ]
            if c in comparison.columns
        ),
        None
    )

    metric_options = [
        c for c in ["MAE", "RMSE", "R2 Score"]
        if c in comparison.columns
    ]

    if model_col and metric_options:

        selected_metric = st.selectbox(
            "Select Performance Metric",
            metric_options
        )

        plot_data = comparison.copy()

        plot_data[selected_metric] = pd.to_numeric(
            plot_data[selected_metric],
            errors="coerce"
        )

        plot_data = plot_data.dropna(
            subset=[selected_metric]
        )

        fig_model = px.bar(
            plot_data,
            x=model_col,
            y=selected_metric,
            color=model_col,
            title=f"Model Comparison - {selected_metric}"
        )

        fig_model.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#151f32",
            font_color="#e2e8f0",
            height=380,
            showlegend=False
        )

        st.plotly_chart(
            fig_model,
            use_container_width=True
        )

else:
    st.warning(
        "Model comparison file was not found or is empty."
    )


# =========================================================
# EXPLAINABLE AI - SHAP
# =========================================================

st.markdown(
    '<div id="explainable-ai" class="section-title">'
    '🔍 Explainable AI - SHAP Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Understand which features influence the E4 model's predictions."
)

SHAP_SUMMARY_PATH = (
    BASE_DIR / "outputs" / "shap_summary.png"
)

SHAP_IMPORTANCE_PATH = (
    BASE_DIR / "outputs" / "shap_feature_importance.png"
)

SHAP_CSV_PATH = (
    BASE_DIR / "outputs" / "shap_importance.csv"
)


if SHAP_SUMMARY_PATH.exists() and SHAP_IMPORTANCE_PATH.exists():

    st.success("SHAP analysis files loaded successfully.")

    if SHAP_CSV_PATH.exists():

        shap_df = pd.read_csv(SHAP_CSV_PATH)

        required_shap_columns = [
            "Feature",
            "Mean_SHAP_Importance"
        ]

        if all(
            c in shap_df.columns for c in required_shap_columns
        ):

            shap_df["Mean_SHAP_Importance"] = pd.to_numeric(
                shap_df["Mean_SHAP_Importance"],
                errors="coerce"
            )

            shap_df = shap_df.dropna(
                subset=["Mean_SHAP_Importance"]
            ).sort_values(
                "Mean_SHAP_Importance",
                ascending=False
            )

            top_features = shap_df.head(5).copy()

            st.subheader("Top 5 Influential Features")

            shap_col1, shap_col2 = st.columns(2)

            with shap_col1:
                st.dataframe(
                    top_features,
                    use_container_width=True,
                    hide_index=True
                )

            with shap_col2:

                fig_shap = px.bar(
                    top_features.sort_values(
                        "Mean_SHAP_Importance",
                        ascending=True
                    ),
                    x="Mean_SHAP_Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 5 SHAP Features"
                )

                fig_shap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#151f32",
                    font_color="#e2e8f0",
                    height=280
                )

                st.plotly_chart(
                    fig_shap,
                    use_container_width=True
                )

    st.divider()

    shap_image_col1, shap_image_col2 = st.columns(2)

    with shap_image_col1:

        st.markdown("**SHAP Feature Importance**")

        st.image(
            str(SHAP_IMPORTANCE_PATH),
            use_container_width=True
        )

    with shap_image_col2:

        st.markdown("**SHAP Summary Plot**")

        st.image(
            str(SHAP_SUMMARY_PATH),
            use_container_width=True
        )

    st.info(
        "SHAP explains model associations and feature contributions. "
        "It does not establish causal effects."
    )

else:

    st.warning(
        "SHAP outputs not found. Generate SHAP plots for the E4 model "
        "before interpreting these results."
    )


# =========================================================
# CONCEPT DRIFT MONITORING
# =========================================================

st.markdown(
    '<div id="concept-drift" class="section-title">'
    '🔄 Concept Drift Monitoring</div>',
    unsafe_allow_html=True
)

st.write(
    "Compare saved E4 predictions with actual occupancy readings "
    "for the same location and prediction target time."
)

st.info(
    "Only enter actual occupancy after it has been observed "
    "at the prediction's selected date and time."
)


saved_prediction = st.session_state.get(
    "prediction_result"
)

saved_location = st.session_state.get(
    "prediction_saved_location"
)

saved_date = st.session_state.get(
    "prediction_saved_date"
)

saved_time = st.session_state.get(
    "prediction_saved_time"
)

saved_created_at = st.session_state.get(
    "prediction_created_at"
)


if saved_prediction is None:

    st.warning(
        "Generate a prediction in the Prediction section first."
    )

else:

    saved_capacity = int(
        saved_prediction["capacity"]
    )

    saved_predicted = int(
        saved_prediction["occupancy"]
    )

    st.subheader("Saved E4 Prediction")

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Parking Location",
        saved_location
    )

    info2.metric(
        "Predicted Occupancy",
        saved_predicted
    )

    info3.metric(
        "Capacity",
        saved_capacity
    )

    st.write(f"**Prediction target date:** {saved_date}")
    st.write(f"**Prediction target time:** {saved_time}")

    st.caption(
        f"Prediction generated at: {saved_created_at}"
    )

    actual_value = st.number_input(
        "Observed Actual Occupancy",
        min_value=0,
        max_value=saved_capacity,
        value=0,
        step=1,
        key="drift_actual_value"
    )

    confirmed_actual = st.checkbox(
        "I confirm this is the actual observed occupancy "
        "for the saved prediction's target time.",
        key="confirm_actual_occupancy"
    )

    if st.button(
        "💾 Save Matched Actual Observation",
        use_container_width=True
    ):

        prediction_key = (
            f"{saved_location}_{saved_date}_"
            f"{saved_time}_{saved_created_at}"
        )

        already_saved = st.session_state.get(
            "saved_prediction_keys", []
        )

        if prediction_key in already_saved:

            st.warning(
                "An observation has already been saved "
                "for this prediction."
            )

        elif not confirmed_actual:

            st.warning(
                "Please confirm that this is a genuine "
                "observed occupancy value."
            )

        else:

            try:

                feature_values = dict(
                    saved_prediction["features"]
                )

                feature_values["Prediction_Target_Date"] = (
                    saved_date
                )

                feature_values["Prediction_Target_Time"] = (
                    saved_time
                )

                feature_values["Prediction_Created_At"] = (
                    saved_created_at
                )

                save_observation(
                    saved_location,
                    saved_predicted,
                    actual_value,
                    feature_values=feature_values
                )

                already_saved.append(prediction_key)

                st.session_state["saved_prediction_keys"] = (
                    already_saved
                )

                st.success(
                    "Matched E4 prediction and actual occupancy saved!"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to save observation: {e}"
                )


st.divider()

st.subheader("📊 Drift Detection Results")

try:

    drift_result = detect_drift(
        window_size=50,
        threshold=1.5
    )

    st.metric(
        "Total Recorded Observations",
        drift_result["records"]
    )

    if drift_result["status"] == "Insufficient Data":

        st.warning(
            "At least 100 valid observations are required "
            "for the current monitoring configuration."
        )

    else:

        mcol1, mcol2 = st.columns(2)

        with mcol1:

            st.metric(
                "Baseline MAE",
                f"{drift_result['baseline_mae']:.2f}"
            )

        with mcol2:

            st.metric(
                "Recent MAE",
                f"{drift_result['recent_mae']:.2f}"
            )

        if drift_result["drift_detected"]:

            st.error(
                "Potential prediction performance degradation detected. "
                "Review the observations before considering retraining."
            )

        else:

            st.success(
                "Prediction performance is stable according "
                "to the current monitoring threshold."
            )

    if OBSERVATION_PATH.exists():

        observation_data = pd.read_csv(
            OBSERVATION_PATH
        )

        if not observation_data.empty:

            st.subheader("Recent Monitoring Records")

            display_columns = [
                c for c in [
                    "Timestamp",
                    "SystemCodeNumber",
                    "Predicted_Occupancy",
                    "Actual_Occupancy"
                ]
                if c in observation_data.columns
            ]

            st.dataframe(
                observation_data[display_columns].tail(10).iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

            if (
                "Predicted_Occupancy" in observation_data.columns
                and "Actual_Occupancy" in observation_data.columns
            ):

                chart_data = observation_data.copy()

                chart_data["Absolute_Error"] = abs(
                    pd.to_numeric(
                        chart_data["Actual_Occupancy"],
                        errors="coerce"
                    )
                    -
                    pd.to_numeric(
                        chart_data["Predicted_Occupancy"],
                        errors="coerce"
                    )
                )

                st.subheader("Prediction Error History")

                fig_error = px.line(
                    chart_data.reset_index(),
                    x="index",
                    y="Absolute_Error",
                    markers=True,
                    title="Absolute Prediction Error by Observation"
                )

                fig_error.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#151f32",
                    font_color="#e2e8f0",
                    height=350
                )

                st.plotly_chart(
                    fig_error,
                    use_container_width=True
                )

except Exception as e:

    st.error(
        f"Drift monitoring error: {e}"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Smart Parking AI | Context-Aware Machine Learning Project<br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
