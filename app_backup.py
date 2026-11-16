
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium
from pathlib import Path
from datetime import datetime, date, time


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Parking Dashboard",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_parking.csv"
LOCATIONS_FILE = BASE_DIR / "dataset" / "parking_locations_demo.csv"
MODEL_FILE = BASE_DIR / "models" / "best_parking_model.pkl"
MODEL_NAME_FILE = BASE_DIR / "models" / "best_model_name.txt"
COMPARISON_FILE = BASE_DIR / "outputs" / "model_comparison.csv"


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background-color: #161b22;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    data["LastUpdated"] = pd.to_datetime(
        data["LastUpdated"],
        errors="coerce"
    )

    data = data.dropna(subset=["LastUpdated"])

    return data


@st.cache_data
def load_locations():

    return pd.read_csv(LOCATIONS_FILE)


@st.cache_resource
def load_model():

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_comparison():

    return pd.read_csv(COMPARISON_FILE)


df = load_data()
locations = load_locations()
model = load_model()
comparison = load_comparison()


# ==================================================
# 4. MODEL INFORMATION
# ==================================================

try:
    BEST_MODEL = MODEL_NAME_FILE.read_text().strip()
except Exception:
    BEST_MODEL = "Random Forest"


parking_options = sorted(
    locations["SystemCodeNumber"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ==================================================
# 5. PREDICTION FUNCTION
# ==================================================

def get_prediction(parking_id, target_datetime):

    location_data = df[
        df["SystemCodeNumber"].astype(str) == str(parking_id)
    ].copy()

    if location_data.empty:
        return None, None, None

    location_data = location_data.sort_values(
        "LastUpdated"
    )

    capacity = int(
        location_data["Capacity"].iloc[0]
    )

    # Historical occupancy for selected weekday and hour
    matching_data = location_data[
        (
            location_data["LastUpdated"].dt.weekday
            == target_datetime.weekday()
        )
        &
        (
            location_data["LastUpdated"].dt.hour
            == target_datetime.hour
        )
    ]

    if not matching_data.empty:

        previous_occupancy = float(
            matching_data["Occupancy"].median()
        )

    else:

        previous_occupancy = float(
            location_data["Occupancy"].median()
        )

    hour = target_datetime.hour
    minute = target_datetime.minute
    day = target_datetime.day
    month = target_datetime.month
    weekday = target_datetime.weekday()

    total_minutes = hour * 60 + minute

    features = pd.DataFrame([{

        "SystemCodeNumber": str(parking_id),

        "Capacity": capacity,

        "Hour": hour,

        "Minute": minute,

        "Day": day,

        "Month": month,

        "Weekday": weekday,

        "IsWeekend": int(weekday >= 5),

        "Previous_Occupancy": previous_occupancy,

        "Time_sin": np.sin(
            2 * np.pi * total_minutes / 1440
        ),

        "Time_cos": np.cos(
            2 * np.pi * total_minutes / 1440
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

    }])

    prediction = model.predict(features)[0]

    prediction = float(
        np.clip(prediction, 0, capacity)
    )

    available = max(
        0,
        capacity - prediction
    )

    return prediction, available, capacity


# ==================================================
# 6. INTERACTIVE PARKING MAP
# ==================================================

def create_parking_map():

    parking_map = folium.Map(
        location=[52.477, -1.895],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    for _, row in locations.iterrows():

        parking_id = str(row["SystemCodeNumber"])

        latitude = row["Latitude"]
        longitude = row["Longitude"]

        capacity = int(row["Capacity"])

        location_type = str(
            row.get(
                "LocationType",
                "Approximate demo marker"
            )
        )

        if pd.isna(latitude) or pd.isna(longitude):
            continue

        is_provisional = "Provisional" in location_type

        marker_color = (
            "blue" if is_provisional else "orange"
        )

        popup_text = f"""
        <b>Parking ID:</b> {parking_id}<br>
        <b>Capacity:</b> {capacity} spaces<br>
        <b>Status:</b> {location_type}
        """

        folium.Marker(

            location=[
                float(latitude),
                float(longitude)
            ],

            tooltip=folium.Tooltip(
                parking_id,
                sticky=True
            ),

            popup=folium.Popup(
                popup_text,
                max_width=350
            ),

            icon=folium.Icon(
                color=marker_color,
                icon="info-sign"
            )

        ).add_to(parking_map)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        width: 340px;
        background-color: white;
        border: 2px solid grey;
        border-radius: 10px;
        padding: 15px;
        z-index: 9999;
        font-size: 14px;
        color: black;
    ">
        <b>Parking Location Legend</b><br>
        <span style="color:blue;">●</span>
        Provisional coordinates<br>
        <span style="color:orange;">●</span>
        Approximate demo coordinates
    </div>
    """

    parking_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    return parking_map


# ==================================================
# 7. MAIN DASHBOARD HEADER
# ==================================================

st.title("🚗 Smart Parking Availability Dashboard")

st.write(
    "Machine Learning Based Parking Occupancy "
    "and Availability Prediction System"
)

st.markdown("---")


# ==================================================
# 8. OVERVIEW
# ==================================================

st.header("📊 Dashboard Overview")

total_locations = df["SystemCodeNumber"].nunique()
total_records = len(df)
total_capacity = locations["Capacity"].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Parking Locations",
        total_locations
    )

with col2:
    st.metric(
        "Dataset Records",
        f"{total_records:,}"
    )

with col3:
    st.metric(
        "Total Parking Capacity",
        f"{total_capacity:,}"
    )

with col4:
    st.metric(
        "Best ML Model",
        BEST_MODEL
    )


st.info(
    "Predictions are generated from historical parking data. "
    "This application does not use live parking sensors."
)

st.markdown("---")


# ==================================================
# 9. PARKING PREDICTION
# ==================================================

st.header("📍 Parking Availability Prediction")

st.write(
    "Select a parking location from the dropdown "
    "or click a marker on the map."
)

# Preserve parking selection
if "selected_location" not in st.session_state:

    st.session_state.selected_location = parking_options[0]

if "last_clicked_parking_id" not in st.session_state:

    st.session_state.last_clicked_parking_id = None


# Interactive map
parking_map = create_parking_map()

map_data = st_folium(
    parking_map,
    width=None,
    height=500,
    returned_objects=[
        "last_object_clicked_tooltip"
    ],
    key="parking_map"
)

clicked_id = None

if map_data:

    clicked_id = map_data.get(
        "last_object_clicked_tooltip"
    )

if (
    clicked_id in parking_options
    and clicked_id != st.session_state.last_clicked_parking_id
):

    st.session_state.selected_location = clicked_id

    st.session_state.last_clicked_parking_id = clicked_id

    st.rerun()


st.markdown("### Select Parking Details")

col1, col2, col3 = st.columns(3)

with col1:

    selected_location = st.selectbox(
        "Parking Location",
        parking_options,
        key="selected_location"
    )

with col2:

    selected_date = st.date_input(
        "Select Date",
        value=date.today(),
        key="prediction_date"
    )

with col3:

    selected_time = st.time_input(
        "Select Time",
        value=time(10, 0),
        key="prediction_time"
    )


target_datetime = datetime.combine(
    selected_date,
    selected_time
)


st.markdown("---")


# Prediction button
if st.button(
    "🔮 Predict Parking Availability",
    use_container_width=True
):

    prediction, available, capacity = get_prediction(
        selected_location,
        target_datetime
    )

    if prediction is None:

        st.error(
            "Unable to generate prediction for this location."
        )

    else:

        st.session_state.prediction_result = {

            "location": selected_location,

            "datetime": target_datetime,

            "prediction": prediction,

            "available": available,

            "capacity": capacity

        }


# ==================================================
# 10. PREDICTION RESULTS
# ==================================================

st.header("📈 Prediction Results")

result = st.session_state.get(
    "prediction_result"
)

if result is not None:

    if (
        result["location"] == selected_location
        and result["datetime"] == target_datetime
    ):

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Occupancy",
                f"{result['prediction']:.0f} spaces"
            )

        with col2:

            st.metric(
                "Available Spaces",
                f"{result['available']:.0f} spaces"
            )

        with col3:

            st.metric(
                "Total Capacity",
                result["capacity"]
            )

        occupancy_percentage = (
            result["prediction"] / result["capacity"]
        ) * 100 if result["capacity"] else 0

        st.write("Parking Occupancy Level")

        st.progress(
            int(
                np.clip(
                    occupancy_percentage,
                    0,
                    100
                )
            )
        )

        st.write(
            f"Estimated occupancy: "
            f"{occupancy_percentage:.2f}%"
        )

        st.caption(
            f"Parking ID: {result['location']} | "
            f"Date and Time: {result['datetime']}"
        )

        report = pd.DataFrame([{

            "Parking ID": result["location"],

            "Date and Time": result["datetime"],

            "Predicted Occupancy": round(
                result["prediction"]
            ),

            "Available Spaces": round(
                result["available"]
            ),

            "Capacity": result["capacity"]

        }])

        st.download_button(
            "⬇️ Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="parking_prediction_report.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Your selected details have changed. "
            "Click Predict Parking Availability to "
            "generate a new prediction."
        )

else:

    st.info(
        "Select a parking location, date and time, "
        "then click Predict to see the results."
    )


st.markdown("---")


# ==================================================
# 11. PARKING ANALYTICS
# ==================================================

st.header("📊 Parking Analytics")

st.subheader("Parking Location Capacity")

capacity_data = locations[
    ["SystemCodeNumber", "Capacity"]
].sort_values(
    "Capacity",
    ascending=False
)

st.bar_chart(
    capacity_data.set_index(
        "SystemCodeNumber"
    )
)


st.markdown("---")

st.subheader("Historical Occupancy Trend")

selected_analytics_location = st.selectbox(
    "Select Parking Location for Analytics",
    sorted(df["SystemCodeNumber"].astype(str).unique()),
    key="analytics_location"
)

analytics_data = df[
    df["SystemCodeNumber"].astype(str)
    == selected_analytics_location
].sort_values(
    "LastUpdated"
)

if not analytics_data.empty:

    chart_data = analytics_data[
        ["LastUpdated", "Occupancy"]
    ].set_index("LastUpdated")

    st.line_chart(chart_data)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Average Occupancy",
            f"{analytics_data['Occupancy'].mean():.0f}"
        )

    with col2:

        st.metric(
            "Maximum Occupancy",
            f"{analytics_data['Occupancy'].max():.0f}"
        )

else:

    st.warning("No analytics data available.")


st.markdown("---")


# ==================================================
# 12. MODEL PERFORMANCE
# ==================================================

st.header("🤖 Machine Learning Model Performance")

st.write(
    "Comparison of the machine learning algorithms "
    "used to predict parking occupancy."
)

st.subheader("Model Comparison Table")

st.dataframe(
    comparison,
    use_container_width=True
)


# Find model name column
model_column = next(
    (
        col for col in comparison.columns
        if col.lower() in [
            "model",
            "model_name",
            "algorithm"
        ]
    ),
    None
)

numeric_columns = comparison.select_dtypes(
    include=np.number
).columns.tolist()


if model_column and numeric_columns:

    st.subheader("Performance Comparison Chart")

    metric_column = st.selectbox(
        "Choose Performance Metric",
        numeric_columns,
        key="performance_metric"
    )

    chart_data = comparison[
        [model_column, metric_column]
    ].set_index(model_column)

    st.bar_chart(chart_data)


st.success(
    f"Best Performing Model: {BEST_MODEL}"
)

st.markdown("""
**Performance Metrics:**

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains
  variation in parking occupancy.
""")


st.markdown("---")


# ==================================================
# 13. ABOUT PROJECT
# ==================================================

st.header("ℹ️ About Project")

st.subheader(
    "Smart Parking Availability Prediction Using Machine Learning"
)

st.write("""
### Objective

To predict parking occupancy and estimate available
parking spaces using historical parking data.

### Dataset

Birmingham Parking Dataset.

### Machine Learning Algorithms

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Folium
- Matplotlib

### Main Features

- Interactive parking location map
- Parking location selection
- Date and time-based prediction
- Parking availability estimation
- Historical occupancy analytics
- Machine learning model comparison
- Prediction report download
""")

st.warning(
    "Map note: Three coordinates are provisional. "
    "The remaining demo markers are approximate "
    "illustrative coordinates, not verified real "
    "parking facility locations."
)


# ==================================================
# 14. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <p>Smart Parking Availability Prediction</p>
    <p>Machine Learning Project</p>
    </center>
    """,
    unsafe_allow_html=True
)
