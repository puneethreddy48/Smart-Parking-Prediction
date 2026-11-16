import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from streamlit_folium import st_folium
from datetime import date, time

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "cleaned_parking.csv"
MAP_PATH = BASE_DIR / "dataset" / "parking_locations_demo.csv"
MODEL_PATH = BASE_DIR / "models" / "best_parking_model.pkl"
COMPARISON_PATH = BASE_DIR / "outputs" / "model_comparison.csv"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

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

.metric-card {
    background: #151f32;
    border: 1px solid #29364b;
    border-radius: 15px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 750;
}

.metric-note {
    color: #38bdf8;
    font-size: 12px;
    margin-top: 5px;
}

.info-card {
    background: #151f32;
    border: 1px solid #29364b;
    border-radius: 15px;
    padding: 20px;
}

div[data-testid="stVerticalBlock"] div[data-testid="stMetric"] {
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

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #151f32;
}

label, .stMarkdown, p {
    color: #e2e8f0;
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


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


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
    model = load_model()
    map_df = load_map_data()
    comparison = load_comparison()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.info(
        "Check that dataset/cleaned_parking.csv and "
        "models/best_parking_model.pkl exist."
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
    "Previous_Occupancy"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing columns in cleaned_parking.csv: {missing_columns}")
    st.stop()


# --------------------------------------------------
# COMMON DATA
# --------------------------------------------------

parking_options = sorted(
    df["SystemCodeNumber"].dropna().unique().tolist()
)

total_locations = len(parking_options)

total_capacity = int(
    df.drop_duplicates("SystemCodeNumber")["Capacity"].sum()
)

latest_occupancy = df["Occupancy"].sum()

latest_available = max(
    total_capacity - latest_occupancy, 0
)

if "selected_location" not in st.session_state:
    st.session_state.selected_location = parking_options[0]

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🅿️ Smart Parking AI</h1>
    <p>
        Intelligent Parking Availability Prediction System
        | Powered by Machine Learning
    </p>
    <p>
        Explore parking locations, predict availability and
        understand parking patterns through data analytics.
    </p>
</div>
""", unsafe_allow_html=True)

nav_items = [
    "Overview",
    "Parking Map",
    "Prediction",
    "Analytics",
    "Model Performance",
    "About"
]

nav_cols = st.columns(len(nav_items))

for col, item in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f'<a href="#{item.lower().replace(" ", "-")}" '
            f'style="display:block;text-align:center;'
            f'padding:11px 4px;background:#17243a;'
            f'border:1px solid #29364b;border-radius:9px;'
            f'color:#7dd3fc;text-decoration:none;'
            f'font-size:13px;font-weight:600;">{item}</a>',
            unsafe_allow_html=True
        )


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div id="overview" class="section-title">📊 Project Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Parking Locations", f"{total_locations}")

with c2:
    st.metric("Total Parking Capacity", f"{total_capacity:,}")

with c3:
    st.metric("Recorded Occupancy", f"{int(latest_occupancy):,}")

with c4:
    st.metric("Calculated Available Spaces", f"{int(latest_available):,}")

st.caption(
    "Overview figures are calculated from the cleaned historical dataset, "
    "not from live parking sensors."
)


# --------------------------------------------------
# PARKING MAP
# --------------------------------------------------

st.markdown(
    '<div id="parking-map" class="section-title">📍 Interactive Parking Map</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore the parking markers. Select a parking location "
    "to use it in the prediction section."
)

if not map_df.empty:

    latitude_col = next(
        (c for c in ["Latitude", "latitude", "Lat", "lat"]
         if c in map_df.columns),
        None
    )

    longitude_col = next(
        (c for c in ["Longitude", "longitude", "Lon", "lon", "Long"]
         if c in map_df.columns),
        None
    )

    id_col = next(
        (c for c in ["SystemCodeNumber", "ParkingID", "Location", "Name"]
         if c in map_df.columns),
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
            latitude = float(row[latitude_col])
            longitude = float(row[longitude_col])

            capacity = row.get("Capacity", "N/A")

            location_type = row.get(
                "LocationType",
                "Location accuracy not specified"
            )

            folium.Marker(
                location=[latitude, longitude],
                tooltip=parking_id,
                popup=folium.Popup(
                    f"<b>{parking_id}</b><br>"
                    f"Capacity: {capacity}<br>"
                    f"Marker information: {location_type}",
                    max_width=280
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="parking",
                    prefix="fa"
                )
            ).add_to(parking_map)

        map_col, details_col = st.columns([2, 1])

        with map_col:

            map_result = st_folium(
                parking_map,
                height=480,
                width=None,
                key="parking_map"
            )

        with details_col:

            st.markdown("### Select Parking Location")

            map_selection = st.selectbox(
                "Parking ID",
                parking_options,
                index=parking_options.index(
                    st.session_state.selected_location
                ),
                key="map_location_select"
            )

            st.session_state.selected_location = map_selection

            selected_info = df[
                df["SystemCodeNumber"] == map_selection
            ]

            if not selected_info.empty:

                st.metric(
                    "Parking Capacity",
                    int(selected_info["Capacity"].iloc[0])
                )

                st.write(f"**Parking ID:** {map_selection}")

            st.info(
                "Map note: Some markers may be approximate demo locations "
                "or provisional coordinates. Verify their real-world "
                "positions before using them for navigation."
            )

    else:
        st.warning(
            "The map CSV does not contain recognizable ID, latitude "
            "and longitude columns. Check parking_locations_demo.csv."
        )

else:
    st.warning("Parking map file was not found.")


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def make_prediction(parking_id, selected_date, selected_time):

    location_data = df[
        df["SystemCodeNumber"] == parking_id
    ].copy()

    if location_data.empty:
        raise ValueError("Parking location not found in dataset.")

    capacity = float(location_data["Capacity"].iloc[0])

    hour = selected_time.hour
    minute = selected_time.minute
    day = selected_date.day
    month = selected_date.month
    weekday = selected_date.weekday()

    is_weekend = 1 if weekday >= 5 else 0

    matching_data = location_data[
        (location_data["Weekday"] == weekday) &
        (location_data["Hour"] == hour)
    ]

    if not matching_data.empty:

        previous_occupancy = float(
            matching_data["Occupancy"].median()
        )

    else:

        previous_occupancy = float(
            location_data["Occupancy"].median()
        )

    minute_of_day = hour * 60 + minute

    time_sin = np.sin(
        2 * np.pi * minute_of_day / 1440
    )

    time_cos = np.cos(
        2 * np.pi * minute_of_day / 1440
    )

    weekday_sin = np.sin(
        2 * np.pi * weekday / 7
    )

    weekday_cos = np.cos(
        2 * np.pi * weekday / 7
    )

    month_sin = np.sin(
        2 * np.pi * (month - 1) / 12
    )

    month_cos = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    input_data = pd.DataFrame([{
        "SystemCodeNumber": parking_id,
        "Capacity": capacity,
        "Hour": hour,
        "Minute": minute,
        "Day": day,
        "Month": month,
        "Weekday": weekday,
        "IsWeekend": is_weekend,
        "Previous_Occupancy": previous_occupancy,
        "Time_sin": time_sin,
        "Time_cos": time_cos,
        "Weekday_sin": weekday_sin,
        "Weekday_cos": weekday_cos,
        "Month_sin": month_sin,
        "Month_cos": month_cos
    }])

    prediction = float(
        model.predict(input_data)[0]
    )

    predicted_occupancy = int(
        np.clip(round(prediction), 0, capacity)
    )

    available_spaces = max(
        int(capacity) - predicted_occupancy, 0
    )

    occupancy_percentage = (
        predicted_occupancy / capacity * 100
        if capacity > 0 else 0
    )

    return {
        "capacity": int(capacity),
        "occupancy": predicted_occupancy,
        "available": available_spaces,
        "occupancy_percentage": occupancy_percentage
    }
# --------------------------------------------------
# PARKING PREDICTION
# --------------------------------------------------

st.markdown(
    '<div id="prediction" class="section-title">🔮 Parking Availability Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Choose a parking location and a future date and time "
    "to estimate parking occupancy and available spaces."
)

with st.container(border=True):

    p1, p2, p3 = st.columns(3)

    with p1:

        selected_location = st.selectbox(
            "Select Parking Location",
            parking_options,
            index=parking_options.index(
                st.session_state.selected_location
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

    predict_button = st.button(
        "🚀 Predict Parking Availability",
        use_container_width=True
    )

    if predict_button:

        try:

            result = make_prediction(
                selected_location,
                selected_date,
                selected_time
            )

            st.session_state.prediction_result = result

            st.session_state.prediction_location_saved = selected_location

            st.session_state.prediction_date_saved = selected_date

            st.session_state.prediction_time_saved = selected_time

        except Exception as e:

            st.error(f"Prediction failed: {e}")


if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    st.markdown("### Prediction Results")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Predicted Available Spaces",
            result["available"]
        )

    with r2:

        st.metric(
            "Predicted Occupancy",
            result["occupancy"]
        )

    with r3:

        st.metric(
            "Total Capacity",
            result["capacity"]
        )

    st.progress(
        min(result["occupancy_percentage"] / 100, 1.0)
    )

    st.write(
        f"**Estimated occupancy:** "
        f"{result['occupancy_percentage']:.1f}%"
    )

    st.caption(
        f"Location: {st.session_state.prediction_location_saved} | "
        f"Date: {st.session_state.prediction_date_saved} | "
        f"Time: {st.session_state.prediction_time_saved}"
    )

    if result["available"] == 0:

        st.error(
            "The model predicts that no spaces will be available."
        )

    elif result["occupancy_percentage"] >= 80:

        st.warning(
            "The model predicts high parking occupancy."
        )

    else:

        st.success(
            "The model predicts that parking spaces may be available."
        )

    st.caption(
        "This is a machine-learning estimate based on historical data, "
        "not a live parking availability reading."
    )


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

st.markdown(
    '<div id="analytics" class="section-title">📈 Parking Analytics</div>',
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
        font_color="#e2e8f0"
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
        xaxis_title="Parking Location",
        yaxis_title="Average Occupancy"
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
        font_color="#e2e8f0"
    )

    st.plotly_chart(
        fig_weekday,
        use_container_width=True
    )


with chart4:

    occupancy_data = pd.DataFrame({
        "Status": ["Occupied", "Available"],
        "Spaces": [
            int(latest_occupancy),
            int(latest_available)
        ]
    })

    fig_pie = px.pie(
        occupancy_data,
        names="Status",
        values="Spaces",
        hole=0.55,
        title="Overall Parking Capacity Distribution"
    )

    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        font_color="#e2e8f0"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div id="model-performance" class="section-title">🧠 Model Performance</div>',
    unsafe_allow_html=True
)

if not comparison.empty:

    st.write(
        "Comparison of the trained machine-learning models."
    )

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
        c for c in ["MAE", "RMSE", "R2", "R²", "R2_Score"]
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
            showlegend=False
        )

        st.plotly_chart(
            fig_model,
            use_container_width=True
        )

else:

    st.warning(
        "Model comparison file not found or contains no data."
    )


try:

    best_model_name_path = (
        BASE_DIR / "models" / "best_model_name.txt"
    )

    if best_model_name_path.exists():

        best_model_name = best_model_name_path.read_text().strip()

        st.success(
            f"Best trained model: {best_model_name}"
        )

except Exception:
    pass

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.markdown(
    '<div id="about" class="section-title">ℹ️ About Our Project</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
    <h3>🅿️ Smart Parking Availability Prediction Using Machine Learning</h3>
    <p>
        Our project predicts parking occupancy using historical parking data.<br>
        It uses machine learning algorithms to estimate available parking spaces.<br>
        Users can select a parking location, date and time to get predictions.<br>
        An interactive map helps users explore different parking locations.<br>
        Analytics and model performance charts provide insights into parking patterns.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Smart Parking AI | Machine Learning Project<br>
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Smart Parking AI | Machine Learning Project<br>
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)