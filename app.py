import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.time_analyzer import safest_time_for_city
from src.recommendation_engine import recommend_transport
from src.route_optimizer import find_safest_route


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="SAKHI SHE SAFE",
    layout="wide"
)


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/processed/final_features.csv")

    df["city_clean"] = df["city"].astype(str).str.strip().str.lower()
    df["time_clean"] = df["time_of_day"].astype(str).str.strip().str.lower()

    return df


@st.cache_resource
def load_model():

    model = joblib.load("models/safety_risk_rf_model.pkl")
    model_features = joblib.load("models/model_features.pkl")

    return model, model_features


df = load_data()
model, model_features = load_model()


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.title("SAKHI SHE SAFE")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Location Safety Check",
        "Travel Recommendation",
        "Best Time To Travel",
        "Safe Route Planner",
        "Crime Risk Map",
        "Crime Trend Analysis",
        "Community Safety Feedback",
        "Emergency Alert"
    ]
)


# -----------------------------------------------------
# DASHBOARD
# -----------------------------------------------------

def dashboard():

    st.title("SAKHI SHE SAFE")
    st.subheader("City Safety Overview")

    cities = sorted(df["city"].unique())

    selected_city = st.selectbox("Select City", cities)

    city_df = df[df["city"] == selected_city]

    avg_safety = round(city_df["severity_score"].mean(), 2)

    dangerous_time = city_df.groupby("time_of_day")["crime_count"].mean().idxmax()

    transport = recommend_transport(selected_city)["recommended_transport"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Safety Score (City)", avg_safety)

    col2.metric("Most Dangerous Time", dangerous_time)

    col3.metric("Recommended Transport", transport)


# -----------------------------------------------------
# LOCATION SAFETY CHECK
# -----------------------------------------------------

def location_safety_check():

    st.header("Location Safety Check")

    cities = sorted(df["city_clean"].unique())
    times = sorted(df["time_clean"].unique())

    col1, col2 = st.columns(2)

    with col1:
        city_input = st.selectbox("Select City", cities)

    with col2:
        time_input = st.selectbox("Select Time", times)

    if st.button("Check Safety"):

        filtered = df[
            (df["city_clean"] == city_input) &
            (df["time_clean"] == time_input)
        ]

        if filtered.empty:

            city_only = df[df["city_clean"] == city_input]

            st.warning("Exact time data unavailable. Using city average.")

            sample = city_only.mean(numeric_only=True)

        else:
            sample = filtered.iloc[0]

        X = sample[model_features].values.reshape(1, -1)

        risk_probability = model.predict_proba(X)[0][1]

        safety_score = round((1 - risk_probability) * 100, 2)

        if safety_score >= 75:
            risk_level = "Low Risk"
        elif safety_score >= 50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        st.success(f"Safety Score: {safety_score}")
        st.warning(f"Risk Level: {risk_level}")


# -----------------------------------------------------
# TRAVEL RECOMMENDATION
# -----------------------------------------------------

def travel_recommendation():

    st.header("Travel Recommendation")

    cities = sorted(df["city"].unique())

    city = st.selectbox("Select City", cities)

    if st.button("Get Recommendation"):

        result = recommend_transport(city)

        st.write("Average Risk:", result["average_risk"])
        st.write("Lighting Index:", result["lighting_index"])
        st.write("Police Presence:", result["police_index"])

        st.success(f"Recommended Transport: {result['recommended_transport']}")


# -----------------------------------------------------
# BEST TIME TO TRAVEL
# -----------------------------------------------------

def best_time_to_travel():

    st.header("Best Time To Travel")

    cities = sorted(df["city"].unique())

    city = st.selectbox("Select City", cities)

    if st.button("Analyze"):

        result = safest_time_for_city(city)

        st.write("Safety Score by Time:")
        st.write(result["scores"])

        st.success(f"Safest Time: {result['safest_time_to_travel']}")


# -----------------------------------------------------
# ROUTE PLANNER
# -----------------------------------------------------

def safe_route_planner():

    st.header("Safe Route Planner")

    cities = sorted(df["city"].unique())

    col1, col2 = st.columns(2)

    with col1:
        start = st.selectbox("Start City", cities)

    with col2:
        end = st.selectbox("Destination City", cities)

    if st.button("Find Route"):

        route = find_safest_route(start, end)

        if route is None:
            st.error("Route not found.")
        else:
            st.success("Suggested Safe Route")
            st.write(" → ".join(route))


# -----------------------------------------------------
# CRIME HEATMAP
# -----------------------------------------------------

def crime_heatmap():

    st.header("Crime Risk Map")

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="risk_index",
        size="risk_index",
        color_continuous_scale=["#ffcccc", "#ff0000"],
        zoom=4,
        height=600
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        coloraxis_colorbar=dict(title="Crime Risk Intensity")
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------
# CRIME TREND ANALYSIS
# -----------------------------------------------------

def crime_trend():

    st.header("Crime Trend Analysis")

    trend = df.groupby("year")["severity_score"].mean()

    st.line_chart(trend)


# -----------------------------------------------------
# COMMUNITY FEEDBACK
# -----------------------------------------------------

def community_feedback():

    st.header("Community Safety Feedback")

    city = st.text_input("City")
    rating = st.slider("Safety Rating", 1, 5)
    comment = st.text_area("Comments")

    if st.button("Submit Feedback"):

        feedback = pd.DataFrame(
            [[city, rating, comment]],
            columns=["city", "rating", "comment"]
        )

        feedback.to_csv(
            "data/feedback/community_feedback.csv",
            mode="a",
            header=False,
            index=False
        )

        st.success("Feedback submitted")


# -----------------------------------------------------
# EMERGENCY ALERT
# -----------------------------------------------------

def emergency_alert():

    st.header("Emergency Alert")

    if st.button("Trigger Alert"):

        st.error("Emergency alert triggered")

        st.write("Sending alerts to emergency contacts")
        st.write("Sharing live location")
        st.write("Notifying authorities")

        st.success("Alert simulation complete")


# -----------------------------------------------------
# PAGE ROUTING
# -----------------------------------------------------

if page == "Dashboard":
    dashboard()

elif page == "Location Safety Check":
    location_safety_check()

elif page == "Travel Recommendation":
    travel_recommendation()

elif page == "Best Time To Travel":
    best_time_to_travel()

elif page == "Safe Route Planner":
    safe_route_planner()

elif page == "Crime Risk Map":
    crime_heatmap()

elif page == "Crime Trend Analysis":
    crime_trend()

elif page == "Community Safety Feedback":
    community_feedback()

elif page == "Emergency Alert":
    emergency_alert()





