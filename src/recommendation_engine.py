import pandas as pd


def recommend_transport(city_name):

    # load dataset
    df = pd.read_csv("data/processed/safety_scores.csv")

    # filter rows for the selected city
    city_df = df[df["city"] == city_name]

    if city_df.empty:
        return None

    # calculate average indicators
    avg_risk = float(city_df["risk_index"].mean())
    avg_lighting = float(city_df["street_lighting_index"].mean())
    avg_police = float(city_df["police_presence_index"].mean())

    # find most common time of day
    time_of_day = city_df["time_of_day"].mode()[0]

    # decision logic
    if avg_risk > 0.7 or avg_lighting < 0.4:
        transport = "Cab"

    elif avg_risk > 0.4 or avg_police < 0.4:
        transport = "Public Transport"

    else:
        transport = "Walk"

    # extra safety rule for night
    if time_of_day == "Night" and transport == "Walk":
        transport = "Cab"

    return {
        "average_risk": avg_risk,
        "lighting_index": avg_lighting,
        "police_index": avg_police,
        "dominant_time_of_day": time_of_day,
        "recommended_transport": transport
    }


# quick test
if __name__ == "__main__":
    print(recommend_transport("Delhi_City_1"))