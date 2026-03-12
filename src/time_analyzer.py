import pandas as pd

def safest_time_for_city(city_name):

    df = pd.read_csv("data/processed/safety_scores.csv")
    

    city_df = df[df["city"] == city_name]

    if city_df.empty:
        return None

    time_scores = city_df.groupby("time_of_day")["safety_score"].mean()

    safest_time = time_scores.idxmax()

    return {
        "scores": time_scores.to_dict(),
        "safest_time_to_travel": safest_time
    }


if __name__ == "__main__":
    print(safest_time_for_city("Delhi_City_1"))