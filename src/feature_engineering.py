import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#load data
df = pd.read_csv("data/processed/cleaned_data.csv")
print(df["crime_type"].unique()) #shows unique crime categories in our dataset

#assigns seriousness levels to each crime type
weights = {
    "Rape": 5,
    "Human Trafficking": 5,
    "Dowry Death": 5,
    "Kidnapping": 4,
    "Domestic Violence": 4,
    "Assault": 3,
    "Stalking": 3,
    "Harassment": 2,
    "Eve Teasing": 2,
    "Cyber Crime": 2
}

#FEATURE 1
df["severity_weight"]=df["crime_type"].map(weights) #create a new column that gives the severity number to each row based on weight
print(df[["crime_type","severity_weight"]].head())

#FEATURE 2
df["severity_score"]=df["crime_count"]*df["severity_weight"] #multiplies how many crimes occured by how serious they are
print(df[["crime_count","severity_weight","severity_score"]].head())

#assigns seriousness levels to each time of the day
times={
    "Morning":2,
    "Afternoon":3,
    "Evening":5,
    "Night":7
}

#FEATURE 3
df["time_severity"]=df["time_of_day"].map(times) #create a new column that gives the severity number to each row based on times
print(df[["time_of_day","time_severity"]].head())

#SCALING TO THE SAME RANGE
scaler=MinMaxScaler()

df[["norm_crime_count", "norm_severity_score", "norm_population_density","norm_police_presence_index","norm_street_lighting_index","norm_repeat_offender_rate","norm_time_severity"]] = scaler.fit_transform(
    df[["crime_count", "severity_score", "population_density","police_presence_index","street_lighting_index","repeat_offender_rate","time_severity"]]
) # normalizes them to fit to the same range for simplicity and ease

df["police_risk"]=1-df["norm_police_presence_index"]
df["lighting_risk"]=1-df["norm_street_lighting_index"]
print(df[["police_risk","lighting_risk"]].describe())


print(df[["norm_crime_count", "norm_severity_score","norm_population_density","norm_police_presence_index","norm_street_lighting_index","norm_repeat_offender_rate","norm_time_severity"]].describe())

#RISK INDEX
df["risk_index"] = (
    0.25 * df["norm_crime_count"] +
    0.25 * df["norm_severity_score"]+
    0.15 * df["norm_repeat_offender_rate"]+
    0.10 * df["norm_population_density"]+
    0.15 * df["norm_time_severity"]+
    0.05 * df["police_risk"] +
    0.05 * df["lighting_risk"]
)

df["risk_index"]=df["risk_index"].clip(0,1) # ensures that the range is not more than than 1 even if rounding happens

print(df["risk_index"].describe())

risk_scaler=MinMaxScaler()
df[["final_risk_index"]] = risk_scaler.fit_transform(df[["risk_index"]]) # normalizing risk_index so we dont get any negative value
print(df["final_risk_index"].describe())

df["risk_level"]=pd.cut(df["final_risk_index"],bins=[-0.01,0.33,0.66,1],labels=["Low Risk",'Medium Risk','High Risk'])
print(df["risk_level"].value_counts())

df.to_csv("data/processed/final_features.csv", index=False)# create a new csv file with the new features obtained by performing feature engineering

# Quick check
print(df.head())
print(df.columns)
