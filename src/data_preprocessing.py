# PREPROCESSING-cleaning the dataset so that we are able to remove all buggy values,NULL, duplicate values etc.

import pandas as pd
df = pd.read_csv("../data/raw/sakhi_she_safe_pan_india_advanced_dataset.csv")
print(df.shape) # shape of the dataset
print(df.head(5)) # first 5 rows of dataset
df.columns=df.columns.str.lower() #make column naming consistent(lowercase)
print(df.isnull().sum()) # sees if any column has null values
df=df.drop_duplicates()

# convert columns to numeric values

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["population_density"] = pd.to_numeric(df["population_density"], errors="coerce")
df["police_presence_index"] = pd.to_numeric(df["police_presence_index"], errors="coerce")
df["street_lighting_index"] = pd.to_numeric(df["street_lighting_index"], errors="coerce")
df["repeat_offender_rate"] = pd.to_numeric(df["repeat_offender_rate"], errors="coerce")
df["crime_count"] = pd.to_numeric(df["crime_count"], errors="coerce")
df=df.dropna() #removes rows where conversion failed
print("final shape-",df.shape) #see the final shape
df.to_csv("../data/processed/cleaned_data.csv", index=False) # puts the processed data into a new csv file in processed folder
print("Preprocessing complete.")



