#CONVERT RISK INTO UNDERSTANDABLE CATEGORIES

import pandas as pd
df=pd.read_csv("data/processed/clustered_data.csv")
df["safety_score"]=(1-df["final_risk_index"])*100 #creates safety score

def safety_category(safety_score):
    if safety_score>=80:
        return "Safe"
    elif safety_score>=50:
        return "Caution"
    else:
        return "High Risk"
df["safety_category"]=df["safety_score"].apply(safety_category) #applies the function to the dataframe
df.to_csv("data/processed/safety_scores.csv",index=False)


#TIME BASED RISK SIMULATION
def adjust_risk_by_time(final_risk_value,time_of_day):
    if time_of_day=="Night":
        return final_risk_value+0.3
    elif time_of_day=="Evening":
        return final_risk_value+0.15
    elif time_of_day=="Afternoon":
        return final_risk_value+0.02
    else:
        return final_risk_value
    
print(df[["city","final_risk_index","safety_score","safety_category"]].head())

    
