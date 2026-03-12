"""
ACCURACY=Correct Predictions / Total Predictions
If model predicted 154 correct out of 200:
Accuracy = 154 / 200 = 0.77


PRECISION=True Positives / (True Positives + False Positives)
Precision answers when the model says something is High Risk… how often is it actually High Risk?
Model predicts 50 cases as High Risk.
But only 40 were actually High Risk.
Precision = 40 / 50 = 0.80
So 20% were false alarms.


RECALL=True Positives / (True Positives + False Negatives)
Recall answers Out of all actual High Risk cases, how many did we correctly detect?
There are 60 actual High Risk cases.
Model caught 45 of them.
Recall = 45 / 60 = 0.75
So it missed 15 dangerous cases.
In women safety context, recall is VERY important.
Missing dangerous zones is worse than false alarms.


F1 SCORE=2 * (Precision * Recall) / (Precision + Recall)
F1 balances precision and recall.
If precision is high but recall is low → F1 drops.
If recall high but precision low → F1 drops.
F1 rewards balance."""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # our baseline classification model
from sklearn.ensemble import RandomForestClassifier  # our main model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib  # to save our model as .pkl


df=pd.read_csv("data/processed/final_features.csv")

def risk_category(final_risk_index):# help to categorise regions based on risk
    if final_risk_index<=0.33:
        return 0
    elif final_risk_index<=0.66:  # we just keep simple 3 categories to keep to human empathetic because we are dealing with a real life scenario and we dont want to over complicate anything
        return 1
    else:
        return 2

# create a target column
df["risk_category"]=df["final_risk_index"].apply(risk_category)

print(df["risk_category"].value_counts())

# remove certain columns from our dataframe so that our model doesn't see the values in the X labels and learn from it(it prevents data leakage), also we remove string based data for clarity of prediction
X = df.drop(columns=["risk_category",
 "final_risk_index",
 "risk_index",
 "State",
 "City",
 "Crime Type",
 "time_category"],errors="ignore")

X = X.select_dtypes(include=["int64", "float64"])

print(X.columns)

joblib.dump(X.columns,"models/model_features.pkl") #saving the feature names 

y=df["risk_category"]# target

#TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
) # test_size=0.2 → 80–20 split,random_state=42 → reproducibility,stratify=y → keeps class balance in train & test

#Check shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#LOGISTIC REGRESSION
log_model=LogisticRegression(max_iter=1000) #max_iter is 1000 because logistic regression doesnt converge with default 100 sometimes
log_model.fit(X_train,y_train) #trains the model
y_pred_log=log_model.predict(X_test) #predicts on the test data

print("Logistic Regression Results:")

#Metrics-Accuracy_score,Precision,Recall,F1 Score based on Logistic Regression
print("ACCURACY:",accuracy_score(y_test,y_pred_log))
print("PRECISION:",precision_score(y_test,y_pred_log,average="weighted"))
print("RECALL:",recall_score(y_test,y_pred_log,average="weighted"))
print("F1 SCORE:",f1_score(y_test,y_pred_log,average="weighted"))
print(classification_report(y_test,y_pred_log))

#RANDOM FOREST
rf_model=RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42) #random forest initialization
"""n_estimators=no. of decision trees in the forest
max_depth=how deep each decision tree can grow(here depth is unlimited)
random_state=controls randomness so that we dont get different results every run"""
rf_model.fit(X_train,y_train) # trains the random forest model
y_pred_rf=rf_model.predict(X_test) #predicts on test data


print("Random Forest Results:")

#Metrics-Accuracy_score,Precision,Recall,F1 Score based on Random Forest
print("ACCURACY:",accuracy_score(y_test,y_pred_rf))
print("PRECISION:",precision_score(y_test,y_pred_rf,average="weighted"))
print("RECALL:",recall_score(y_test,y_pred_rf,average="weighted"))
print("F1 SCORE:",f1_score(y_test,y_pred_rf,average="weighted"))
print(classification_report(y_test,y_pred_rf))


# saving our trained random forest model in the form of a .pkl file
joblib.dump(rf_model,"models/safety_risk_rf_model.pkl")



""" our application app.py then loads the .pkl file and makes the predictions instantly """