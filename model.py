import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
df = pd.read_csv("churn_predict.csv")
df.sample(frac=1, random_state=seed)

encode = ['PreferedOrderCat', 'Gender', 'MaritalStatus']

for col in encode:
    dummy = pd.get_dummies(df[col])
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# selecting features and target data
X = df.drop('Churn', axis=1)
y = df[['Churn']]

# X.to_csv('X_sample.csv', index=False)
# y.to_csv('y_sample.csv')

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)


# Create a instance of XGBoost classifier
xgc = xgb.XGBClassifier()

# train the classifier on the training data
xgc.fit(X_train.values, y_train.values)

# predict on the test set
y_pred = xgc.predict(X_test.values)

y_pred_prob = xgc.predict_proba(X_test.values)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(xgc, "xgc_model.sav")