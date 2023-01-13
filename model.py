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

# selecting features and target data
X = df[['Tenure', 'Complain', 'SatisfactionScore', 'CashbackAmount', 'DaySinceLastOrder']]
y = df[['Churn']]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
# clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
# clf.fit(X_train.values, y_train.values)

# # predict on the test set
# y_pred = clf.predict(X_test.values)

# # calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# # save the model to disk
# joblib.dump(clf, "rf_model.sav")


# Create a instance of XGBoost classifier
xgc = xgb.XGBClassifier()

# train the classifier on the training data
xgc.fit(X_train.values, y_train.values)

# predict on the test set
y_pred = xgc.predict(X_test.values)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(xgc, "xgc_model.sav")