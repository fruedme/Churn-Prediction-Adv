import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib


st.title('Customer Churn Prediction')
st.markdown('Use customers features to predict whether this customer is going to churn or not')

st.header("Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    Tenure = st.slider('Tenure', 0, 31, 1)
    Complain = st.slider('Complain', 0, 1, 1)
    Cashback = st.slider('CashbackAmount', 110, 324, 1)
    SatisfactionScore = st.slider('SatisfactionScore', 1, 5, 1)
    DaysSinceLastOrder = st.slider('DaySinceLastOrder', 0, 30, 1)

with col2:
    pass

with col3:
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    MaritalStatus = st.selectbox("MaritalStatus", ('Single', 'Divorced', 'Married'))
    PreferedOrderCat = st.selectbox('PreferedOrderCat', ('Fashion', 'Grocery', 'Laptop & Accessory', 'Mobile', 'Mobile Phone'))



def predict(data):
    clf = joblib.load("xgc_model.sav")
    return clf.predict(data)

def predict_proba(data):
    clf = joblib.load("xgc_model.sav")
    return clf.predict_proba(data)

df = pd.read_csv("churn_predict.csv")
df.sample(frac=1, random_state=42)

encode = ['PreferedOrderCat', 'Gender', 'MaritalStatus']

for col in encode:
    dummy = pd.get_dummies(df[col])
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# selecting features and target data
X = df.drop('Churn', axis=1)
y = df[['Churn']]

st.text('')
if st.button("Customer churn prediction"):
    # result = predict(
    #     np.array([[Tenure, Complain, Cashback, SatisfactionScore, DaysSinceLastOrder]]))
    # st.text(result[0])

    prob_result = predict_proba(np.array(X.columns))
        # np.array([[Tenure, Complain, Cashback, SatisfactionScore, DaysSinceLastOrder]]))
    st.text("Possibility of churn: "+str(round(prob_result[0][1]*100,2))+"%")    


st.text('')
st.text('')
st.markdown(
    '`Create by` [Zhengmian Chang | ')





