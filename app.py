import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib


st.title('Customer Churn Prediction')
st.markdown('Use customers features to predict whether this customer is going to churn or not')

st.header("Customer Profile")
col1, col2 = st.columns(2)

with col1:
    Tenure = st.slider('Tenure', 0, 31, 1)
    Complain = st.slider('Complain', 0, 1, 1)
    Cashback = st.slider('CashbackAmount', 110, 324, 1)
    SatisfactionScore = st.slider('SatisfactionScore', 1, 5, 1)
    DaysSinceLastOrder = st.slider('DaySinceLastOrder', 0, 30, 1)

with col2:
    pass

def predict(data):
    clf = joblib.load("xgc_model.sav")
    return clf.predict(data)

def predict_proba(data):
    clf = joblib.load("xgc_model.sav")
    return clf.predict_proba(data)



st.text('')
if st.button("Customer churn prediction"):
    result = predict(
        np.array([[Tenure, Complain, Cashback, SatisfactionScore, DaysSinceLastOrder]]))
    st.text(result[0])

    prob_result = predict_proba(
        np.array([[Tenure, Complain, Cashback, SatisfactionScore, DaysSinceLastOrder]]))
    st.text(str(round(prob_result[0][1]*100,2))+"%")    


st.text('')
st.text('')
st.markdown(
    '`Create by` [Zhengmian Chang | ')





